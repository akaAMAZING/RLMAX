//! Compiles RLMAX CUDA sources to fat binaries from a build script.
//!
//! Measured on CUDA 13.2 (DESIGN §10a); any toolkit that can build the card is accepted, with a warning. Configuration comes from the environment so the repo
//! carries no machine paths:
//!
//! - `RLMAX_CUDA_PATH` (else `CUDA_PATH`): toolkit root containing `bin/nvcc`.
//! - `RLMAX_CUDA_ARCHS`: comma list of SASS targets, `compute_XY` entries add PTX. Default: the cards `nvidia-smi`
//!   reports on this machine; with none reported, [`ALL_ARCHS`].
//! - `RLMAX_CCBIN` (Windows only, optional): directory of `cl.exe`; otherwise found through the registry.

use std::path::{Path, PathBuf};
use std::process::Command;

pub const PINNED_CUDA: &str = "13.2";

/// Whether a toolkit release can build these targets: 12.4 for the kernels' `__maxnreg__` budgets, 12.8 for sm_120.
fn toolkit_fits(release: &str, archs: &[String]) -> Result<(), String> {
    let mut parts = release.split('.').map(|p| p.parse::<u32>());
    let (Some(Ok(major)), Some(Ok(minor))) = (parts.next(), parts.next()) else {
        return Err(format!("unrecognised release {release}"));
    };
    let version = (major, minor);
    if version < (12, 4) {
        return Err("RLMAX needs CUDA 12.4 or newer".into());
    }
    if version < (12, 8) && archs.iter().any(|a| a.ends_with("120")) {
        return Err("this card needs CUDA 12.8 or newer".into());
    }
    Ok(())
}

pub fn cuda_root() -> PathBuf {
    let root = std::env::var_os("RLMAX_CUDA_PATH")
        .or_else(|| std::env::var_os("CUDA_PATH"))
        .map(PathBuf::from)
        .expect("set RLMAX_CUDA_PATH (or CUDA_PATH) to a CUDA 13.2 toolkit root");
    assert!(nvcc_in(&root).exists(), "no nvcc under {}", root.display());
    root
}

fn nvcc_in(root: &Path) -> PathBuf {
    root.join("bin").join(if cfg!(windows) { "nvcc.exe" } else { "nvcc" })
}

/// The `release X.Y` string reported by the toolkit's nvcc.
pub fn nvcc_release(root: &Path) -> String {
    let out = Command::new(nvcc_in(root)).arg("--version").output().expect("run nvcc --version");
    let text = String::from_utf8_lossy(&out.stdout);
    text.split("release ")
        .nth(1)
        .and_then(|s| s.split(',').next())
        .map(str::to_owned)
        .unwrap_or_else(|| panic!("unrecognised nvcc --version output:\n{text}"))
}

/// The targets of a build for any card: slower to compile, so only used when the machine's card can't be read.
pub const ALL_ARCHS: &str = "86,89,120,compute_120";

pub fn archs() -> Vec<String> {
    let list = std::env::var("RLMAX_CUDA_ARCHS").unwrap_or_else(|_| detected_archs());
    list.split(',').map(|s| s.trim().to_owned()).filter(|s| !s.is_empty()).collect()
}

/// The architectures of the cards on this machine, or [`ALL_ARCHS`] if `nvidia-smi` can't say.
fn detected_archs() -> String {
    Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader,nounits"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| archs_from_compute_caps(&String::from_utf8_lossy(&o.stdout)))
        .unwrap_or_else(|| ALL_ARCHS.into())
}

/// `12.0` lines to `120`, one entry per distinct card. `None` if any line isn't a compute capability.
fn archs_from_compute_caps(text: &str) -> Option<String> {
    let mut archs: Vec<String> = Vec::new();
    for line in text.lines().map(str::trim).filter(|l| !l.is_empty()) {
        let (major, minor) = line.split_once('.')?;
        if !major.bytes().chain(minor.bytes()).all(|b| b.is_ascii_digit()) {
            return None;
        }
        let arch = format!("{major}{minor}");
        if !archs.contains(&arch) {
            archs.push(arch);
        }
    }
    (!archs.is_empty()).then(|| archs.join(","))
}

fn gencode(arch: &str) -> String {
    match arch.strip_prefix("compute_") {
        Some(v) => format!("arch=compute_{v},code=compute_{v}"),
        None => format!("arch=compute_{arch},code=sm_{arch}"),
    }
}

#[cfg(windows)]
fn ccbin() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("RLMAX_CCBIN") {
        return Some(PathBuf::from(dir));
    }
    cc::windows_registry::find_tool("x86_64-pc-windows-msvc", "cl.exe")
        .and_then(|t| t.path().parent().map(Path::to_path_buf))
}

#[cfg(not(windows))]
fn ccbin() -> Option<PathBuf> {
    None
}

/// Compiles `src` to a fat binary at `out`. `defines` become `-DNAME=VALUE`.
pub fn fatbin(src: &Path, out: &Path, includes: &[&Path], defines: &[(&str, &str)]) {
    fatbin_with(src, out, includes, defines, &[]);
}

/// [`fatbin`] with extra nvcc arguments (e.g. the exact-math flags of the oracle build). ptxas' resource report (registers,
/// stack frame and spills per kernel and arch) is written beside the output as `<out>.resources.txt`.
pub fn fatbin_with(src: &Path, out: &Path, includes: &[&Path], defines: &[(&str, &str)], extra: &[&str]) {
    let root = cuda_root();
    let release = nvcc_release(&root);
    if let Err(why) = toolkit_fits(&release, &archs()) {
        panic!("CUDA {release} at {}: {why}", root.display());
    }
    if release != PINNED_CUDA {
        println!("cargo:warning=CUDA {release}: RLMAX is measured on {PINNED_CUDA}; run its check on this card");
    }
    let mut cmd = Command::new(nvcc_in(&root));
    cmd.args(["--fatbin", "-std=c++17", "-O3", "-lineinfo", "--resource-usage"]);
    split_compile(&mut cmd);
    if let Some(dir) = ccbin() {
        cmd.arg("-ccbin").arg(dir);
    }
    for arch in archs() {
        cmd.arg("-gencode").arg(gencode(&arch));
    }
    for inc in includes {
        cmd.arg("-I").arg(inc);
    }
    for (k, v) in defines {
        cmd.arg(format!("-D{k}={v}"));
    }
    cmd.args(extra);
    cmd.arg(src);
    let mut report = out.as_os_str().to_owned();
    report.push(".resources.txt");
    println!("cargo:rerun-if-changed={}", src.display());
    for var in ["RLMAX_CUDA_PATH", "CUDA_PATH", "RLMAX_CUDA_ARCHS", "RLMAX_CCBIN"] {
        println!("cargo:rerun-if-env-changed={var}");
    }
    let cached = cache_dir().map(|dir| {
        let key = format!("{:016x}", image_key(&release, &cmd, src, includes));
        (dir.join(format!("{key}.fatbin")), dir.join(format!("{key}.resources.txt")))
    });
    if let Some((image, log)) = &cached
        && image.is_file()
        && log.is_file()
    {
        std::fs::copy(image, out).expect("copy the cached image");
        std::fs::copy(log, &report).expect("copy the cached resource report");
        return;
    }
    cmd.arg("-o").arg(out);
    let o = cmd.output().expect("spawn nvcc");
    let log = format!("{}{}", String::from_utf8_lossy(&o.stdout), String::from_utf8_lossy(&o.stderr));
    assert!(o.status.success(), "nvcc failed on {}: {log}", src.display());
    std::fs::write(&report, &log).expect("write the resource report");
    if let Some((image, cached_log)) = &cached {
        // written under temporary names and renamed, so a build beside this one never reads half a file
        let dir = image.parent().unwrap();
        std::fs::create_dir_all(dir).expect("create the kernel cache");
        let tmp = |p: &Path| p.with_extension(format!("tmp{}", std::process::id()));
        std::fs::copy(out, tmp(image)).expect("store the image");
        std::fs::write(tmp(cached_log), &log).expect("store the resource report");
        std::fs::rename(tmp(cached_log), cached_log).expect("store the resource report");
        std::fs::rename(tmp(image), image).expect("store the image");
    }
}

/// Where compiled images are kept between builds: `RLMAX_KERNEL_CACHE`, else `$RLMAX_DATA/kernel_cache`; none if
/// neither is set. Every tool that links the rlmax crate is its own cargo
/// workspace and runs this build script itself; with the cache they share one compile of each image.
/// `RLMAX_KERNEL_CACHE=off` disables it.
fn cache_dir() -> Option<PathBuf> {
    println!("cargo:rerun-if-env-changed=RLMAX_KERNEL_CACHE");
    println!("cargo:rerun-if-env-changed=RLMAX_DATA");
    match std::env::var_os("RLMAX_KERNEL_CACHE") {
        Some(v) if v == "off" => None,
        Some(v) => Some(PathBuf::from(v)),
        None => std::env::var_os("RLMAX_DATA").map(|d| PathBuf::from(d).join("kernel_cache")),
    }
}

/// FNV-1a 64 over everything that decides an image: the toolkit release, nvcc's arguments (the host compiler's path
/// among them: on Windows it preprocesses the device code too) and every source the compile can read: `src` and the
/// headers (.h, .cuh, .inc, .cu) under the include directories, with their paths relative to those directories.
fn image_key(release: &str, cmd: &Command, src: &Path, includes: &[&Path]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    let mut mix = |bytes: &[u8]| {
        for &b in bytes.iter().chain(&[0xff]) {
            h = (h ^ b as u64).wrapping_mul(0x100000001b3);
        }
    };
    mix(release.as_bytes());
    // the source and include paths count by what they hold, not where they are, so two checkouts of the same sources
    // share their images
    let ours = |a: &std::ffi::OsStr| Path::new(a) == src || includes.iter().any(|i| Path::new(a) == *i);
    for arg in cmd.get_args().filter(|a| !ours(a)) {
        mix(arg.as_encoded_bytes());
    }
    mix(&std::fs::read(src).expect("read the kernel source"));
    for inc in includes {
        let mut files = Vec::new();
        let mut dirs = vec![inc.to_path_buf()];
        while let Some(d) = dirs.pop() {
            for e in std::fs::read_dir(&d).expect("read an include directory").flatten() {
                let p = e.path();
                if p.is_dir() {
                    dirs.push(p);
                } else if p.extension().is_some_and(|x| ["h", "cuh", "inc", "cu"].iter().any(|k| x == *k)) {
                    files.push(p);
                }
            }
        }
        files.sort();
        for f in files {
            mix(f.strip_prefix(inc).unwrap().to_string_lossy().replace('\\', "/").as_bytes());
            mix(&std::fs::read(&f).expect("read a header"));
        }
    }
    h
}

/// `RLMAX_NVCC_SPLIT=N`: nvcc optimises and assembles up to N kernels at once (`-split-compile`). Every tick kernel is
/// the whole physics inlined, so each concurrent ptxas holds several GB: the default is one at a time.
pub fn split_jobs() -> u32 {
    println!("cargo:rerun-if-env-changed=RLMAX_NVCC_SPLIT");
    std::env::var("RLMAX_NVCC_SPLIT").ok().and_then(|v| v.trim().parse().ok()).filter(|&n: &u32| n > 0).unwrap_or(1)
}

fn split_compile(cmd: &mut Command) {
    let n = split_jobs();
    if n > 1 {
        cmd.arg(format!("-split-compile={n}"));
    }
}

/// Compiles `src` to a cubin for one SASS `arch` (e.g. "86") with ptxas' resource report; returns the report.
pub fn cubin(src: &Path, out: &Path, includes: &[&Path], defines: &[(&str, &str)], extra: &[&str], arch: &str) -> String {
    let root = cuda_root();
    let mut cmd = Command::new(nvcc_in(&root));
    cmd.args(["--cubin", "-std=c++17", "-O3", "--resource-usage"]).arg(format!("-arch=sm_{arch}"));
    split_compile(&mut cmd);
    if let Some(dir) = ccbin() {
        cmd.arg("-ccbin").arg(dir);
    }
    for inc in includes {
        cmd.arg("-I").arg(inc);
    }
    for (k, v) in defines {
        cmd.arg(format!("-D{k}={v}"));
    }
    cmd.args(extra).arg(src).arg("-o").arg(out);
    let o = cmd.output().expect("spawn nvcc");
    let log = format!("{}{}", String::from_utf8_lossy(&o.stdout), String::from_utf8_lossy(&o.stderr));
    assert!(o.status.success(), "nvcc failed on {}: {log}", src.display());
    log
}

/// A tool from the toolkit's bin directory (e.g. "cuobjdump").
pub fn tool(name: &str) -> PathBuf {
    cuda_root().join("bin").join(if cfg!(windows) { format!("{name}.exe") } else { name.to_owned() })
}

/// Directory holding the driver import library (`cuda.lib`) or the `libcuda.so` link stub.
pub fn driver_lib_dir() -> PathBuf {
    let root = cuda_root();
    if cfg!(windows) { root.join("lib").join("x64") } else { root.join("lib64").join("stubs") }
}

#[cfg(test)]
mod tests {
    use super::archs_from_compute_caps as caps;

    #[test]
    fn compute_caps_become_archs() {
        assert_eq!(caps("12.0
").as_deref(), Some("120"));
        assert_eq!(caps("8.6
").as_deref(), Some("86"));
        assert_eq!(caps("8.6
12.0
8.6
").as_deref(), Some("86,120"));
        assert_eq!(caps(""), None);
        assert_eq!(caps("[N/A]
"), None);
    }

    /// Two checkouts of the same sources share an image key; any header's change or a different argument changes it.
    #[test]
    fn image_key_follows_the_sources() {
        let root = std::env::temp_dir().join(format!("rlmax_key_{}", std::process::id()));
        let tree = |name: &str, header: &str| {
            let inc = root.join(name).join("src");
            std::fs::create_dir_all(inc.join("physics")).unwrap();
            std::fs::write(inc.join("physics/tick.h"), header).unwrap();
            std::fs::write(inc.join("step.cu"), "#include \"physics/tick.h\"").unwrap();
            inc
        };
        let key = |inc: &std::path::Path, define: &str| {
            let mut cmd = std::process::Command::new("nvcc");
            cmd.arg("-I").arg(inc).arg(define).arg(inc.join("step.cu"));
            super::image_key("13.2", &cmd, &inc.join("step.cu"), &[inc])
        };
        let (a, b, c) = (tree("a", "int x;"), tree("b", "int x;"), tree("c", "int y;"));
        assert_eq!(key(&a, "-DX=1"), key(&b, "-DX=1"), "same sources, another path");
        assert_ne!(key(&a, "-DX=1"), key(&c, "-DX=1"), "a header changed");
        assert_ne!(key(&a, "-DX=1"), key(&a, "-DX=2"), "an argument changed");
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn toolkit_rule() {
        let a = |s: &str| s.split(',').map(str::to_owned).collect::<Vec<_>>();
        assert!(super::toolkit_fits("13.2", &a("120")).is_ok());
        assert!(super::toolkit_fits("12.8", &a("120,compute_120")).is_ok());
        assert!(super::toolkit_fits("12.6", &a("80")).is_ok());
        assert!(super::toolkit_fits("12.6", &a("86,120")).is_err());
        assert!(super::toolkit_fits("12.3", &a("80")).is_err());
        assert!(super::toolkit_fits("11.8", &a("80")).is_err());
        assert!(super::toolkit_fits("x", &a("80")).is_err());
    }
}
