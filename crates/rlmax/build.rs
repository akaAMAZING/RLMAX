use std::path::PathBuf;

fn main() {
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let include = repo.join("src");

    // RLMAX_CPU_ONLY=1: empty kernel images (GPU calls fail at load), for fast CPU-only iteration; never a gate build
    println!("cargo:rerun-if-env-changed=RLMAX_CPU_ONLY");
    let cpu_only = std::env::var("RLMAX_CPU_ONLY").is_ok_and(|v| v == "1");
    // under clippy (its own build-script run and OUT_DIR) the images are only type-checked, so nvcc is not run twice
    let lint_only = std::env::var("RUSTC_WORKSPACE_WRAPPER").is_ok_and(|w| w.contains("clippy"));
    if cpu_only || lint_only {
        for f in ["smoke.fatbin", "step_exact.fatbin", "step_prof.fatbin", "staged.fatbin"] {
            std::fs::write(out.join(f), b"").unwrap();
        }
    } else {
        println!("cargo:rustc-env=RLMAX_CUDA_RELEASE={}", rlmax_nvcc::nvcc_release(&rlmax_nvcc::cuda_root()));
        gpu_images(&repo, &out, &include);
    }
    if !(cpu_only || lint_only) {
        let staged = repo.join("src/staged");
        rlmax_nvcc::fatbin_with(&staged.join("stages.cu"), &out.join("staged.fatbin"),
            &[&staged, &include], &[], &["--fmad=false", "-prec-div=true", "-prec-sqrt=true", "-ftz=false"]);
    }
    cpu_build(&repo, &include);
    // Separate translation unit: stage-only headers must not change the ordinary CPU tick.
    let mut stage_size=cc::Build::new();
    stage_size.cpp(true).file(repo.join("src/staged/sizes.cpp"))
        .include(repo.join("src/staged")).include(&include);
    let stage_compiler=stage_size.get_compiler();
    if stage_compiler.is_like_msvc() { stage_size.compiler(clang_cl(&stage_compiler)).flag("/std:c++17"); }
    else { stage_size.flag("-std=c++17"); }
    stage_size.compile("rlmax_staged_size");
    println!("cargo:rerun-if-changed={}", repo.join("src/staged").display());
    println!("cargo:rustc-link-search=native={}", rlmax_nvcc::driver_lib_dir().display());
    println!("cargo:rustc-link-lib=dylib=cuda");
}

fn gpu_images(repo: &std::path::Path, out: &std::path::Path, include: &std::path::Path) {
    let include = include.to_path_buf();
    // RLMAX_REGS=2=128,6=224: register budgets of the tick kernels per car count (src/kernels/step.cu), for measuring
    // the per-arch choice; never a gate build
    println!("cargo:rerun-if-env-changed=RLMAX_REGS");
    let regs: Vec<(String, String)> = std::env::var("RLMAX_REGS")
        .unwrap_or_default()
        .split(',')
        .filter_map(|kv| kv.split_once('='))
        .map(|(k, v)| (format!("RLMAX_REGS_{}", k.trim()), v.trim().to_string()))
        .collect();
    let regs: Vec<(&str, &str)> = regs.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
    let prof: Vec<(&str, &str)> = regs.iter().copied().chain([("RLMAX_PROFILE", "1")]).collect();
    rlmax_nvcc::fatbin(&repo.join("src/kernels/smoke.cu"), &out.join("smoke.fatbin"), &[&include], &[]);
    // the oracle build of the step kernel: exact math, so it equals the CPU build on every word
    rlmax_nvcc::fatbin_with(
        &repo.join("src/kernels/step.cu"),
        &out.join("step_exact.fatbin"),
        &[&include],
        &regs,
        &["--fmad=false", "-prec-div=true", "-prec-sqrt=true", "-ftz=false"],
    );
    // the same build with the per-stage timers (tick.h RLMAX_PROF), for profiling only: built when RLMAX_PROF_BUILD=1
    // (it doubles the step kernel's compile time), else an empty image that fails at load
    println!("cargo:rerun-if-env-changed=RLMAX_PROF_BUILD");
    if !std::env::var("RLMAX_PROF_BUILD").is_ok_and(|v| v == "1") {
        std::fs::write(out.join("step_prof.fatbin"), b"").unwrap();
        return;
    }
    rlmax_nvcc::fatbin_with(
        &repo.join("src/kernels/step.cu"),
        &out.join("step_prof.fatbin"),
        &[&include],
        &prof,
        &["--fmad=false", "-prec-div=true", "-prec-sqrt=true", "-ftz=false"],
    );
}

// CPU build of the physics: exact math (no FMA contraction, IEEE division and sqrt), so it equals the GPU build. On
// Windows the compiler is clang-cl (Visual Studio's "C++ Clang tools"), not cl.exe: MSVC 14.43 /O2 miscompiles the
// solver, and clang is the compiler family of the Linux build too.
fn cpu_build(repo: &std::path::Path, include: &std::path::Path) {
    let mut cpu = cc::Build::new();
    cpu.cpp(true).file(repo.join("src/cpu/cpu.cpp")).include(include).warnings(true);
    let msvc_target = cpu.get_compiler().is_like_msvc();
    let clang_cl = if msvc_target { Some(clang_cl(&cpu.get_compiler())) } else { None };
    if let Some(cc) = &clang_cl {
        cpu.compiler(cc);
        // clang-cl ignores GCC-style flags it does not know ("unknown argument ignored"); /clang: passes them through
        cpu.flag("/std:c++17").flag("/O2").flag("/EHsc").flag("/clang:-ffp-contract=off").flag("/clang:-fno-fast-math");
    } else {
        cpu.flag("-std=c++17").flag("-O2").flag("-ffp-contract=off").flag("-fno-fast-math");
    }
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    std::fs::write(out.join("cpu_compiler.txt"), cpu.get_compiler().path().display().to_string()).unwrap();
    // RLMAX_CPU_UBSAN=1: UndefinedBehaviorSanitizer on the CPU physics, for hunting defects; never a gate build
    println!("cargo:rerun-if-env-changed=RLMAX_CPU_UBSAN");
    let ubsan = std::env::var("RLMAX_CPU_UBSAN").is_ok_and(|v| v == "1");
    if ubsan {
        cpu.flag("-fsanitize=undefined").flag("-fno-sanitize=vptr");
    }
    // the impulse capture (lead 56): the CPU build only, never the GPU images
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_IMPULSE_LOG");
    if std::env::var_os("CARGO_FEATURE_IMPULSE_LOG").is_some() {
        cpu.define("RLMAX_IMPULSE_LOG", "1");
    }
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_FULL_STATE");
    if std::env::var_os("CARGO_FEATURE_FULL_STATE").is_some() {
        cpu.define("RLMAX_FULL_STATE", "1");
    }
    cpu.compile("rlmax_cpu");
    if ubsan {
        if let Some(cc) = &clang_cl {
            let dir = ubsan_lib_dir(cc);
            println!("cargo:rustc-link-search=native={}", dir.display());
            println!("cargo:rustc-link-lib=static=clang_rt.ubsan_standalone-x86_64");
            println!("cargo:rustc-link-lib=static=clang_rt.ubsan_standalone_cxx-x86_64");
        } else {
            println!("cargo:rustc-link-lib=ubsan");
        }
    }
    for dir in ["src/physics", "src/capi", "src/cpu", "src/kernels"] {
        println!("cargo:rerun-if-changed={}", repo.join(dir).display());
    }
}

// clang-cl: $RLMAX_CLANG_CL, else the one beside the MSVC toolset cc found (VC/Tools/Llvm/x64/bin).
fn clang_cl(cl: &cc::Tool) -> PathBuf {
    println!("cargo:rerun-if-env-changed=RLMAX_CLANG_CL");
    if let Some(p) = std::env::var_os("RLMAX_CLANG_CL") {
        return PathBuf::from(p);
    }
    // cl.exe is VC/Tools/MSVC/<version>/bin/Host<arch>/<arch>/cl.exe
    let tools = cl.path().ancestors().nth(6).map(|p| p.to_path_buf());
    let found = tools.map(|t| t.join("Llvm/x64/bin/clang-cl.exe")).filter(|p| p.is_file());
    found.unwrap_or_else(|| {
        panic!(
            "clang-cl not found beside {} (install Visual Studio's \"C++ Clang tools for Windows\" or set RLMAX_CLANG_CL)",
            cl.path().display()
        )
    })
}

// clang's runtime libraries: Llvm/x64/lib/clang/<major>/lib/windows beside clang-cl
fn ubsan_lib_dir(clang_cl: &std::path::Path) -> PathBuf {
    let root = clang_cl.parent().and_then(|b| b.parent()).unwrap().join("lib/clang");
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(&root).unwrap().flatten().map(|e| e.path().join("lib/windows")).collect();
    dirs.sort();
    dirs.pop().unwrap()
}
