//! RLMAX host library. P0 holds only the driver binding and the toolchain smoke kernel; the physics
//! kernels arrive in P2 (DESIGN §11).

pub mod arena;
pub mod cpu;
pub mod driver;
pub mod gpu;

/// Fat binary of `src/kernels/smoke.cu`.
pub static SMOKE_FATBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/smoke.fatbin"));

#[cfg(test)]
mod tests {
    use super::driver::{Buffer, Device, Module};
    use std::ffi::c_void;

    #[test]
    fn smoke_kernel_runs() {
        let dev = Device::open(0).expect("open device 0");
        let info = dev.info().unwrap();
        eprintln!("{info:?}");
        let module = Module::load(&dev, super::SMOKE_FATBIN).unwrap();
        let f = module.function(c"smoke_axpy").unwrap();

        let n: usize = 1 << 16;
        let x: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let y0: Vec<f32> = (0..n).map(|i| (n - i) as f32).collect();
        let xb = Buffer::from_slice(&dev, &x).unwrap();
        let yb = Buffer::from_slice(&dev, &y0).unwrap();

        let (mut a, mut px, mut py, mut nn) = (2.0f32, xb.ptr(), yb.ptr(), n as i32);
        let mut params = [
            (&mut a as *mut f32).cast::<c_void>(),
            (&mut px as *mut u64).cast(),
            (&mut py as *mut u64).cast(),
            (&mut nn as *mut i32).cast(),
        ];
        unsafe { f.launch(n.div_ceil(128) as u32, 128, &mut params).unwrap() };
        dev.synchronize().unwrap();

        let mut y = vec![0.0f32; n];
        yb.read(&mut y).unwrap();
        for i in 0..n {
            assert_eq!(y[i], 2.0f32.mul_add(x[i], y0[i]), "lane {i}");
        }
    }
}
