//! The GPU build: the step kernel (`src/kernels/step.cu`), one thread per arena, on device-resident state.

use crate::arena::Tri;
use crate::cpu::ArenaData;
use crate::driver::{Buffer, DevicePtr, Device, Module, Result};
use std::ffi::c_void;

/// Fat binary of `src/kernels/step.cu` built under exact math (the oracle build: equal to the CPU build on every word).
pub static STEP_EXACT_FATBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/step_exact.fatbin"));

/// Small-batch staged image, built separately from the ordinary tick.
pub static STAGED_FATBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/staged.fatbin"));

/// The exact build with per-stage cycle counters (`RLMAX_PROFILE`), for profiling; empty unless built with
/// `RLMAX_PROF_BUILD=1`.
pub static STEP_PROF_FATBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/step_prof.fatbin"));

/// `ArenaData` as the kernel takes it (by value): device pointers, then the grid. Must match `src/physics/arena_data.h`.
#[repr(C)]
#[derive(Clone, Copy)]
struct DevArenaData {
    tris: DevicePtr,
    cell_off: DevicePtr,
    cell_idx: DevicePtr,
    goal_ly: DevicePtr,
    tri_box: DevicePtr,
    grid_min: [f32; 3],
    inv_cell: f32,
    nx: i32,
    ny: i32,
    nz: i32,
    max_cell_count: u32,
}

impl DevArenaData {
    fn same_graph_args(&self, rhs: &Self) -> bool {
        self.tris == rhs.tris && self.cell_off == rhs.cell_off && self.cell_idx == rhs.cell_idx &&
        self.goal_ly == rhs.goal_ly && self.tri_box == rhs.tri_box &&
        self.grid_min.map(f32::to_bits) == rhs.grid_min.map(f32::to_bits) &&
        self.inv_cell.to_bits() == rhs.inv_cell.to_bits() && self.nx == rhs.nx && self.ny == rhs.ny &&
        self.nz == rhs.nz && self.max_cell_count == rhs.max_cell_count
    }
}

/// The arena's static data on the device.
pub struct GpuArenaData {
    _tris: Buffer<Tri>,
    _cell_off: Buffer<u32>,
    _cell_idx: Buffer<u16>,
    _goal_ly: Buffer<f32>,
    _tri_box: Buffer<f32>,
    raw: DevArenaData,
}

impl GpuArenaData {
    pub fn upload(dev: &Device, data: &ArenaData) -> Result<GpuArenaData> {
        let tris = Buffer::from_slice(dev, &data.tris)?;
        let cell_off = Buffer::from_slice(dev, &data.cell_off)?;
        let cell_idx = Buffer::from_slice(dev, &data.cell_idx)?;
        let goal_ly = Buffer::from_slice(dev, &data.goal_ly)?;
        let tri_box = Buffer::from_slice(dev, &data.tri_box)?;
        let raw = DevArenaData {
            tris: tris.ptr(),
            cell_off: cell_off.ptr(),
            cell_idx: cell_idx.ptr(),
            goal_ly: goal_ly.ptr(),
            tri_box: tri_box.ptr(),
            grid_min: data.spec.min,
            inv_cell: 1.0 / data.spec.cell,
            nx: data.spec.dims[0] as i32,
            ny: data.spec.dims[1] as i32,
            nz: data.spec.dims[2] as i32,
            max_cell_count: data.max_cell_count,
        };
        Ok(GpuArenaData { _tris: tris, _cell_off: cell_off, _cell_idx: cell_idx, _goal_ly: goal_ly, _tri_box: tri_box, raw })
    }
}

struct StagedGraph {
    data: DevArenaData,
    exec: crate::driver::GraphExec,
}

/// A batch of arenas with the same car count, resident on the device.
pub struct GpuBatch {
    // Drop graph executables before the module and buffers they reference.
    staged_graph: std::cell::RefCell<[Option<StagedGraph>; 8]>,  // one per ticks in a graph, 1 to 8
    module: Module,
    body_split_module: Option<Module>,
    num_cars: u32,
    has_psyclops: bool,
    num_arenas: u32,
    states: Buffer<u32>,
    /// Ticks per step launch, sized from the last launch's rate to keep every launch under [`MAX_LAUNCH_S`].
    ticks_per_launch: u32,
    /// Threads per block of the tick kernels ([`tick_block`]), fixed at construction from the arena count.
    block: u32,
    staged_scratch: Option<Buffer<u32>>,
    staged_block: u32,
    staged_finish_block: u32,
    staged_car_block: u32,
    staged_wheel_block: u32,
}

/// Threads per block of the tick kernels. `RLMAX_BLOCK` (1, 2, 4, 8, 16, 32, 64 or 128) overrides, for measurement;
/// otherwise [`auto_block`] from the arenas per SM.
pub fn tick_block(compute_capability: (i32, i32), num_arenas: u32, sm_count: i32) -> u32 {
    match std::env::var("RLMAX_BLOCK") {
        Ok(v) => match v.trim().parse::<u32>() {
            Ok(b @ (1 | 2 | 4 | 8 | 16 | 32 | 64 | 128)) => b,
            _ => panic!("RLMAX_BLOCK must be 1, 2, 4, 8, 16, 32, 64 or 128, got {v:?}"),
        },
        Err(_) => auto_block(compute_capability, num_arenas, sm_count),
    }
}

/// The automatic block. With few arenas per SM a 32-thread block leaves most SMs idle or one warp deep; 8 or 16
/// arenas per block spreads the arenas over more warps (RTX 5090, 2 Nov, 1 repeat, M arena-ticks/s, 1v1 at 1,024 / 2,048 /
/// 4,096 / 8,192 / 16,384 arenas: block 32 1.77 / 3.31 / 6.43 / 12.34 / 22.04, 16 1.93 / 3.61 / 7.15 / 12.71 / 20.31, 8
/// 2.20 / 3.80 / 7.26 / 11.89 / 14.93; 3v3 the same order). Arenas per SM (integer division) at most 24: 8 threads; at
/// most 48: 16; above: the card's default, 32 on sm_120 (+2.5% over 128 on R 1v1 65,536,
/// 3 seeds within 0.3%; 64 +1.7%, 256 -3.7%) and 128 elsewhere (sm_86: 32 and 64 within drift, P3.md §4). On sm_120 at
/// most 12 per SM: 2 threads (4 Nov, best of 2 seeds, arenas 1,024, blocks 1 / 2 / 4 / 8: 1v1 3.46 / 3.56 / 3.38 / 3.03,
/// 3v3 1.54 / 1.66 / 1.56 / 1.35, mix 2.29 / 2.25 / 1.95 / 1.60; at 24 per SM 8 stays best on 1v1 and 3v3). The kernels'
/// launch bound is 128 or a `__maxnreg__` budget, so any block up to 128 is valid.
///
/// The block never changes a result: a tick kernel runs one arena per thread, reading and writing only that arena's own
/// state and its own local scratch, with no `__syncthreads`, shuffle, vote, shared memory or atomic on the physics path
/// (the `RLMAX_PROF` counters alone use an atomic, and are not physics), and the arena index is
/// `blockIdx.x * blockDim.x + threadIdx.x`, so the same arena gets the same thread-private computation at any block size.
pub fn auto_block(compute_capability: (i32, i32), num_arenas: u32, sm_count: i32) -> u32 {
    let per_sm = num_arenas / sm_count.max(1) as u32;
    match per_sm {
        0..=12 if compute_capability.0 >= 12 => 2,
        0..=24 => 8,
        25..=48 => 16,
        _ => if compute_capability.0 >= 12 { 32 } else { 128 },
    }
}

/// The most arenas per SM that run on the staged path. Its branches overlap, so it wins while the card has idle room; it
/// flattens once it's full, and the large path, which keeps climbing, overtakes it. `RLMAX_STAGED_PER_SM` overrides, for
/// measurement. sm_120 (5090, 5 Nov, two seeds, M arena-ticks/s, large / staged): 1v1 20,480 27.4 / 33.0, 24,576 30.5 /
/// 32.3, 28,672 33.3 / 32.2; 2v2 20,480 16.1 / 17.0, 24,576 17.9 / 16.8; 3v3 16,384 10.0 / 11.4, 20,480 11.6 / 11.5.
/// sm_86 (3080, 5 Nov, second seed, large / staged): 1v1 4,096 5.12 / 6.89, 8,192 8.38 / 8.45, 12,288 9.67 / 9.22; 3v3 4,096 2.18 /
/// 3.09, 8,192 3.71 / 3.63. Other archs keep the first limit until measured.
pub fn staged_max_per_sm(compute_capability: (i32, i32), num_cars: u32) -> u32 {
    if let Some(v) = std::env::var("RLMAX_STAGED_PER_SM").ok().and_then(|v| v.parse().ok()) {
        return v;
    }
    match (compute_capability, num_cars) {
        ((12.., _), ..=2) => 150,
        ((12.., _), ..=4) => 128,
        ((12.., _), _) => 116,
        ((8, 6), _) => 100,
        _ => 48,
    }
}

/// The longest a single launch may run (wall): the 3080 also drives the display, whose watchdog (TDR, 2 s by
/// default) ends longer launches, and a long launch stalls the desktop. Launches are sized to 80% of it.
pub const MAX_LAUNCH_S: f64 = 0.5;

impl GpuBatch {
    /// `words` holds `num_arenas` arena states back to back (`rlmax_state_words(num_cars)` words each).
    /// This selects the standard kernels; use [`GpuBatch::new_with_psyclops`] if any car can be Psyclops.
    pub fn new(dev: &Device, num_cars: u32, words: &[u32]) -> Result<GpuBatch> {
        Self::with_image_and_psyclops(dev, num_cars, words, STEP_EXACT_FATBIN, false)
    }

    /// Select the three-wheel variant for the whole batch when any arena can contain a Psyclops car.
    pub fn new_with_psyclops(dev: &Device, num_cars: u32, words: &[u32], has_psyclops: bool) -> Result<GpuBatch> {
        Self::with_image_and_psyclops(dev, num_cars, words, STEP_EXACT_FATBIN, has_psyclops)
    }

    /// As [`GpuBatch::new`] with another build of the step kernel (e.g. [`STEP_PROF_FATBIN`]).
    pub fn with_image(dev: &Device, num_cars: u32, words: &[u32], image: &[u8]) -> Result<GpuBatch> {
        Self::with_image_and_psyclops(dev, num_cars, words, image, false)
    }

    /// As [`GpuBatch::new_with_psyclops`] with an explicit kernel image.
    pub fn with_image_and_psyclops(dev: &Device, num_cars: u32, words: &[u32], image: &[u8], has_psyclops: bool) -> Result<GpuBatch> {
        let per = crate::cpu::state_words(num_cars) as usize;
        assert!(per > 0 && words.len().is_multiple_of(per), "state buffer is not a whole number of arenas");
        // no stack limit is set: nothing recurses, so the driver sizes each thread's local memory from the loaded
        // kernel's static frame on this device
        assert!(!image.is_empty(), "empty kernel image (an RLMAX_CPU_ONLY build, or the profiling build without RLMAX_PROF_BUILD=1)");
        let module = Module::load(dev, image)?;
        let states = Buffer::from_slice(dev, words)?;
        let num_arenas = (words.len() / per) as u32;
        let info = dev.info()?;
        let block = tick_block(info.compute_capability, num_arenas, info.sm_count);
        // Below the measured small-batch crossover, keep one scratch record per arena.
        // Custom images and Psyclops retain their original kernels.
        let per_sm = num_arenas / info.sm_count.max(1) as u32;
        let staged_words = crate::cpu::staged_words(num_cars) as usize;
        let standard_image = std::ptr::eq(image.as_ptr(), STEP_EXACT_FATBIN.as_ptr()) && image.len() == STEP_EXACT_FATBIN.len();
        let safe_grid = num_arenas.checked_mul(num_cars.saturating_mul(4)).and_then(|v| v.checked_add(127)).is_some();
        let staged_scratch = if standard_image && !has_psyclops && staged_words != 0 && num_arenas != 0 && per_sm <= staged_max_per_sm(info.compute_capability, num_cars) && safe_grid {
            let total=(staged_words + num_cars as usize + 1).checked_mul(num_arenas as usize).expect("staged scratch overflow");
            Some(Buffer::zeroed(dev, total)?)
        } else { None };
        let body_split_module = if staged_scratch.is_some() {
            assert!(!STAGED_FATBIN.is_empty(), "staged image is empty");
            Some(Module::load(dev, STAGED_FATBIN)?)
        } else { None };
        // Spread short stages across SMs.
        let mut staged_block = 8;
        let threads_per_sm = num_arenas.div_ceil(info.sm_count.max(1) as u32);
        while staged_block < 64 && staged_block < threads_per_sm { staged_block *= 2; }
        // Spread the long per-arena finish
        // over about six blocks per SM, using the regular block above eight.
        let arenas_per_six_sms = num_arenas / (info.sm_count.max(1) as u32 * 6);
        let mut finish_block = 1;
        while finish_block < 16 && finish_block * 2 <= arenas_per_six_sms { finish_block *= 2; }
        let staged_finish_block = if finish_block <= 8 { finish_block } else { staged_block };
        let mut staged_car_block = 8;
        let cars_per_sm = (num_arenas * num_cars).div_ceil(info.sm_count.max(1) as u32);
        while staged_car_block < 64 && staged_car_block < cars_per_sm { staged_car_block *= 2; }
        let mut staged_wheel_block = 8;
        let wheels_per_sm = (num_arenas * num_cars * 4).div_ceil(info.sm_count.max(1) as u32);
        while staged_wheel_block < 64 && staged_wheel_block < wheels_per_sm { staged_wheel_block *= 2; }
        // When the rounded-up block size leaves SMs idle, split the grid until
        // it can cover the device. Wheel blocks must retain whole four-lane groups.
        let sms = info.sm_count.max(1) as u32;
        while staged_car_block > 1 && (num_arenas * num_cars).div_ceil(staged_car_block) < sms {
            staged_car_block /= 2;
        }
        while staged_wheel_block > 4 && (num_arenas * num_cars * 4).div_ceil(staged_wheel_block) < sms {
            staged_wheel_block /= 2;
        }
        Ok(GpuBatch { staged_graph: std::cell::RefCell::new(Default::default()), module, body_split_module, num_cars, has_psyclops, num_arenas, states, ticks_per_launch: 8, block, staged_scratch, staged_block, staged_finish_block, staged_car_block, staged_wheel_block })
    }

    /// `ticks` ticks, synchronised, in launches of at most [`MAX_LAUNCH_S`] each: the batch's first launch is 8 ticks,
    /// and each later one is sized from the previous one's rate. The result does not depend on the split (each launch
    /// reads the states, steps them and writes them back).
    pub fn step(&mut self, dev: &Device, data: &GpuArenaData, ticks: u32) -> Result<()> {
        let mut left = ticks;
        while left > 0 {
            let k = self.ticks_per_launch.min(left);
            let t = std::time::Instant::now();
            self.launch_step(data, k)?;
            dev.synchronize()?;
            let s = t.elapsed().as_secs_f64().max(1e-6);
            left -= k;
            self.ticks_per_launch = ((0.8 * MAX_LAUNCH_S * k as f64 / s) as u32).clamp(1, 1 << 16);
        }
        Ok(())
    }

    /// Change the selected tick variant before writing or branching in states with a different set of body presets.
    /// `true` is required whenever any arena may contain Psyclops, including device-side source buffers.
    pub fn set_has_psyclops(&mut self, has_psyclops: bool) {
        self.has_psyclops = has_psyclops;
    }

    /// The step kernel for this car count.
    pub fn step_kernel(&self) -> Result<crate::driver::Function<'_>> {
        self.tick_kernel("rlmax_step")
    }

    /// One launch of `ticks` ticks, not synchronised.
    pub fn launch_step(&self, data: &GpuArenaData, ticks: u32) -> Result<()> {
        if self.staged_scratch.is_some() && !self.has_psyclops {
            return unsafe { self.launch_staged(data, ticks, 0) };
        }
        let f = self.step_kernel()?;
        let (mut p, mut n, mut t, mut ad) = (self.states.ptr(), self.num_arenas, ticks, data.raw);
        let mut params = [
            (&mut p as *mut DevicePtr).cast::<c_void>(),
            (&mut n as *mut u32).cast(),
            (&mut t as *mut u32).cast(),
            (&mut ad as *mut DevArenaData).cast(),
        ];
        unsafe { f.launch(self.num_arenas.div_ceil(self.block), self.block, &mut params) }
    }

    /// The profiling build's per-stage warp cycles since the last read (lane 0 of each warp), then cleared.
    pub fn read_profile(&self, dev: &Device) -> Result<[u64; 32]> {
        let buf = Buffer::from_slice(dev, &[0u64; 32])?;
        let f = self.module.function(c"rlmax_prof_read")?;
        let mut p = buf.ptr();
        let mut params = [(&mut p as *mut DevicePtr).cast::<c_void>()];
        unsafe { f.launch(1, 1, &mut params)? };
        dev.synchronize()?;
        let mut out = [0u64; 32];
        buf.read(&mut out)?;
        Ok(out)
    }

    /// Launches `entry` from `image_batch`'s module on this batch's states. A bench probe: a kernel from another image, same ABI as a step.
    ///
    /// # Safety
    /// `entry` must take (states, n, ticks, arena data), and the grid and block must suit it.
    pub unsafe fn probe_launch(&self, image_batch: &GpuBatch, entry: &std::ffi::CStr, data: &GpuArenaData, ticks: u32, grid: u32, block: u32) -> Result<()> {
        let f = image_batch.module.function(entry)?;
        let (mut p, mut n, mut t, mut ad) = (self.states.ptr(), self.num_arenas, ticks, data.raw);
        let mut params = [
            (&mut p as *mut DevicePtr).cast::<c_void>(),
            (&mut n as *mut u32).cast(),
            (&mut t as *mut u32).cast(),
            (&mut ad as *mut DevArenaData).cast(),
        ];
        unsafe { f.launch_on(grid, block, &mut params, 0) }
    }

    /// The device address of the arena states, for a bench kernel that acts on them between steps (workload T).
    pub fn states_ptr(&self) -> DevicePtr {
        self.states.ptr()
    }

    pub fn read(&self, out: &mut [u32]) -> Result<()> {
        self.states.read(out)
    }

    pub fn num_arenas(&self) -> u32 {
        self.num_arenas
    }

    fn kernel(&self, base: &str) -> Result<crate::driver::Function<'_>> {
        let name = std::ffi::CString::new(format!("{base}_{}", self.num_cars)).expect("kernel name");
        if base.starts_with("rlmax_staged_") {
            self.body_split_module.as_ref().expect("staged module").function(&name)
        } else { self.module.function(&name) }
    }

    /// Set a fraction per arena on the device, or restore the default 0.5 with None.
    /// Negative values mark earlier ticks in a render frame; other values must be at most 1.
    pub fn set_pad_frame_fractions(&self, fractions: Option<&Buffer<f32>>) -> Result<()> {
        if let Some(f) = fractions { assert_eq!(f.len(), self.num_arenas as usize); }
        let f = self.kernel("rlmax_pad_fraction")?;
        let (mut states, mut count, mut values) = (self.states.ptr(), self.num_arenas, fractions.map_or(0, |v| v.ptr()));
        let mut params = [(&mut states as *mut DevicePtr).cast::<c_void>(), (&mut count as *mut u32).cast(), (&mut values as *mut DevicePtr).cast()];
        unsafe { f.launch(count.div_ceil(128), 128, &mut params) }
    }

    fn tick_kernel(&self, base: &str) -> Result<crate::driver::Function<'_>> {
        if self.has_psyclops {
            self.kernel(&format!("{base}_psyclops"))
        } else {
            self.kernel(base)
        }
    }

    /// Every arena to a kickoff (`rlmax_kickoff`), its rng seeded from `seed` and the arena index. Not synchronised.
    pub fn kickoff_all(&mut self, seed: u32) -> Result<()> {
        let f = self.kernel("rlmax_kickoff")?;
        let (mut p, mut n, mut s) = (self.states.ptr(), self.num_arenas, seed);
        let mut params =
            [(&mut p as *mut DevicePtr).cast::<c_void>(), (&mut n as *mut u32).cast(), (&mut s as *mut u32).cast()];
        unsafe { f.launch(self.num_arenas.div_ceil(self.block), self.block, &mut params) }
    }

    /// Every arena's state from the host (`num_arenas` states back to back), e.g. start states taken from replays.
    /// Set [`GpuBatch::set_has_psyclops`] if the new states can contain Psyclops.
    pub fn write(&mut self, words: &[u32]) -> Result<()> {
        self.states.write(words)
    }

    /// A device-side copy of every arena's state, to branch from while the batch's own arenas are overwritten.
    pub fn snapshot(&self, dev: &Device) -> Result<Buffer<u32>> {
        Buffer::copy_of(dev, &self.states)
    }

    /// Reset-and-branch (`rlmax_branch`): arena i becomes a copy of arena `map[i]` of `src` (start states uploaded with
    /// [`Buffer::from_slice`], or a [`GpuBatch::snapshot`]; never this batch's own buffer), its rng reseeded from
    /// (`seed`, i). `map` holds one source index per arena; an out-of-range index leaves that arena as it was. Not
    /// synchronised. Set [`GpuBatch::set_has_psyclops`] if the source states can contain Psyclops.
    pub fn branch(&mut self, src: &Buffer<u32>, map: &Buffer<u32>, seed: u32) -> Result<()> {
        let per = crate::cpu::state_words(self.num_cars) as usize;
        assert!(src.len().is_multiple_of(per), "source buffer is not a whole number of arenas");
        assert_eq!(map.len(), self.num_arenas as usize, "one source index per arena");
        unsafe { self.branch_raw(src.ptr(), (src.len() / per) as u32, map.ptr(), seed, 0) }
    }

    /// [`GpuBatch::branch`] on raw device addresses (memory a trainer's tensors own) and a stream (0: the default).
    ///
    /// # Safety
    /// `src` holds `n_src` arena states and is not this batch's own buffer; `map` holds one u32 per arena.
    pub unsafe fn branch_raw(&mut self, src: DevicePtr, n_src: u32, map: DevicePtr, seed: u32, stream: u64) -> Result<()> {
        assert_ne!(src, self.states.ptr(), "branch from a snapshot, not the batch's own states");
        let f = self.kernel("rlmax_branch")?;
        let (mut d, mut n, mut s, mut ns, mut m, mut sd) = (self.states.ptr(), self.num_arenas, src, n_src, map, seed);
        let mut params = [
            (&mut d as *mut DevicePtr).cast::<c_void>(),
            (&mut n as *mut u32).cast(),
            (&mut s as *mut DevicePtr).cast(),
            (&mut ns as *mut u32).cast(),
            (&mut m as *mut DevicePtr).cast(),
            (&mut sd as *mut u32).cast(),
        ];
        // one warp per arena
        unsafe { f.launch_on((self.num_arenas as u64 * 32).div_ceil(128) as u32, 128, &mut params, stream) }
    }

    /// One decision (`rlmax_decide`): goal resets, each car's action row (one byte per car, arena-major) through the
    /// 8-float action `table` (any number of rows up to 256), then `ticks` ticks; resets are counted per arena in
    /// `goals`. Not synchronised.
    pub fn decide(&mut self, data: &GpuArenaData, ticks: u32, rows: &Buffer<u8>, table: &Buffer<f32>, goals: &Buffer<u32>) -> Result<()> {
        assert_eq!(rows.len(), (self.num_arenas * self.num_cars) as usize, "one row byte per car");
        unsafe { self.decide_raw(data, ticks, rows.ptr(), table.ptr(), goals.ptr(), 0) }
    }

    /// [`GpuBatch::decide`] on raw device addresses and a stream (0: the default).
    ///
    /// # Safety
    /// `rows` holds num_arenas x num_cars bytes, every one a row of `table` (8 floats per row); `goals` one u32 per arena.
    pub unsafe fn decide_raw(&mut self, data: &GpuArenaData, ticks: u32, rows: DevicePtr, table: DevicePtr, goals: DevicePtr, stream: u64) -> Result<()> {
        unsafe { self.tick_launch("rlmax_decide", data, ticks, rows, table, goals, stream) }
    }

    /// [`GpuBatch::decide`] with raw controls in place of rows: `controls` holds 8 floats per car, arena-major (the
    /// action table's order: throttle, steer, pitch, yaw, roll, jump, boost, handbrake), for action spaces past 256
    /// rows; the same floats give the same physics as a table row. Not synchronised.
    ///
    /// # Safety
    /// `controls` holds num_arenas x num_cars x 8 floats; `goals` one u32 per arena.
    pub unsafe fn decide_controls_raw(&mut self, data: &GpuArenaData, ticks: u32, controls: DevicePtr, goals: DevicePtr, stream: u64) -> Result<()> {
        unsafe { self.tick_launch("rlmax_decide", data, ticks, 0, controls, goals, stream) }
    }

    /// One decision of a short rollout (`rlmax_rollout`): as [`GpuBatch::decide`], but a goal stops its arena where it
    /// is. `stop[i]`: bits 0..15 the tick of this call on which the goal fell (1-based), 0xFFFF if none, 0 if the arena
    /// had already stopped; bits 16..17 the side that scored (1 blue, 2 orange). Not synchronised.
    pub fn rollout(&mut self, data: &GpuArenaData, ticks: u32, rows: &Buffer<u8>, table: &Buffer<f32>, stop: &Buffer<u32>) -> Result<()> {
        assert_eq!(rows.len(), (self.num_arenas * self.num_cars) as usize, "one row byte per car");
        assert_eq!(stop.len(), self.num_arenas as usize, "one stop word per arena");
        unsafe { self.rollout_raw(data, ticks, rows.ptr(), table.ptr(), stop.ptr(), 0) }
    }

    /// [`GpuBatch::rollout`] on raw device addresses and a stream (0: the default).
    ///
    /// # Safety
    /// As [`GpuBatch::decide_raw`], with `stop` one u32 per arena.
    pub unsafe fn rollout_raw(&mut self, data: &GpuArenaData, ticks: u32, rows: DevicePtr, table: DevicePtr, stop: DevicePtr, stream: u64) -> Result<()> {
        unsafe { self.tick_launch("rlmax_rollout", data, ticks, rows, table, stop, stream) }
    }

    /// Every arena's extract row (physics/extract.h; [`crate::cpu::extract_row_floats`] floats per arena,
    /// field names from [`crate::cpu::extract_fields`]) into `out`. Not synchronised.
    pub fn extract(&self, out: &Buffer<f32>) -> Result<()> {
        assert_eq!(out.len(), (self.num_arenas * crate::cpu::extract_row_floats(self.num_cars)) as usize, "one row per arena");
        unsafe { self.extract_raw(out.ptr(), 0) }
    }

    /// [`GpuBatch::extract`] into a raw device address (a float tensor of num_arenas x row floats) on a stream.
    ///
    /// # Safety
    /// `out` holds num_arenas x [`crate::cpu::extract_row_floats`] floats.
    pub unsafe fn extract_raw(&self, out: DevicePtr, stream: u64) -> Result<()> {
        let f = self.kernel("rlmax_extract")?;
        let (mut p, mut n, mut o) = (self.states.ptr(), self.num_arenas, out);
        let mut params =
            [(&mut p as *mut DevicePtr).cast::<c_void>(), (&mut n as *mut u32).cast(), (&mut o as *mut DevicePtr).cast()];
        // a warp per block, 32 arenas per warp, 16 with eight cars (the shared-memory transpose in step.cu)
        let a = if self.num_cars == 8 { 16 } else { 32 };
        unsafe { f.launch_on(self.num_arenas.div_ceil(a), a, &mut params, stream) }
    }

    unsafe fn launch_staged(&self, data: &GpuArenaData, ticks: u32, stream: u64) -> Result<()> {
        if ticks == 0 { return Ok(()); }
        let (mut p,mut n,mut ad,mut scratch)=(self.states.ptr(),self.num_arenas,data.raw,self.staged_scratch.as_ref().expect("staged scratch").ptr());
        let mut params=[(&mut p as *mut DevicePtr).cast::<c_void>(),(&mut n as *mut u32).cast(),(&mut ad as *mut DevArenaData).cast(),(&mut scratch as *mut DevicePtr).cast()];
        let function=|base: &str| { let name=std::ffi::CString::new(format!("{base}_{}",self.num_cars)).unwrap(); self.body_split_module.as_ref().expect("staged module").function(&name) };
        let begin=function("rlmax_staged_begin")?; let pre=function("rlmax_staged_pre_cars")?; let early_ball=function("rlmax_staged_early_ball")?;
        let gate=function("rlmax_vehicle_gate")?; let fallback=function("rlmax_vehicle_fallback")?; let wheels=function("rlmax_vehicle_wheels")?;
        let world=function("rlmax_body_world")?; let ball=function("rlmax_body_ball")?;
        let predict=function("rlmax_body_predict")?; let sweep=function("rlmax_body_sweep")?;
        let finish=function("rlmax_staged_pre_integrate")?; let post=function("rlmax_staged_integrate_post")?;
        let grid=n.div_ceil(self.staged_block);
        let car_grid=(n*self.num_cars).div_ceil(self.staged_car_block);
        // Four adjacent lanes scan contiguous portions of one car's cell.
        let world_block=32;
        let world_grid=(n*self.num_cars*4).div_ceil(world_block);
        let pre_grid=car_grid;
        let wheel_grid=(n*self.num_cars*4).div_ceil(self.staged_wheel_block);
        let body_grid=(n*(self.num_cars+1)).div_ceil(self.staged_car_block);
        let lanes=if self.num_cars<=2 { 4 } else { 8 };
        let post_block=32;
        let post_grid=(n*lanes).div_ceil(post_block);
        let finish_grid=n.div_ceil(self.staged_finish_block);
        let mut left=ticks;
        while left!=0 {
            let count=left.min(8); let mut graphs=self.staged_graph.borrow_mut(); let cached=&mut graphs[count as usize-1];
            let rebuild=cached.as_ref().is_none_or(|g|!g.data.same_graph_args(&data.raw));
            if rebuild {
                let mut graph=crate::driver::GraphBuilder::new()?; let mut tails=Vec::new();
                // Stage edges per tick:
                // begin -> {pre -> gate -> fallback -> wheels -> predict -> sweep, world, ball, early ball-world}
                // all four -> ordered finish -> integrate/post -> next begin.
                // World and ball read only post-begin poses/geometry and own distinct manifolds;
                // pre and vehicle branches change velocity and car controls, not those fields.
                for _ in 0..count { unsafe {
                    let a=graph.add_after(&begin,grid,self.staged_block,&mut params,&tails)?;
                    // Body::active is the awake value fixed by begin, before vehicle wake.
                    let early=graph.add_after(&early_ball,grid,self.staged_block,&mut params,&[a])?;
                    let pre_node=graph.add_after(&pre,pre_grid,self.staged_car_block,&mut params,&[a])?;
                    let g=graph.add_after(&gate,grid,self.staged_block,&mut params,&[pre_node])?;
                    let f=graph.add_after(&fallback,finish_grid,self.staged_finish_block,&mut params,&[g])?;
                    let w=graph.add_after(&wheels,wheel_grid,self.staged_wheel_block,&mut params,&[f])?;
                    let pr=graph.add_after(&predict,body_grid,self.staged_car_block,&mut params,&[w])?;
                    let sw=graph.add_after(&sweep,body_grid,self.staged_car_block,&mut params,&[pr])?;
                    let cw=graph.add_after(&world,world_grid,world_block,&mut params,&[a])?;
                    let cb=graph.add_after(&ball,car_grid,self.staged_car_block,&mut params,&[a])?;
                    let fin=graph.add_after(&finish,finish_grid,self.staged_finish_block,&mut params,&[sw,cw,cb,early])?;
                    tails=vec![graph.add_after(&post,post_grid,post_block,&mut params,&[fin])?];
                } }
                *cached=Some(StagedGraph{data:data.raw,exec:graph.instantiate()?});
            }
            unsafe { cached.as_ref().unwrap().exec.launch(stream)?; }
            left-=count;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn tick_launch(&self, base: &str, data: &GpuArenaData, ticks: u32, rows: DevicePtr, table: DevicePtr, out: DevicePtr, stream: u64) -> Result<()> {
        if base == "rlmax_decide" && self.staged_scratch.is_some() && !self.has_psyclops {
            let f = self.kernel("rlmax_staged_controls")?;
            let (mut p, mut n, mut r, mut tb, mut g) = (self.states.ptr(), self.num_arenas, rows, table, out);
            let mut params = [(&mut p as *mut DevicePtr).cast::<c_void>(), (&mut n as *mut u32).cast(),
                              (&mut r as *mut DevicePtr).cast(), (&mut tb as *mut DevicePtr).cast(), (&mut g as *mut DevicePtr).cast()];
            unsafe { f.launch_on(self.num_arenas.div_ceil(self.staged_block), self.staged_block, &mut params, stream)?; }
            return unsafe { self.launch_staged(data, ticks, stream) };
        }
        let f = self.tick_kernel(base)?;
        let (mut p, mut n, mut t, mut ad) = (self.states.ptr(), self.num_arenas, ticks, data.raw);
        let (mut r, mut tb, mut g) = (rows, table, out);
        let mut params = [
            (&mut p as *mut DevicePtr).cast::<c_void>(),
            (&mut n as *mut u32).cast(),
            (&mut t as *mut u32).cast(),
            (&mut ad as *mut DevArenaData).cast(),
            (&mut r as *mut DevicePtr).cast(),
            (&mut tb as *mut DevicePtr).cast(),
            (&mut g as *mut DevicePtr).cast(),
        ];
        unsafe { f.launch_on(self.num_arenas.div_ceil(self.block), self.block, &mut params, stream) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::{parity_arenas, Arena, BallState, Controls};

    /// The automatic block on the 5090's 170 SMs and the 3080's 68 (CPU only, no device).
    #[test]
    fn auto_block_by_arenas_per_sm() {
        assert_eq!(auto_block((12, 0), 1024, 170), 2); // 6 per SM
        assert_eq!(auto_block((12, 0), 12 * 170, 170), 2);
        assert_eq!(auto_block((12, 0), 13 * 170, 170), 8);
        assert_eq!(auto_block((8, 6), 1024, 68), 8); // 15 per SM; sm_86 unmeasured below 24 per SM
        assert_eq!(auto_block((8, 6), 12 * 68, 68), 8);
        assert_eq!(auto_block((12, 0), 24 * 170, 170), 8); // 24 per SM
        assert_eq!(auto_block((12, 0), 25 * 170, 170), 16); // 25 per SM
        assert_eq!(auto_block((12, 0), 8192, 170), 16); // 48 per SM
        assert_eq!(auto_block((12, 0), 49 * 170, 170), 32);
        assert_eq!(auto_block((12, 0), 65536, 170), 32);
        assert_eq!(auto_block((8, 6), 65536, 68), 128);
        assert_eq!(auto_block((8, 6), 68 * 24, 68), 8);
        assert_eq!(auto_block((8, 6), 1, 0), 8);
    }

    #[test]
    fn staged_limit_by_card_and_cars() {
        assert_eq!(staged_max_per_sm((12, 0), 2), 150);
        assert_eq!(staged_max_per_sm((12, 0), 4), 128);
        assert_eq!(staged_max_per_sm((12, 0), 6), 116);
        assert_eq!(staged_max_per_sm((8, 6), 2), 100);
        assert_eq!(staged_max_per_sm((8, 0), 2), 48);
    }

    /// The driver's local-memory reservation for each step kernel at 65,536 arenas:
    /// free device memory before the batch's first launch, while it runs, and after it.
    #[test]
    fn local_memory_reservation() {
        let Ok(data) = ArenaData::soccar_from_env() else {
            eprintln!("no meshes; skipped");
            return;
        };
        let dev = Device::open(0).expect("open device 0");
        let gdata = GpuArenaData::upload(&dev, &data).unwrap();
        let mb = |b: usize| b as f64 / (1024.0 * 1024.0);
        for num_cars in [2u32, 4, 6] {
            let one = parity_arenas(num_cars, 1).remove(0);
            let words: Vec<u32> = (0..65_536).flat_map(|_| one.words().iter().copied()).collect();
            let batch = GpuBatch::new(&dev, num_cars, &words).unwrap();
            let (before, total) = dev.mem_info().unwrap();
            batch.launch_step(&gdata, 2).unwrap();
            let (during, _) = dev.mem_info().unwrap();
            dev.synchronize().unwrap();
            let (after, _) = dev.mem_info().unwrap();
            let frame = batch.step_kernel().unwrap().local_bytes().unwrap();
            eprintln!(
                "{num_cars} cars: frame {frame} B/thread; free {:.0} MB before the first launch, {:.0} during, {:.0} after (of {:.0} MB)",
                mb(before), mb(during), mb(after), mb(total)
            );
        }
    }

    /// The P2 gate's residency clause: the frame (local memory) does not cap resident threads, so every tick kernel's
    /// resident blocks per SM, as the driver computes them, equal the bound from its registers alone.
    #[test]
    fn residency_is_register_bound() {
        let dev = Device::open(0).expect("open device 0");
        let info = dev.info().unwrap();
        let block = 128u32;
        for num_cars in [2u32, 4, 6] {
            let words = parity_arenas(num_cars, 1).remove(0).words().to_vec();
            let batch = GpuBatch::new(&dev, num_cars, &words).unwrap();
            for base in ["rlmax_step", "rlmax_decide"] {
                let f = batch.kernel(base).unwrap();
                let regs = f.num_regs().unwrap();
                // registers are allocated per warp in units of 256
                let per_warp = (regs * 32 + 255) / 256 * 256;
                let warps = info.max_registers_per_sm / per_warp;
                let bound = (warps / (block as i32 / 32))
                    .min(info.max_blocks_per_sm)
                    .min(info.max_threads_per_sm / block as i32);
                let got = f.max_active_blocks(block).unwrap();
                eprintln!("{base}_{num_cars}: {regs} registers, {got} blocks of {block} per SM (register bound {bound})");
                assert_eq!(got, bound, "{base}_{num_cars}: residency capped below the register bound");
            }
        }
    }

    /// Reset-and-branch: each branch equals its source on every word but the reseeded rng, and the branches then step
    /// as the CPU build steps the same copies.
    #[test]
    fn branch_copies_and_steps_exactly() {
        let Ok(data) = ArenaData::soccar_from_env() else {
            eprintln!("no meshes; skipped");
            return;
        };
        let dev = Device::open(0).expect("open device 0");
        let gdata = GpuArenaData::upload(&dev, &data).unwrap();
        for num_cars in [2u32, 4, 6] {
            let (n_src, n) = (16usize, 64usize);
            let per = crate::cpu::state_words(num_cars) as usize;
            let src_words: Vec<u32> = parity_arenas(num_cars, n_src).iter().flat_map(|a| a.words().iter().copied()).collect();
            let map: Vec<u32> = (0..n as u32).map(|i| (i * 7) % n_src as u32).collect();
            let src = Buffer::from_slice(&dev, &src_words).unwrap();
            let mapb = Buffer::from_slice(&dev, &map).unwrap();
            let mut batch = GpuBatch::new(&dev, num_cars, &vec![0u32; n * per]).unwrap();
            batch.branch(&src, &mapb, 5).unwrap();
            dev.synchronize().unwrap();
            let mut got = vec![0u32; n * per];
            batch.read(&mut got).unwrap();
            let mut arenas = Vec::new();
            for (i, &m) in map.iter().enumerate() {
                let g = &got[i * per..(i + 1) * per];
                let s = &src_words[m as usize * per..(m as usize + 1) * per];
                let diff = g.iter().zip(s).filter(|(x, y)| x != y).count();
                assert!(diff <= 1, "{num_cars} cars, branch {i}: {diff} words differ from its source");
                arenas.push(crate::cpu::Arena::from_words(num_cars, g));
            }
            batch.step(&dev, &gdata, 240).unwrap();
            batch.read(&mut got).unwrap();
            for (i, a) in arenas.iter_mut().enumerate() {
                a.step(&data, 240);
                assert_eq!(a.words(), &got[i * per..(i + 1) * per], "{num_cars} cars, branch {i} after 240 ticks");
            }
            let path = if per.is_multiple_of(4) { "16-byte" } else { "4-byte" };
            eprintln!("{num_cars} cars: {n} branches of {n_src} states ({per} words, {path} copies) copied and stepped exactly");
        }
    }

    /// The trainer's extract: the GPU rows equal the CPU build's on every float, after play, at 2, 4 and 6 cars; the
    /// field names cover the row.
    #[test]
    fn extract_equals_cpu() {
        let Ok(data) = ArenaData::soccar_from_env() else {
            eprintln!("no meshes; skipped");
            return;
        };
        let dev = Device::open(0).expect("open device 0");
        let gdata = GpuArenaData::upload(&dev, &data).unwrap();
        for num_cars in [2u32, 4, 6] {
            let n = 64usize;
            let mut arenas = parity_arenas(num_cars, n);
            let words: Vec<u32> = arenas.iter().flat_map(|a| a.words().iter().copied()).collect();
            let mut batch = GpuBatch::new(&dev, num_cars, &words).unwrap();
            batch.step(&dev, &gdata, 120).unwrap();
            for a in arenas.iter_mut() {
                a.step(&data, 120);
            }
            let row = crate::cpu::extract_row_floats(num_cars) as usize;
            assert_eq!(crate::cpu::extract_fields(num_cars).len(), row);
            let out = Buffer::from_slice(&dev, &vec![0f32; n * row]).unwrap();
            batch.extract(&out).unwrap();
            dev.synchronize().unwrap();
            let mut got = vec![0f32; n * row];
            out.read(&mut got).unwrap();
            for (i, a) in arenas.iter().enumerate() {
                let want = a.extract();
                let g = &got[i * row..(i + 1) * row];
                if let Some(k) = want.iter().zip(g).position(|(x, y)| x.to_bits() != y.to_bits()) {
                    panic!("{num_cars} cars, arena {i}: extract differs at float {k} ({}): cpu {} gpu {}", crate::cpu::extract_fields(num_cars)[k], want[k], g[k]);
                }
            }
            eprintln!("{num_cars} cars: {n} extract rows of {row} floats equal the CPU build");
        }
    }

    /// The rollout decision: a goal stops the arena on its tick, reports the tick and side, and keeps the final state;
    /// a stopped arena stays as it is. Checked against the CPU build stepped tick by tick.
    #[test]
    fn rollout_stops_on_goal() {
        let Ok(data) = ArenaData::soccar_from_env() else {
            eprintln!("no meshes; skipped");
            return;
        };
        let dev = Device::open(0).expect("open device 0");
        let gdata = GpuArenaData::upload(&dev, &data).unwrap();
        let num_cars = 2u32;
        // the ball rolling into the blue goal (orange scores), cars parked far from it
        let mut a = crate::cpu::Arena::new(num_cars);
        a.kickoff(1);
        let mut b = a.ball();
        b.body.pos = [0.0, -5000.0, 93.0];
        b.body.vel = [0.0, -2000.0, 0.0];
        b.sleeping = 0;
        a.set_ball(&b);
        let per = crate::cpu::state_words(num_cars) as usize;
        let mut batch = GpuBatch::new(&dev, num_cars, a.words()).unwrap();
        let table = Buffer::from_slice(&dev, &[0f32; 8]).unwrap();
        let rows = Buffer::from_slice(&dev, &[0u8; 2]).unwrap();
        let stop = Buffer::from_slice(&dev, &[0u32]).unwrap();
        let mut cpu = a.clone();
        let mut goal_tick = None;
        for t in 1..=120u32 {
            cpu.step(&data, 1);
            if cpu.take_events() & 3 != 0 {
                goal_tick = Some(t);
                break;
            }
        }
        let goal_tick = goal_tick.expect("the CPU build scores within 120 ticks");
        batch.rollout(&gdata, 120, &rows, &table, &stop).unwrap();
        dev.synchronize().unwrap();
        let mut s = [0u32];
        stop.read(&mut s).unwrap();
        assert_eq!(s[0] & 0xFFFF, goal_tick, "stop tick");
        assert_eq!(s[0] >> 16, 2, "orange scored");
        let mut got = vec![0u32; per];
        batch.read(&mut got).unwrap();
        // the CPU arena's events were taken; the GPU's are kept, so compare everything but the events word
        let diff = got.iter().zip(cpu.words()).filter(|(x, y)| x != y).count();
        assert!(diff <= 1, "frozen state differs from the CPU build in {diff} words");
        batch.rollout(&gdata, 8, &rows, &table, &stop).unwrap();
        dev.synchronize().unwrap();
        stop.read(&mut s).unwrap();
        assert_eq!(s[0], 2 << 16, "an already stopped arena reports 0 and its side");
        let mut again = vec![0u32; per];
        batch.read(&mut again).unwrap();
        assert_eq!(again, got, "a stopped arena does not step");
        eprintln!("goal at tick {goal_tick}: stopped, reported, frozen");
    }

    /// Raw controls: decide with each car's 8 floats equals decide with the same floats as table rows, on every state
    /// word and the goal counts, at 2, 4 and 6 cars, over decisions with play and goal resets.
    #[test]
    fn decide_controls_equal_table_rows() {
        let Ok(data) = ArenaData::soccar_from_env() else {
            eprintln!("no meshes; skipped");
            return;
        };
        let dev = Device::open(0).expect("open device 0");
        let gdata = GpuArenaData::upload(&dev, &data).unwrap();
        // a table of distinct rows: every throttle, steer, pitch, yaw and roll value of -1, 0, 1 and buttons by bits
        let table: Vec<f32> = (0..96u32)
            .flat_map(|r| {
                let f = |k: u32| ((r / 3u32.pow(k)) % 3) as f32 - 1.0;
                let b = |k: u32| ((r >> k) & 1) as f32;
                [f(0), f(1), f(2), f(3), f(4) * 0.5, b(2), b(4), b(6)]
            })
            .collect();
        let tb = Buffer::from_slice(&dev, &table).unwrap();
        for num_cars in [2u32, 4, 6] {
            let n = 64usize;
            let per = crate::cpu::state_words(num_cars) as usize;
            // every other arena with its ball rolling into the blue goal, so the decisions cross goal resets
            let mut arenas = parity_arenas(num_cars, n);
            for a in arenas.iter_mut().step_by(2) {
                let mut b = a.ball();
                b.body.pos = [0.0, -5000.0, 93.0];
                b.body.vel = [0.0, -2000.0, 0.0];
                b.sleeping = 0;
                a.set_ball(&b);
            }
            let words: Vec<u32> = arenas.iter().flat_map(|a| a.words().iter().copied()).collect();
            let mut by_rows = GpuBatch::new(&dev, num_cars, &words).unwrap();
            let mut by_ctl = GpuBatch::new(&dev, num_cars, &words).unwrap();
            let goals_r = Buffer::from_slice(&dev, &vec![0u32; n]).unwrap();
            let goals_c = Buffer::from_slice(&dev, &vec![0u32; n]).unwrap();
            for d in 0..60u32 {
                let rows: Vec<u8> = (0..n * num_cars as usize).map(|k| ((k as u32 * 37 + d * 11) % 96) as u8).collect();
                let ctl: Vec<f32> = rows.iter().flat_map(|&r| table[r as usize * 8..r as usize * 8 + 8].iter().copied()).collect();
                let rb = Buffer::from_slice(&dev, &rows).unwrap();
                let cb = Buffer::from_slice(&dev, &ctl).unwrap();
                by_rows.decide(&gdata, 8, &rb, &tb, &goals_r).unwrap();
                unsafe { by_ctl.decide_controls_raw(&gdata, 8, cb.ptr(), goals_c.ptr(), 0).unwrap() };
                dev.synchronize().unwrap();
            }
            let (mut a, mut b) = (vec![0u32; n * per], vec![0u32; n * per]);
            by_rows.read(&mut a).unwrap();
            by_ctl.read(&mut b).unwrap();
            let (mut ga, mut gb) = (vec![0u32; n], vec![0u32; n]);
            goals_r.read(&mut ga).unwrap();
            goals_c.read(&mut gb).unwrap();
            assert_eq!(a, b, "{num_cars} cars: raw controls differ from table rows");
            assert_eq!(ga, gb, "{num_cars} cars: goal counts differ");
            assert!(ga.iter().sum::<u32>() >= n as u32 / 2, "{num_cars} cars: the rolling balls score");
            eprintln!("{num_cars} cars: 60 decisions of 8 ticks, raw controls == table rows ({} goals)", ga.iter().sum::<u32>());
        }
    }

    /// The oracle check of the P2 gate: the GPU build equals the CPU build on every state word.
    #[test]
    fn gpu_equals_cpu_every_word() {
        let Ok(data) = ArenaData::soccar_from_env() else {
            eprintln!("no meshes; skipped");
            return;
        };
        let dev = Device::open(0).expect("open device 0");
        let gdata = GpuArenaData::upload(&dev, &data).unwrap();
        for num_cars in [2u32, 4, 6] {
            // varied arenas: ball spread over the field, per-arena controls (the synthetic parity set)
            let n = 256usize;
            let mut arenas = parity_arenas(num_cars, n);
            let words: Vec<u32> = arenas.iter().flat_map(|a| a.words().iter().copied()).collect();
            let mut batch = GpuBatch::new(&dev, num_cars, &words).unwrap();
            let ticks = 240;
            batch.step(&dev, &gdata, ticks).unwrap();
            for a in arenas.iter_mut() {
                a.step(&data, ticks);
            }
            let mut got = vec![0u32; words.len()];
            batch.read(&mut got).unwrap();
            let per = words.len() / n;
            let mut bad = 0;
            for (i, a) in arenas.iter().enumerate() {
                let g = &got[i * per..(i + 1) * per];
                if let Some(w) = a.words().iter().zip(g).position(|(x, y)| x != y) {
                    if bad < 5 {
                        eprintln!("{num_cars} cars, arena {i}: first differing word {w}: cpu {:#x} gpu {:#x}", a.words()[w], g[w]);
                    }
                    bad += 1;
                }
            }
            eprintln!("{num_cars} cars: {bad} of {n} arenas differ after {ticks} ticks");
            assert_eq!(bad, 0);
        }
    }

    /// A ball uploaded asleep with a car driving into it: the floor point it is primed with and the manifold it keeps in the
    /// dispatcher's array (the tick's held planes), on the card as on the CPU. The synthetic set uploads no resting ball.
    #[test]
    fn sleeping_uploaded_ball_gpu_equals_cpu_every_word() {
        let Ok(data) = ArenaData::soccar_from_env() else {
            eprintln!("no meshes; skipped");
            return;
        };
        let dev = Device::open(0).expect("open device 0");
        let gdata = GpuArenaData::upload(&dev, &data).unwrap();
        for num_cars in [2u32, 4, 6] {
            let n = 64usize;
            let mut arenas: Vec<Arena> = (0..n)
                .map(|i| {
                    let mut a = Arena::new(num_cars);
                    let mut b = a.ball();
                    b.body.pos = [(i % 4) as f32 * 2.0, 0.0, 92.75];
                    b.body.vel = [0.0; 3];
                    b.body.ang_vel = [0.0; 3];
                    b.sleeping = BallState::uploaded_sleeping(&b.body);
                    assert_eq!(b.sleeping, 3);
                    a.set_ball(&b);
                    for c in 0..num_cars.min(3) {
                        let mut car = a.car(c);
                        let side = if c % 2 == 0 { -1.0 } else { 1.0 };
                        car.body.pos = [-260.0 - 40.0 * c as f32, side * (20.0 + 15.0 * ((i + c as usize) % 5) as f32), 17.0];
                        car.body.vel = [900.0 + 120.0 * (i % 7) as f32, 0.0, 0.0];
                        car.body.ang_vel = [0.0; 3];
                        car.body.rot = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
                        a.set_car(c, &car);
                        a.set_controls(c, &Controls { throttle: 1.0, boost: c % 2, ..Default::default() });
                    }
                    a
                })
                .collect();
            let words: Vec<u32> = arenas.iter().flat_map(|a| a.words().iter().copied()).collect();
            let mut batch = GpuBatch::new(&dev, num_cars, &words).unwrap();
            let ticks = 90;
            batch.step(&dev, &gdata, ticks).unwrap();
            for a in arenas.iter_mut() {
                a.step(&data, ticks);
            }
            let mut got = vec![0u32; words.len()];
            batch.read(&mut got).unwrap();
            let per = words.len() / n;
            let mut bad = 0;
            let mut moved = 0;
            for (i, a) in arenas.iter().enumerate() {
                if a.ball().body.vel != [0.0; 3] {
                    moved += 1;
                }
                let g = &got[i * per..(i + 1) * per];
                if let Some(w) = a.words().iter().zip(g).position(|(x, y)| x != y) {
                    if bad < 5 {
                        eprintln!("{num_cars} cars, arena {i}: first differing word {w}: cpu {:#x} gpu {:#x}", a.words()[w], g[w]);
                    }
                    bad += 1;
                }
            }
            eprintln!("{num_cars} cars: {bad} of {n} arenas differ after {ticks} ticks ({moved} balls were hit)");
            assert!(moved > 0, "the cars reach the ball");
            assert_eq!(bad, 0);
        }
    }

    /// A kickoff on the card, every body preset: the states it writes step exactly as on the CPU, and the cars start at rest
    /// with their own bodies.
    #[test]
    fn kickoff_gpu_equals_cpu_every_word() {
        let Ok(data) = ArenaData::soccar_from_env() else {
            eprintln!("no meshes; skipped");
            return;
        };
        let dev = Device::open(0).expect("open device 0");
        let gdata = GpuArenaData::upload(&dev, &data).unwrap();
        for num_cars in [2u32, 3, 4, 5, 6, 8] {
            let n = 60usize;
            let mut start = parity_arenas(num_cars, n);
            for (i, a) in start.iter_mut().enumerate() {
                for c in 0..num_cars {
                    let mut car = a.car(c);
                    car.preset = (i as u32 + c) % 8;
                    a.set_car(c, &car);
                }
            }
            let words: Vec<u32> = start.iter().flat_map(|a| a.words().iter().copied()).collect();
            let mut batch = GpuBatch::new_with_psyclops(&dev, num_cars, &words, true).unwrap();
            batch.kickoff_all(7).unwrap();
            dev.synchronize().unwrap();
            let mut kicked = vec![0u32; words.len()];
            batch.read(&mut kicked).unwrap();
            let per = words.len() / n;
            let mut arenas: Vec<Arena> = kicked.chunks(per).map(|w| Arena::from_words(num_cars, w)).collect();
            for (i, a) in arenas.iter().enumerate() {
                for c in 0..num_cars {
                    assert_eq!(a.car(c).preset, (i as u32 + c) % 8, "arena {i} car {c}: body changed by the kickoff");
                }
            }
            let ticks = 240;
            batch.step(&dev, &gdata, ticks).unwrap();
            for a in arenas.iter_mut() {
                a.step(&data, ticks);
            }
            let mut got = vec![0u32; words.len()];
            batch.read(&mut got).unwrap();
            let bad = arenas.iter().enumerate().filter(|(i, a)| a.words() != &got[i * per..(i + 1) * per]).count();
            eprintln!("{num_cars} cars: {bad} of {n} arenas differ {ticks} ticks after a kickoff");
            assert_eq!(bad, 0);
        }
    }
}
