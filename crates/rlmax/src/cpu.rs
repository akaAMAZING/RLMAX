//! The CPU build of the physics, through the C ABI in `src/capi/rlmax.h`.

use crate::arena::{self, GridSpec, Tri};
use std::path::Path;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Controls {
    pub throttle: f32,
    pub steer: f32,
    pub pitch: f32,
    pub yaw: f32,
    pub roll: f32,
    pub jump: u32,
    pub boost: u32,
    pub handbrake: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BodyState {
    pub pos: [f32; 3],
    /// Columns: forward, right, up.
    pub rot: [f32; 9],
    pub vel: [f32; 3],
    pub ang_vel: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BallState {
    pub body: BodyState,
    pub sleeping: u32,
    pub deactivation_time: f32,
}

/// The height a ball rests at awake, uu (consts.h BALL_REST_Z_UU).
pub const BALL_REST_Z_UU: f32 = 93.15;

impl BallState {
    /// The activation a ball uploaded from a recorded frame takes, which carries no activation state: a ball not exactly still is
    /// awake (0); one exactly still has lain where it is, and below the awake rest height less 0.2 uu (the kickoff ball sits 0.4 inside
    /// it, which a simulated ball cannot hold) it is asleep and keeps its floor point (3, uploaded); at the rest height it builds its
    /// floor contacts on the first tick and then sleeps (1).
    pub fn uploaded_sleeping(body: &BodyState) -> u32 {
        if body.vel != [0.0; 3] || body.ang_vel != [0.0; 3] {
            0
        } else if body.pos[2] < BALL_REST_Z_UU - 0.2 {
            3
        } else {
            1
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CarState {
    pub body: BodyState,
    pub controls: Controls,
    pub flags: u32,
    pub team: u32,
    /// Body preset: 0 Octane, 1 Dominus, 2 Plank, 3 Breakout, 4 Hybrid, 5 Merc, 6 Batmobile, 7 Psyclops.
    pub preset: u32,
    pub boost: f32,
    pub flip_torque: [f32; 2],
    pub jump_ticks: u32,
    pub flip_time: f32,
    pub air_time: f32,
    pub air_time_since_jump: f32,  // C field time_since_jump (since the last jump, on the ground too); name kept for external harnesses
    pub handbrake_val: f32,
    pub supersonic_timer: f32,
    pub time_since_boosted: f32,
    pub boosting_time: f32,
    pub auto_flip_timer: f32,
    pub auto_flip_torque_scale: f32,
    pub bump_cooldown: f32,
    pub bump_victim: u32,
    pub last_extra_hit_tick: u32,
    pub demo_respawn_timer: f32,
    pub wheel_normal_sum: [f32; 3],
    pub world_contact_normal: [f32; 3],
}

pub mod flags {
    pub const HAS_JUMPED: u32 = 1 << 0;
    pub const HAS_DOUBLE_JUMPED: u32 = 1 << 1;
    pub const HAS_FLIPPED: u32 = 1 << 2;
    pub const IS_FLIPPING: u32 = 1 << 3;
    pub const IS_JUMPING: u32 = 1 << 4;
    pub const JUMP_HOLD_BROKEN: u32 = 1 << 5;
    pub const IS_AUTO_FLIPPING: u32 = 1 << 6;
    pub const IS_SUPERSONIC: u32 = 1 << 7;
    pub const IS_BOOSTING: u32 = 1 << 8;
    pub const IS_DEMOED: u32 = 1 << 9;
    pub const ON_GROUND: u32 = 1 << 10;
    pub const STICKY_GATE: u32 = 1 << 11;
    pub const WORLD_CONTACT: u32 = 1 << 12;
    pub const PREV_JUMP: u32 = 1 << 13;
    pub const HAS_EXTRA_HIT: u32 = 1 << 14;
    pub const HAS_BUMP_VICTIM: u32 = 1 << 15;
    pub const WHEEL0: u32 = 1 << 16;
}

#[repr(C)]
struct RawArenaData {
    tris: *const std::ffi::c_void,
    cell_off: *const u32,
    cell_idx: *const u16,
    goal_ly: *const f32,
    tri_box: *const f32,
    grid_min: [f32; 3],
    inv_cell: f32,
    nx: i32,
    ny: i32,
    nz: i32,
    max_cell_count: u32,
}

unsafe extern "C" {
    fn rlmax_abi_version() -> u32;
    fn rlmax_state_words(num_cars: u32) -> u32;
    fn rlmax_staged_words(num_cars: u32) -> u32;
    fn rlmax_kickoff(num_cars: u32, state: *mut u32, seed: u32) -> i32;
    fn rlmax_init_arena(num_cars: u32, state: *mut u32) -> i32;
    fn rlmax_get_ball(num_cars: u32, state: *const u32, out: *mut BallState) -> i32;
    fn rlmax_set_ball(num_cars: u32, state: *mut u32, input: *const BallState) -> i32;
    fn rlmax_get_car(num_cars: u32, state: *const u32, car: u32, out: *mut CarState) -> i32;
    fn rlmax_get_bt(num_cars: u32, state: *const u32, body: u32, out: *mut f32) -> i32;
    fn rlmax_set_boost(num_cars: u32, state: *mut u32, car: u32, boost: f32) -> i32;
    fn rlmax_set_car(num_cars: u32, state: *mut u32, car: u32, input: *const CarState) -> i32;
    fn rlmax_set_body_basis(num_cars: u32, state: *mut u32, body: u32, basis: *const f32, inverse: *const f32) -> i32;
    fn rlmax_set_pad_frame_fraction(num_cars: u32, state: *mut u32, fraction: f32) -> i32;
    fn rlmax_set_controls(num_cars: u32, state: *mut u32, car: u32, input: *const Controls) -> i32;
    fn rlmax_tick_count(num_cars: u32, state: *const u32) -> u32;
    fn rlmax_events(num_cars: u32, state: *mut u32, clear: i32) -> u32;
    fn rlmax_contact_census(num_cars: u32, state: *const u32) -> u32;
    fn rlmax_local_sizes(num_cars: u32, out: *mut u32) -> i32;
    fn rlmax_refresh_sticky_gate(num_cars: u32, state: *mut u32, car: u32, arena: *const RawArenaData) -> i32;
    fn rlmax_refresh_sticky_gate_with_wheels(num_cars: u32, state: *mut u32, car: u32, wheels: u32, arena: *const RawArenaData) -> i32;
    #[cfg(feature = "full_state")]
    fn rlmax_seed_handle(num_cars: u32, state: *mut u32, body: u32, handle: u32) -> i32;
    fn rlmax_seed_wheel_contacts(
        num_cars: u32,
        state: *mut u32,
        car: u32,
        wheel_mask: u32,
        normals: *const f32,
        arena: *const RawArenaData,
    ) -> i32;
    fn rlmax_debug_car_car(num_cars: u32, state: *const u32, out: *mut f32) -> i32;
    fn rlmax_debug_car_world(num_cars: u32, state: *const u32, car: u32, arena: *const RawArenaData, out: *mut f32) -> i32;
    fn rlmax_debug_ball_world(
        num_cars: u32,
        state: *const u32,
        arena: *const RawArenaData,
        raw: *mut f32,
        raw_cap: u32,
        raw_n: *mut u32,
        manifold: *mut f32,
        cell_count: *mut u32,
    ) -> i32;
    fn rlmax_debug_car_ball(num_cars: u32, state: *const u32, car: u32, out: *mut f32) -> i32;
    fn rlmax_debug_car_wman(num_cars: u32, state: *const u32, car: u32, out: *mut f32) -> i32;
    fn rlmax_debug_pair_man(num_cars: u32, state: *const u32, out: *mut f32) -> i32;
    #[cfg(feature = "impulse_log")]
    fn rlmax_impulse_log(car: u32, out: *mut f32) -> i32;
    #[cfg(feature = "impulse_log")]
    fn rlmax_wheel_log(car: u32, out: *mut f32);
    #[cfg(feature = "impulse_log")]
    fn rlmax_seed_basis(car: u32, m: *const f32);
    #[cfg(feature = "full_state")]
    fn rlmax_fs_car(num_cars: u32, state: *mut u32, car: u32, f: *const f32) -> i32;
    #[cfg(feature = "full_state")]
    fn rlmax_fs_ball(num_cars: u32, state: *mut u32, f: *const f32) -> i32;
    #[cfg(feature = "full_state")]
    fn rlmax_fs_manifold(num_cars: u32, state: *mut u32, kind: u32, a: u32, obj: u32, count: u32, pts: *const f32) -> i32;
    #[cfg(feature = "full_state")]
    fn rlmax_fs_visit(num_cars: u32, state: *mut u32, packed: u32, present: u32) -> i32;
    #[cfg(feature = "full_state")]
    fn rlmax_fs_sweep(num_cars: u32, state: *mut u32, f: *const f32) -> i32;
    #[cfg(feature = "full_state")]
    fn rlmax_fs_car_extra(num_cars: u32, state: *mut u32, car: u32, e: *const f32) -> i32;
    #[cfg(feature = "full_state")]
    fn rlmax_fs_spawn_z(car: u32, z: f32);
    #[cfg(feature = "full_state")]
    fn rlmax_fs_jump(num_cars: u32, state: *mut u32, car: u32, jumped: f32, used: f32, inactive: f32, ground: f32) -> i32;
    #[cfg(feature = "full_state")]
    fn rlmax_fs_block_pads(num_cars: u32, state: *mut u32) -> i32;
    #[cfg(feature = "full_state")]
    fn rlmax_fs_free(num_cars: u32, state: *mut u32, n: u32, f: *const f32) -> i32;
    fn rlmax_cpu_step(num_cars: u32, states: *mut u32, num_arenas: u32, ticks: u32, arena: *const RawArenaData) -> i32;
    fn rlmax_extract_row_floats(num_cars: u32) -> u32;
    fn rlmax_extract(num_cars: u32, state: *const u32, out: *mut f32) -> i32;
    fn rlmax_extract_fields(which: i32) -> *const std::ffi::c_char;
}

pub fn abi_version() -> u32 {
    unsafe { rlmax_abi_version() }
}

/// Floats in one arena's extract row for `num_cars` cars (src/physics/extract.h; 0 when unsupported).
pub fn extract_row_floats(num_cars: u32) -> u32 {
    unsafe { rlmax_extract_row_floats(num_cars) }
}

/// The extract row's field names in order, from the header's field lists: the arena block, the 34 pad timers
/// (pad_0 .. pad_33), then one car block per car, its names prefixed `car{i}_`.
pub fn extract_fields(num_cars: u32) -> Vec<String> {
    let list = |which| {
        let s = unsafe { std::ffi::CStr::from_ptr(rlmax_extract_fields(which)) };
        s.to_str().expect("ascii").split(',').filter(|f| !f.is_empty()).map(str::to_string).collect::<Vec<_>>()
    };
    let mut out = list(0);
    out.extend((0..34).map(|p| format!("pad_{p}")));
    let car = list(1);
    for i in 0..num_cars {
        out.extend(car.iter().map(|f| format!("car{i}_{f}")));
    }
    debug_assert_eq!(out.len(), extract_row_floats(num_cars) as usize);
    out
}

/// 32-bit words in one arena's state for `num_cars` cars (0 when unsupported).
pub(crate) fn staged_words(num_cars: u32) -> u32 { unsafe { rlmax_staged_words(num_cars) } }

pub fn state_words(num_cars: u32) -> u32 {
    unsafe { rlmax_state_words(num_cars) }
}

/// Grid parameters. A cell lists every triangle within reach of any point in it, so a smaller cell and a shorter reach
/// mean fewer triangles tested per lookup; results don't change (strict suite 121 of 121 at 16 uu with 96 uu reach, 5 Nov).
/// 16 uu cells (5090, 5 Nov, M arena-ticks/s, 32 / 16 uu: 1v1 1,024 5.92 / 6.10, 8,192 24.56 / 24.86, 65,536 42.66 / 42.92;
/// 3080 8,192 8.33 / 8.48), about 240 MiB more on the card than 32 uu and a few seconds more to build. Finer still gains
/// 1-2% and costs far more: 8 uu took about 30 s to build. 96 uu reach: the ball's radius and contact margin, about 95.2
/// uu, is the farthest any lookup needs; 92 uu changes results.
pub const CELL_UU: f32 = 16.0;
pub const REACH_UU: f32 = 96.0;

/// The arena's static data, owned on the host.
pub struct ArenaData {
    pub tris: Vec<Tri>,
    pub cell_off: Vec<u32>,
    pub cell_idx: Vec<u16>,
    /// The goal triangles' local-frame vertex y, three per triangle (`arena::load_soccar_goal`)
    pub goal_ly: Vec<f32>,
    /// Each triangle's BVH leaf box, six floats (`arena::load_soccar_goal`)
    pub tri_box: Vec<f32>,
    pub spec: GridSpec,
    pub max_cell_count: u32,
}

impl ArenaData {
    pub fn soccar(mesh_dir: &Path) -> Result<ArenaData, String> {
        let (tris, goal_ly, tri_box) = arena::load_soccar_goal(mesh_dir)?;
        // RLMAX_CELL_UU, RLMAX_REACH_UU: another grid for measurement
        let cell = std::env::var("RLMAX_CELL_UU").ok().and_then(|v| v.parse().ok()).unwrap_or(CELL_UU);
        let reach = std::env::var("RLMAX_REACH_UU").ok().and_then(|v| v.parse().ok()).unwrap_or(REACH_UU);
        let spec = GridSpec::soccar(cell, reach);
        let lists = arena::build_cell_lists(&tris, &spec);
        let mut cell_off = Vec::with_capacity(lists.len() + 1);
        let mut cell_idx = Vec::new();
        let mut max = 0;
        for l in &lists {
            cell_off.push(cell_idx.len() as u32);
            max = max.max(l.len() as u32);
            cell_idx.extend(l.iter().map(|&t| t as u16));
        }
        cell_off.push(cell_idx.len() as u32);
        Ok(ArenaData { tris, cell_off, cell_idx, goal_ly, tri_box, spec, max_cell_count: max })
    }

    /// `$RLMAX_DATA/meshes/soccar`
    pub fn soccar_from_env() -> Result<ArenaData, String> {
        let dir = std::env::var_os("RLMAX_DATA").ok_or("RLMAX_DATA is not set")?;
        Self::soccar(&Path::new(&dir).join("meshes").join("soccar"))
    }

    fn raw(&self) -> RawArenaData {
        RawArenaData {
            tris: self.tris.as_ptr().cast(),
            cell_off: self.cell_off.as_ptr(),
            cell_idx: self.cell_idx.as_ptr(),
            goal_ly: self.goal_ly.as_ptr(),
            tri_box: self.tri_box.as_ptr(),
            grid_min: self.spec.min,
            inv_cell: 1.0 / self.spec.cell,
            nx: self.spec.dims[0] as i32,
            ny: self.spec.dims[1] as i32,
            nz: self.spec.dims[2] as i32,
            max_cell_count: self.max_cell_count,
        }
    }
}

/// One captured impulse source (lead 56): the summed velocity delta and how many adds made it.
#[cfg(feature = "impulse_log")]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ImpEntry {
    pub lin: [f32; 3],
    pub ang: [f32; 3],
    pub n: u32,
}

/// One car's impulse capture for one tick. `src[s][accum]`: s indexes SOURCE_NAMES; `accum` 0 = applied to the
/// velocity directly, 1 = accumulated for the solver. `wheel[0]` suspension and `wheel[1]` friction, per wheel.
#[cfg(feature = "impulse_log")]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ImpulseLog {
    pub src: [[ImpEntry; 2]; ImpulseLog::SOURCES],
    pub wheel: [[ImpEntry; 4]; 2],
}

#[cfg(feature = "impulse_log")]
impl ImpulseLog {
    pub const SOURCES: usize = 11;
    pub const FLOATS: usize = (Self::SOURCES * 2 + 8) * 7;
    /// RLMAX source -> the game's ImpulseRecordType (src/physics/impulse_log.h): suspension WheelsSuspension, friction
    /// WheelsFriction, sticky StickyForce, jump Jump, djump DoubleJump, dodge DodgeImpulse, dtorque DodgeTorque, air
    /// AirControl, boost Boost; autoroll and autoflip have no game type known.
    pub const SOURCE_NAMES: [&'static str; Self::SOURCES] =
        ["susp", "fric", "sticky", "jump", "djump", "dodge", "dtorque", "air", "boost", "autoroll", "autoflip"];
}

/// One arena on the CPU build.
#[derive(Clone)]
pub struct Arena {
    pub num_cars: u32,
    words: Vec<u32>,
}

impl Arena {
    pub fn new(num_cars: u32) -> Arena {
        let n = unsafe { rlmax_state_words(num_cars) } as usize;
        assert!(n > 0, "unsupported car count {num_cars}");
        let mut words = vec![0u32; n];
        unsafe { rlmax_init_arena(num_cars, words.as_mut_ptr()) };
        Arena { num_cars, words }
    }

    /// An arena from saved state words (`words().len()` for this car count).
    pub fn from_words(num_cars: u32, words: &[u32]) -> Arena {
        assert_eq!(words.len(), state_words(num_cars) as usize, "state words for {num_cars} cars");
        Arena { num_cars, words: words.to_vec() }
    }

    /// Reset to a kickoff with the arena's rng set to `seed`.
    pub fn kickoff(&mut self, seed: u32) {
        unsafe { rlmax_kickoff(self.num_cars, self.words.as_mut_ptr(), seed) };
    }

    pub fn ball(&self) -> BallState {
        let mut b = BallState::default();
        unsafe { rlmax_get_ball(self.num_cars, self.words.as_ptr(), &mut b) };
        b
    }

    pub fn set_ball(&mut self, b: &BallState) {
        unsafe { rlmax_set_ball(self.num_cars, self.words.as_mut_ptr(), b) };
    }

    pub fn car(&self, i: u32) -> CarState {
        let mut c = CarState::default();
        unsafe { rlmax_get_car(self.num_cars, self.words.as_ptr(), i, &mut c) };
        c
    }

    /// Position and velocity of a body (cars, then the ball at `num_cars`) as held, in Bullet units: the bits to compare with the game's.
    pub fn body_bt(&self, i: u32) -> [f32; 6] {
        let mut r = [0f32; 6];
        unsafe { rlmax_get_bt(self.num_cars, self.words.as_ptr(), i, r.as_mut_ptr()) };
        r
    }

    /// A car's boost alone; [`Arena::set_car`] would also take its body through uu and move it by a few ulp.
    pub fn set_boost(&mut self, i: u32, boost: f32) {
        unsafe { rlmax_set_boost(self.num_cars, self.words.as_mut_ptr(), i, boost) };
    }

    pub fn set_car(&mut self, i: u32, c: &CarState) {
        unsafe { rlmax_set_car(self.num_cars, self.words.as_mut_ptr(), i, c) };
    }

    /// Set the exact basis and optional inverse inertia rows for a car, or the ball at `num_cars`.
    /// The matrix stays in arena words until that body integrates.
    pub fn set_body_basis(&mut self, body: u32, basis: &[f32; 9], inverse: Option<&[f32; 9]>) {
        assert!(body <= self.num_cars);
        let inv = inverse.map_or(std::ptr::null(), |m| m.as_ptr());
        assert_eq!(unsafe { rlmax_set_body_basis(self.num_cars, self.words.as_mut_ptr(), body, basis.as_ptr(), inv) }, 0);
    }

    /// An arena from physical values (start states for [`crate::gpu::GpuBatch::write`], e.g. from replays): the ball
    /// and each car as given, in uu and uu/s (set-state divides by 50, the inverse of the game's BT x 50). Then the
    /// rules for what a recording does not carry, as the RLPR backend applies them: the ball as `BallState::uploaded_sleeping` has it (a resting ball of a running game has lain on the floor), awake if moving, each car's sticky-force gate rebuilt from
    /// its wheels' rays at this pose, and the contact manifolds, broadphase pairs and pad timers empty (rebuilt within
    /// a tick or two). Fields a replay lacks keep their `Default` (0, not jumping, no flip); start from frames where
    /// they do not matter, or from live snapshots, whose hidden state is exact.
    pub fn from_physical(arena: &ArenaData, ball: &BallState, cars: &[CarState]) -> Arena {
        let mut a = Arena::new(cars.len() as u32);
        let mut b = *ball;
        b.sleeping = BallState::uploaded_sleeping(&b.body);
        a.set_ball(&b);
        for (i, c) in cars.iter().enumerate() {
            a.set_car(i as u32, c);
            a.refresh_sticky_gate(arena, i as u32);
        }
        a
    }

    /// Test-only (feature `full_state`): car `i`'s recorded body block of 28 floats, in Bullet units: pos, vel, ang vel,
    /// basis columns forward/right/up, inverse inertia rows, valid flag. Call after `set_car`.
    #[cfg(feature = "full_state")]
    pub fn fs_car(&mut self, i: u32, f: &[f32]) {
        assert!(f.len() >= 28);
        unsafe { rlmax_fs_car(self.num_cars, self.words.as_mut_ptr(), i, f.as_ptr()) };
    }

    /// Test-only: the ball's block of 28 floats as `fs_car`, then sleeping state and deactivation time. Call after `set_ball`.
    #[cfg(feature = "full_state")]
    pub fn fs_ball(&mut self, f: &[f32]) {
        assert!(f.len() >= 30);
        unsafe { rlmax_fs_ball(self.num_cars, self.words.as_mut_ptr(), f.as_ptr()) };
    }

    /// Test-only: a recorded manifold. `kind` 0 car `a`'s world manifold, 1 the ball's world manifold (`obj` the mesh index, or
    /// 0x100 + plane), 2 car pair `a`, 3 ball against car `a`; `pts` is 10 floats per point: point on A (A's frame), point on B
    /// (B's frame), normal on B, distance. Call after `fs_car`/`fs_ball`, which empty the world manifolds.
    #[cfg(feature = "full_state")]
    pub fn fs_manifold(&mut self, kind: u32, a: u32, obj: u32, pts: &[f32]) -> i32 {
        unsafe { rlmax_fs_manifold(self.num_cars, self.words.as_mut_ptr(), kind, a, obj, (pts.len() / 10) as u32, pts.as_ptr()) }
    }

    /// Test-only: the free handles of a recorded state whose broadphase can't be loaded, popped last first.
    #[cfg(feature = "full_state")]
    pub fn fs_free(&mut self, f: &[f32]) -> i32 {
        unsafe { rlmax_fs_free(self.num_cars, self.words.as_mut_ptr(), f.len() as u32, f.as_ptr()) }
    }

    /// Test-only: no pad can give on the next step, on the card too (`rlmax_fs_block_pads`).
    #[cfg(feature = "full_state")]
    pub fn fs_block_pads(&mut self) {
        unsafe { rlmax_fs_block_pads(self.num_cars, self.words.as_mut_ptr()) };
    }

    /// The state words, to carry the arena to and from the card.
    pub fn words_mut(&mut self) -> &mut [u32] {
        &mut self.words
    }

    /// Test-only: car `i`'s recorded jump state.
    #[cfg(feature = "full_state")]
    pub fn fs_jump(&mut self, i: u32, jumped: f32, used: f32, inactive: f32, ground: f32) {
        unsafe { rlmax_fs_jump(self.num_cars, self.words.as_mut_ptr(), i, jumped, used, inactive, ground) };
    }

    /// Test-only: car `i`'s recorded spawn height, uu.
    #[cfg(feature = "full_state")]
    pub fn fs_spawn_z(&mut self, i: u32, z: f32) {
        unsafe { rlmax_fs_spawn_z(i, z) };
    }

    /// Test-only: car `i`'s further hidden scalars (layout in src/cpu/full_state.h rlmax_fs_car_extra). Call after `set_car`.
    #[cfg(feature = "full_state")]
    pub fn fs_car_extra(&mut self, i: u32, e: &[f32]) {
        unsafe { rlmax_fs_car_extra(self.num_cars, self.words.as_mut_ptr(), i, e.as_ptr()) };
    }

    /// Test-only: the recorded car visit order (src/cpu/full_state.h rlmax_fs_visit).
    #[cfg(feature = "full_state")]
    pub fn fs_visit(&mut self, packed: u32, present: u32) -> i32 {
        unsafe { rlmax_fs_visit(self.num_cars, self.words.as_mut_ptr(), packed, present) }
    }

    /// Test-only: the recorded broadphase (layout in src/cpu/full_state.h rlmax_fs_sweep). Call last, after every body is placed.
    #[cfg(feature = "full_state")]
    pub fn fs_sweep(&mut self, f: &[f32]) -> i32 {
        unsafe { rlmax_fs_sweep(self.num_cars, self.words.as_mut_ptr(), f.as_ptr()) }
    }

    /// Set the render-frame fraction; negative means an earlier tick of the frame.
    pub fn set_pad_frame_fraction(&mut self, fraction: f32) {
        assert!(fraction.is_finite() && fraction <= 1.0);
        assert_eq!(unsafe { rlmax_set_pad_frame_fraction(self.num_cars, self.words.as_mut_ptr(), fraction) }, 0);
    }

    pub fn set_controls(&mut self, i: u32, c: &Controls) {
        unsafe { rlmax_set_controls(self.num_cars, self.words.as_mut_ptr(), i, c) };
    }

    pub fn tick_count(&self) -> u32 {
        unsafe { rlmax_tick_count(self.num_cars, self.words.as_ptr()) }
    }

    pub fn take_events(&mut self) -> u32 {
        unsafe { rlmax_events(self.num_cars, self.words.as_mut_ptr(), 1) }
    }

    pub fn step(&mut self, arena: &ArenaData, ticks: u32) {
        let raw = arena.raw();
        unsafe { rlmax_cpu_step(self.num_cars, self.words.as_mut_ptr(), 1, ticks, &raw) };
    }

    /// Lead 56 debug capture (cargo feature `impulse_log`): car `i`'s per-source velocity deltas of the last tick
    /// stepped, in the units of the game's impulse records (BT velocity, rad/s).
    /// Debug: car `i`'s wheels at the suspension step of the last tick: hit, length, relative velocity, normal, steer, engine, brake, lateral and long friction, pushback.
    #[cfg(feature = "impulse_log")]
    pub fn seed_basis(&self, i: u32, m: [f32; 9]) { unsafe { rlmax_seed_basis(i, m.as_ptr()) } }

    #[cfg(feature = "impulse_log")]
    pub fn wheel_log(&self, i: u32) -> [[f32; 12]; 4] {
        let mut r = [[0f32; 12]; 4];
        unsafe { rlmax_wheel_log(i, r.as_mut_ptr() as *mut f32) };
        r
    }

    #[cfg(feature = "impulse_log")]
    pub fn impulse_log(&self, i: u32) -> ImpulseLog {
        let mut raw = [0f32; ImpulseLog::FLOATS];
        unsafe { rlmax_impulse_log(i, raw.as_mut_ptr()) };
        let ent = |k: usize| {
            let r = &raw[k * 7..k * 7 + 7];
            ImpEntry { lin: [r[0], r[1], r[2]], ang: [r[3], r[4], r[5]], n: r[6] as u32 }
        };
        let mut log = ImpulseLog::default();
        for s in 0..ImpulseLog::SOURCES {
            for a in 0..2 {
                log.src[s][a] = ent(s * 2 + a);
            }
        }
        let base = ImpulseLog::SOURCES * 2;
        for k in 0..2 {
            for w in 0..4 {
                log.wheel[k][w] = ent(base + k * 4 + w);
            }
        }
        log
    }

    pub fn refresh_sticky_gate(&mut self, arena: &ArenaData, i: u32) {
        let raw = arena.raw();
        unsafe { rlmax_refresh_sticky_gate(self.num_cars, self.words.as_mut_ptr(), i, &raw) };
    }

    /// Rebuild previous world-wheel contacts from pose rays and known previous wheel-contact bits.
    /// Rays are cast at the pose one tick back (x - v dt, basis turned back by w dt), where the game cast them.
    /// Place every body first. Bits 0..3 of `wheels` identify wheels in RLMAX order that were in contact.
    pub fn refresh_sticky_gate_with_wheels(&mut self, arena: &ArenaData, i: u32, wheels: u32) {
        let raw = arena.raw();
        unsafe { rlmax_refresh_sticky_gate_with_wheels(self.num_cars, self.words.as_mut_ptr(), i, wheels, &raw) };
    }

    /// Seed car `i`'s sticky gate and wheel normals from a recorded state's previous-step wheel contacts (bit per wheel in
    /// `mask`, four contact normals in RLMAX's wheel order), keeping the world contacts.
    /// Place every body first. Only for a harness setting a state; the step is untouched.
    #[doc(hidden)]
    #[cfg(feature = "full_state")]
    pub fn seed_handle(&mut self, body: u32, handle: u32) {
        unsafe { rlmax_seed_handle(self.num_cars, self.words.as_mut_ptr(), body, handle) };
    }

    pub fn seed_wheel_contacts(&mut self, arena: &ArenaData, i: u32, mask: u32, normals: &[[f32; 3]; 4]) {
        let raw = arena.raw();
        unsafe { rlmax_seed_wheel_contacts(self.num_cars, self.words.as_mut_ptr(), i, mask, normals.as_ptr().cast(), &raw) };
    }

    /// Debug: RLMAX's car-car contacts for cars 0 and 1 at the current pose: (point on B, normal on B, depth, ok) in uu.
    pub fn debug_car_car(&self) -> [[f32; 8]; 4] {
        let mut out = [[0.0f32; 8]; 4];
        unsafe { rlmax_debug_car_car(self.num_cars, self.words.as_ptr(), out.as_mut_ptr().cast()) };
        out
    }

    /// Debug: RLMAX's car-world contacts for one car at the current pose (same layout as `debug_car_car`).
    pub fn debug_car_world(&self, arena: &ArenaData, i: u32) -> [[f32; 8]; 4] {
        let mut out = [[0.0f32; 8]; 4];
        let raw = arena.raw();
        unsafe { rlmax_debug_car_world(self.num_cars, self.words.as_ptr(), i, &raw, out.as_mut_ptr().cast()) };
        out
    }

    /// Debug: RLMAX's ball-world contacts at the current pose, as the tick builds them: every contact found (point uu,
    /// normal, depth uu, group, triangle), the manifold points kept (last word 1 when valid), and the cell's triangle count.
    pub fn debug_ball_world(&self, arena: &ArenaData) -> (Vec<[f32; 9]>, [[f32; 9]; 16], u32) {
        let mut found = vec![[0.0f32; 9]; 128];
        let mut kept = [[0.0f32; 9]; 16];
        let (mut n, mut cell) = (0u32, 0u32);
        let raw = arena.raw();
        unsafe {
            rlmax_debug_ball_world(
                self.num_cars,
                self.words.as_ptr(),
                &raw,
                found.as_mut_ptr().cast(),
                found.len() as u32,
                &mut n,
                kept.as_mut_ptr().cast(),
                &mut cell,
            )
        };
        found.truncate(n as usize);
        (found, kept, cell)
    }

    /// Debug: RLMAX's raw car-ball contact for one car at the current pose (same layout as `debug_car_car`).
    pub fn debug_car_ball(&self, i: u32) -> [f32; 8] {
        let mut out = [0.0f32; 8];
        unsafe { rlmax_debug_car_ball(self.num_cars, self.words.as_ptr(), i, out.as_mut_ptr()) };
        out
    }

    /// One car's world manifolds after the last tick: 16 slots of (point on the world uu, normal, distance uu,
    /// static object, ok).
    pub fn debug_car_wman(&self, i: u32) -> [[f32; 9]; 16] {
        let mut out = [[0.0f32; 9]; 16];
        unsafe { rlmax_debug_car_wman(self.num_cars, self.words.as_ptr(), i, out.as_mut_ptr().cast()) };
        out
    }

    /// The car pair (0, 1) manifold after the last tick: 4 slots of (point on car 1 uu, normal, distance uu, ok).
    pub fn debug_pair_man(&self) -> [[f32; 8]; 4] {
        let mut out = [[0.0f32; 8]; 4];
        unsafe { rlmax_debug_pair_man(self.num_cars, self.words.as_ptr(), out.as_mut_ptr().cast()) };
        out
    }

    /// This arena's extract row (the GPU kernel's oracle).
    pub fn extract(&self) -> Vec<f32> {
        let mut out = vec![0f32; extract_row_floats(self.num_cars) as usize];
        unsafe { rlmax_extract(self.num_cars, self.words.as_ptr(), out.as_mut_ptr()) };
        out
    }

    pub fn words(&self) -> &[u32] {
        &self.words
    }
}

// ---- case files for C++ case runners: everything little-endian

fn put_u32s(out: &mut Vec<u8>, v: &[u32]) {
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
}

impl ArenaData {
    /// The arena data as one `arena.bin` for C++ case runners.
    pub fn write_blob(&self, path: &Path) -> std::io::Result<()> {
        let raw = self.raw();
        let mut b = Vec::new();
        put_u32s(&mut b, &[0x414d_4c52, 1, self.tris.len() as u32, self.cell_off.len() as u32, self.cell_idx.len() as u32]);
        put_u32s(&mut b, &raw.grid_min.map(f32::to_bits));
        put_u32s(&mut b, &[raw.inv_cell.to_bits(), raw.nx as u32, raw.ny as u32, raw.nz as u32, raw.max_cell_count]);
        for t in &self.tris {
            let mut w = Vec::with_capacity(16);
            w.extend(t.p.iter().flatten().chain(&t.n).chain(&t.ang).map(|f| f.to_bits()));
            w.push(t.flags);
            put_u32s(&mut b, &w);
        }
        put_u32s(&mut b, &self.cell_off);
        for i in &self.cell_idx {
            b.extend_from_slice(&i.to_le_bytes());
        }
        std::fs::write(path, b)
    }
}

/// One arena's workload census after a tick: whether any car holds a contact with the ball or
/// with another car (their persistent manifolds are not empty), and how many cars are airborne (not on the ground) or
/// on a wall or the ceiling (on the ground with the up axis below 0.7 in z). The same function reads the recorded
/// games and the bench workloads, so their rows compare.
#[derive(Debug, Clone, Copy, Default)]
pub struct Census {
    pub car_ball: bool,
    pub car_car: bool,
    pub airborne: u32,
    pub wall: u32,
}

pub fn census(num_cars: u32, words: &[u32]) -> Census {
    assert_eq!(words.len(), state_words(num_cars) as usize, "one arena's words");
    let bits = unsafe { rlmax_contact_census(num_cars, words.as_ptr()) };
    let arena = Arena::from_words(num_cars, words);
    let mut c = Census { car_ball: bits & 1 != 0, car_car: bits & 2 != 0, ..Census::default() };
    for i in 0..num_cars {
        let car = arena.car(i);
        if car.flags & flags::IS_DEMOED != 0 {
            continue;
        }
        if car.flags & flags::ON_GROUND == 0 {
            c.airborne += 1;
        } else if car.body.rot[8] < 0.7 {
            c.wall += 1;
        }
    }
    c
}

/// Shares over many arena-ticks: car-ball and car-car per arena-tick, airborne and wall per car-tick.
#[derive(Debug, Clone, Copy, Default)]
pub struct CensusSum {
    pub arena_ticks: u64,
    pub car_ticks: u64,
    pub car_ball: u64,
    pub car_car: u64,
    pub airborne: u64,
    pub wall: u64,
}

impl CensusSum {
    pub fn add(&mut self, num_cars: u32, c: Census) {
        self.arena_ticks += 1;
        self.car_ticks += num_cars as u64;
        self.car_ball += c.car_ball as u64;
        self.car_car += c.car_car as u64;
        self.airborne += c.airborne as u64;
        self.wall += c.wall as u64;
    }

    /// `{"samples": .., "car_ball": .., "car_car": .., "airborne": .., "wall": ..}`
    pub fn json(&self) -> String {
        let (a, c) = (self.arena_ticks.max(1) as f64, self.car_ticks.max(1) as f64);
        format!(
            "{{\"samples\": {}, \"car_ball\": {:.4}, \"car_car\": {:.4}, \"airborne\": {:.4}, \"wall\": {:.4}}}",
            self.arena_ticks,
            self.car_ball as f64 / a,
            self.car_car as f64 / a,
            self.airborne as f64 / c,
            self.wall as f64 / c
        )
    }
}

/// The tick's whitelisted local-memory arrays in bytes per thread: EPA scratch, solver, contact points, per-body
/// arrays.
pub fn local_sizes(num_cars: u32) -> [u32; 4] {
    let mut out = [0u32; 4];
    assert_eq!(unsafe { rlmax_local_sizes(num_cars, out.as_mut_ptr()) }, 0, "car count");
    out
}

/// A `case.bin` for C++ case runners: `arenas` as they start, the tick count, and their words after `ticks` ticks of
/// this (native) CPU build. The arenas are left stepped.
pub fn write_case(path: &Path, data: &ArenaData, arenas: &mut [Arena], ticks: u32) -> std::io::Result<()> {
    let num_cars = arenas[0].num_cars;
    let per = state_words(num_cars);
    let mut b = Vec::new();
    put_u32s(&mut b, &[0x434d_4c52, 1, num_cars, arenas.len() as u32, ticks, per]);
    for a in arenas.iter() {
        put_u32s(&mut b, a.words());
    }
    for a in arenas.iter_mut() {
        a.step(data, ticks);
        put_u32s(&mut b, a.words());
    }
    std::fs::write(path, b)
}

/// The synthetic parity set: `n` arenas, ball positions and velocities spread over the field, per-arena controls held
/// for the whole run (the GPU parity test and the MSan cases).
pub fn parity_arenas(num_cars: u32, n: usize) -> Vec<Arena> {
    (0..n)
        .map(|i| {
            let mut a = Arena::new(num_cars);
            let mut b = a.ball();
            b.body.pos = [(i as f32 * 37.0) % 3000.0 - 1500.0, (i as f32 * 53.0) % 4000.0 - 2000.0, 93.15 + (i % 7) as f32 * 120.0];
            b.body.vel = [(i % 5) as f32 * 300.0 - 600.0, (i % 3) as f32 * 400.0 - 400.0, 0.0];
            a.set_ball(&b);
            for c in 0..num_cars {
                let k = (i as u32 * 7 + c * 13) % 17;
                a.set_controls(c, &Controls {
                    throttle: if k.is_multiple_of(3) { -1.0 } else { 1.0 },
                    steer: (k as f32 / 8.0) - 1.0,
                    pitch: ((k % 5) as f32 / 2.0) - 1.0,
                    yaw: 0.0,
                    roll: 0.0,
                    jump: k.is_multiple_of(4) as u32,
                    boost: k.is_multiple_of(6) as u32,
                    handbrake: k.is_multiple_of(5) as u32,
                });
            }
            a
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uploaded_basis_survives_words_and_sleep() {
        let Ok(data) = ArenaData::soccar_from_env() else { return };
        let basis = [0.6, 0.8, 0.0, -0.8, 0.6, 0.0, 0.0, 0.0, 1.0];
        let mut arena = Arena::new(2);
        let mut ball = arena.ball();
        ball.body.rot = basis;
        ball.body.pos = [0.0, 0.0, 92.75];
        ball.sleeping = 2;
        arena.set_ball(&ball);
        let mut car = arena.car(0);
        car.body.rot = basis;
        arena.set_car(0, &car);
        let mut restored = Arena::from_words(2, arena.words());
        assert_eq!(restored.ball().body.rot.map(f32::to_bits), basis.map(f32::to_bits));
        assert_eq!(restored.car(0).body.rot.map(f32::to_bits), basis.map(f32::to_bits));
        restored.step(&data, 3);
        assert_eq!(restored.ball().body.rot.map(f32::to_bits), basis.map(f32::to_bits));
        restored.kickoff(1);
        assert_ne!(restored.ball().body.rot.map(f32::to_bits), basis.map(f32::to_bits));
    }

    #[test]
    fn math_matches_libm() {
        // the deterministic sin/cos/atan2 used by the physics, against the platform libm
        unsafe extern "C" {
            fn rlmax_test_sincos(x: f32, s: *mut f32, c: *mut f32);
            fn rlmax_test_atan2(y: f32, x: f32) -> f32;
        }
        let mut worst = (0.0f32, 0.0f32);
        for i in -20000..=20000 {
            let x = i as f32 * 0.0005;
            let (mut s, mut c) = (0.0, 0.0);
            unsafe { rlmax_test_sincos(x, &mut s, &mut c) };
            worst.0 = worst.0.max((s - x.sin()).abs()).max((c - x.cos()).abs());
            let y = (i as f32 * 0.37).sin();
            let xx = (i as f32 * 0.21).cos();
            worst.1 = worst.1.max((unsafe { rlmax_test_atan2(y, xx) } - y.atan2(xx)).abs());
        }
        eprintln!("max abs error: sin/cos {:.3e}, atan2 {:.3e}", worst.0, worst.1);
        assert!(worst.0 < 3e-7 && worst.1 < 5e-7, "{worst:?}");
    }

    #[test]
    fn reset_wheels_respect_previous_contacts() {
        let Ok(data) = ArenaData::soccar_from_env() else { return };
        let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let mut a = Arena::new(2);
        let mut c = CarState::default();
        c.body.rot = identity;
        c.body.pos = [0.0, 0.0, 17.0];
        a.set_car(0, &c);
        a.refresh_sticky_gate(&data, 0);
        assert_ne!(a.car(0).flags & flags::STICKY_GATE, 0);
        a.refresh_sticky_gate_with_wheels(&data, 0, 0);
        let no_contact = a.car(0);
        assert_eq!(no_contact.flags & flags::STICKY_GATE, 0);
        assert_eq!(no_contact.wheel_normal_sum, [0.0; 3]);
        a.refresh_sticky_gate_with_wheels(&data, 0, 1);
        let one_contact = a.car(0);
        assert_eq!((one_contact.flags >> 20) & 15, 1);
        assert_eq!(one_contact.wheel_normal_sum, [0.0, 0.0, 1.0]);
        a.refresh_sticky_gate_with_wheels(&data, 0, 15);
        assert_ne!(a.car(0).flags & flags::STICKY_GATE, 0);
    }

    /// A demolished car comes back with its own hitbox, in every preset.
    #[test]
    fn respawn_keeps_the_body() {
        let Ok(arena) = ArenaData::soccar_from_env() else {
            eprintln!("no meshes; skipped");
            return;
        };
        for preset in 0..6u32 {
            let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
            let ball = BallState { body: BodyState { pos: [0.0, 0.0, 93.15], rot: identity, ..Default::default() }, ..Default::default() };
            let mut car = CarState::default();
            car.body.rot = identity;
            car.body.pos = [0.0, -1000.0, 36.0];
            car.preset = preset;
            car.flags = flags::IS_DEMOED;
            car.demo_respawn_timer = 0.05;
            let mut other = car;
            other.body.pos = [0.0, 1000.0, 36.0];
            other.team = 1;
            other.flags = 0;
            other.demo_respawn_timer = 0.0;
            let mut a = Arena::from_physical(&arena, &ball, &[car, other]);
            a.step(&arena, 60);
            let c = a.car(0);
            assert_eq!(c.flags & flags::IS_DEMOED, 0, "preset {preset}: still demolished");
            assert_eq!(c.preset, preset, "preset {preset}: changed by the respawn");
        }
    }


    /// A kickoff is a rest state: idle for 120 ticks, every car (both teams, all five spots, every preset) stays where it
    /// started to within `TOL_UU` / `TOL_VEL` / `TOL_ANG`, and the rest heights equal the recordings'.
    #[test]
    fn kickoff_is_at_rest() {
        let Ok(arena) = ArenaData::soccar_from_env() else {
            eprintln!("no meshes; skipped");
            return;
        };
        // measured on this tree, worst over 6 presets x 5 spots x 2 teams x 199 seeds: 6.5e-4 uu (position or rotation x 50),
        // 2.0e-3 uu/s, 1.7e-4 rad/s: the spots at other yaws turn the table's yaw-90 pose and round the quaternion at 1e-7,
        // and the settle is not bit-fixed there. The tolerances are about 3 times that and 20 times under the transient they
        // replace (0.04 uu, 2.6 uu/s, 0.5 deg for the Octane)
        const TOL_UU: f32 = 2e-3;
        const TOL_VEL: f32 = 5e-3;
        const TOL_ANG: f32 = 5e-4;
        // recorded at "go" (the last row before any car moves): nexto Octane, daizen Plank, veai Merc; the rest is path-dependent,
        // and these are what a car dropped from spawn settles to (consts.h kickoff_rest)
        let recorded = [(0u32, 17.0123f32), (2, 18.6523), (5, 18.7919)];
        let mut seen = [false; 5];
        let mut worst = (0f32, 0f32, 0f32);
        for seed in 1..200u32 {
            for preset in 0..6u32 {
                let mut a = Arena::new(2);
                a.kickoff(seed);
                for i in 0..2 {
                    let mut c = a.car(i);
                    c.preset = preset;
                    a.set_car(i, &c);
                }
                a.kickoff(seed);
                let before: Vec<CarState> = (0..2).map(|i| a.car(i)).collect();
                let spot = before[0].body.pos;
                let s = match (spot[0].round() as i32, spot[1].round() as i32) {
                    (-2048, -2560) => 0,
                    (2048, -2560) => 1,
                    (-256, -3840) => 2,
                    (256, -3840) => 3,
                    _ => 4,
                };
                seen[s] = true;
                a.step(&arena, 120);
                for (i, b) in before.iter().enumerate() {
                    let c = a.car(i as u32);
                    let d = |p: &[f32], q: &[f32]| p.iter().zip(q).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
                    let (dp, dv, da) = (d(&c.body.pos, &b.body.pos), d(&c.body.vel, &b.body.vel), d(&c.body.ang_vel, &b.body.ang_vel));
                    // the rotation matrix, in uu-scale terms: 1e-5 is 0.0005 uu over a 50 uu lever
                    let dr = d(&c.body.rot, &b.body.rot) * 50.0;
                    worst = (worst.0.max(dp.max(dr)), worst.1.max(dv), worst.2.max(da));
                    assert!(dp < TOL_UU && dr < TOL_UU, "preset {preset} seed {seed} car {i}: moved {dp} uu / rot {dr}");
                    assert!(dv < TOL_VEL && da < TOL_ANG, "preset {preset} seed {seed} car {i}: vel {dv} ang {da}");
                    assert_eq!(c.flags & !flags::PREV_JUMP, b.flags, "preset {preset} seed {seed} car {i}: flags changed");
                }
                if seed == 1 {
                    for (p, z) in recorded {
                        if p == preset {
                            eprintln!("preset {preset}: rest z {:.5} uu, recorded {z}", before[0].body.pos[2]);
                            assert!((before[0].body.pos[2] - z).abs() < 1e-4, "preset {preset}: rest z {} against the recorded {z}", before[0].body.pos[2]);
                        }
                    }
                }
            }
        }
        eprintln!("worst over presets, spots and teams: pos/rot {} uu, vel {} uu/s, ang {} rad/s", worst.0, worst.1, worst.2);
        assert!(seen.iter().all(|&x| x), "not every spot was drawn: {seen:?}");
    }

    #[test]
    fn psyclops_kickoff_stays_at_rest() {
        let Ok(arena) = ArenaData::soccar_from_env() else {
            eprintln!("no meshes; skipped");
            return;
        };
        for seed in 1..=10 {
            let mut a = Arena::new(2);
            for i in 0..2 {
                let mut c = a.car(i);
                c.preset = 7;
                a.set_car(i, &c);
            }
            a.kickoff(seed);
            let before = [a.car(0), a.car(1)];
            a.step(&arena, 120);
            for (i, b) in before.iter().enumerate() {
                let c = a.car(i as u32);
                let max_diff = |x: &[f32], y: &[f32]| x.iter().zip(y).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
                assert!(max_diff(&c.body.pos, &b.body.pos) < 2e-3, "seed {seed}, car {i}: position");
                assert!(max_diff(&c.body.vel, &b.body.vel) < 5e-3, "seed {seed}, car {i}: velocity");
                assert!(max_diff(&c.body.ang_vel, &b.body.ang_vel) < 5e-4, "seed {seed}, car {i}: angular velocity");
            }
        }
    }

    #[test]
    fn extended_car_counts_preserve_teams_and_step_contacts() {
        let Ok(arena) = ArenaData::soccar_from_env() else {
            eprintln!("no meshes; skipped");
            return;
        };
        for n in [3u32, 5, 8] {
            let mut a = Arena::new(n);
            for i in 0..n {
                let mut c = a.car(i);
                c.team = u32::from(i >= n.div_ceil(2));
                a.set_car(i, &c);
            }
            a.kickoff(1);
            for i in 0..n {
                assert_eq!(a.car(i).team, u32::from(i >= n.div_ceil(2)), "{n} cars, slot {i}");
            }
            for (i, x) in [(n - 2, 1000.0), (n - 1, 1030.0)] {
                let mut c = a.car(i);
                c.body.pos = [x, 1000.0, 17.0];
                c.body.vel = [0.0; 3];
                c.body.ang_vel = [0.0; 3];
                a.set_car(i, &c);
            }
            a.step(&arena, 4);
            let left = a.car(n - 2).body;
            let right = a.car(n - 1).body;
            assert!(left.pos.into_iter().chain(right.pos).chain(left.vel).chain(right.vel).all(f32::is_finite));
            assert!((right.pos[0] - left.pos[0]).abs() > 30.0, "{n} cars: overlap did not separate");
        }
    }

    #[test]
    fn ball_falls_and_bounces() {
        let Ok(arena) = ArenaData::soccar_from_env() else {
            eprintln!("no meshes; skipped");
            return;
        };
        let mut a = Arena::new(2);
        let mut b = a.ball();
        b.body.pos = [0.0, 0.0, 1000.0];
        a.set_ball(&b);
        let mut min_z = f32::MAX;
        let mut bounced = false;
        let mut last_vz = 0.0;
        for _ in 0..240 {
            a.step(&arena, 1);
            let b = a.ball();
            min_z = min_z.min(b.body.pos[2]);
            if last_vz < 0.0 && b.body.vel[2] > 0.0 {
                bounced = true;
            }
            last_vz = b.body.vel[2];
        }
        eprintln!("min z {min_z}, bounced {bounced}");
        assert!(bounced);
        assert!(min_z > 80.0 && min_z < 95.0, "ball penetrated or floated: {min_z}");
    }
}
