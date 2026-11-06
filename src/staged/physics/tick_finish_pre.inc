    RLMAX_PROF(6)
    // ---- collision detection at the current pose
    // the solver's points, occupied slots only, in slot order (the order the rows are built in): writing every slot
    // every tick was the largest share of the tick's local stores (P3 §3.7); add_point skips an empty slot, so the
    // rows are the same
    // The staged stages always hand over the world, ball, predictive and sweep results (stages.cu), so the
    // points, counts and flags are the arena's global records.
    ContactPoint* ball_world = work.split_ball_world;
    ContactPoint (*car_world)[CAR_WORLD_SLOTS] = work.world_stage_points;
    ContactPoint (*car_ball)[CAR_BALL_SLOTS] = work.ball_stage_points;
    int ball_world_n = 0;
    int* car_world_n = work.world_stage_counts;
    int* car_ball_n = work.ball_stage_counts;
    bool* car_ball_touch = work.ball_stage_touch;
    ContactPoint (*car_car)[CAR_CAR_SLOTS] = work.split_car_car;
    bool& any_car_car = work.split_any_car_car;
    any_car_car = false;  // callbacks run only when a pair manifold holds points
    bool* car_world_hit = work.split_car_world_hit;
    V3* car_world_normal = work.split_car_world_normal;
    RLMAX_PROF(7)
    RLMAX_PROF(8)
    {
        // ball vs static (special points): one manifold per (ball, mesh component or plane) pair (manifold.h). The
        // sphere's pairs are plain shapes: this tick's points added, then refreshed. A sleeping ball and the static
        // world are both inactive: no pair runs and the manifolds stay as they are, unless a car touches the ball:
        // it wakes after applyGravity, joins the car's island and is solved this tick with its world contacts too
        bool car_touch = false;
        for (int i = 0; i < NC; i++) car_touch = car_touch || car_ball_n[i] > 0;
        // a car's touch reaches the world only if the ball's manifolds hold something; a wheel's always does
        // Vehicle work can wake WANTS_DEACTIVATION but never changes the SLEEPING classification here.
        const bool early = bs.sleeping != BALL_SLEEPING;
        const bool awake = early || wheels_woke_ball || (car_touch && s.ball_wman.n != 0);
        auto& w = s.ball_wman;
        if (!early) {
#include "ball_world_body.inc"
        }
#include "ball_world_points.inc"
    }
    // the broadphase sweep (sweep.h) from this tick's boxes: the solver's manifold order, and which car pairs hold a
    // manifold (the dispatcher's, made when the pair's compound child algorithm is, destroyed with it)
    PredictiveHit* predictive = work.predictive;
    uint32_t predictive_mask = work.predictive_mask;
    for (int b = 0; b < NB; b++) {
        const bool hit = (predictive_mask >> b & 1u) != 0;
        if (hit) {
            predictive_mask |= 1u << b;
            if (b && car_world_hit[b - 1]) {
                const auto& w = s.cars[b - 1].wman;
                V3 sum = v3s(0.0f);
                bool added = false;
                uint32_t last = 0;
                for (uint32_t n = 0; n < w.n; n++) {
                    int best = -1;
                    for (int q = 0; q < (int)w.n; q++) {
                        if (n && w.obj[q] <= last) continue;
                        if (best < 0 || w.obj[q] < w.obj[best]) best = q;
                    }
                    if (best < 0) break;
                    last = w.obj[best];
                    if (!added && last >= 0x100u) { sum += predictive[b].n; added = true; }
                    for (int pi = 0; pi < (int)w.m[best].n; pi++) sum += w.m[best].p[pi].n;
                }
                if (!added) sum += predictive[b].n;
                const float l2 = len2(sum);
                if (l2 >= 1e-8f) car_world_normal[b - 1] = sum * (1.0f / sqrtf(l2));
            }
        }
    }
    bool known_solver_points = predictive_mask != 0 || ball_world_n != 0;
    for (int i = 0; i < NC && !known_solver_points; i++)
        known_solver_points = car_world_n[i] != 0 || car_ball_n[i] != 0;
    const uint32_t sw_had = sweep_car_pairs<NC>(s.sweep);
    SweepKey<NC> sw_order[sweep_man_cap<NC>()];
    int sw_n;
    SweepIn<NC>& prepared = work.prepared_sweep;
    prepared.bodies = bodies;
    prepared.geo = geo;
    prepared.visit = s.visit;
    prepared.predictive = predictive_mask;
    prepared.ball_planes_held = ball_held_planes(s.ball_wman);
    sw_n = sweep_step<NC>(s.sweep, prepared, sw_order, known_solver_points);
    s.sweep.hdr |= 2u;
    const uint32_t sw_has = sweep_car_pairs<NC>(s.sweep);
    RLMAX_PROF(18)
    RLMAX_PROF(9)
    {
        int p = 0;
        for (int i = 0; i < NC; i++) {
            for (int j = i + 1; j < NC; j++, p++) {
                // the pair's persistent manifold: this tick's box-box points added, then refreshed (manifold.h)
                // A is the pair's higher broadphase handle
                const int ia = s.sweep.handle[1 + i] > s.sweep.handle[1 + j] ? i : j, ib = i + j - ia;
                const Body& ba = bodies[1 + ia];
                const Body& bb = bodies[1 + ib];
                Manifold& m = s.pair_man[p];
                V3 wpa[MANIFOLD_CAPACITY], wpb[MANIFOLD_CAPACITY];
                // a pair holds its points only as long as its manifold lives in the sweep: none made (or destroyed) this tick
                // means no collision, one made this tick starts empty
                if (!(ba.contact_response && bb.contact_response && (sw_has >> p & 1u))) {
                    if (m.n) m.n = 0;
                } else {
                    if (!(sw_had >> p & 1u)) m.n = 0;
                    const float thr = fmin_(geo[i].contact_threshold, geo[j].contact_threshold);
                    Contact fresh[CAR_CAR_SLOTS];
                    V3 fresh_a[CAR_CAR_SLOTS];
                    obb_obb<CAR_CAR_SLOTS>(car_obb(ba, geo[ia]), car_obb(bb, geo[ib]), thr, fresh, fresh_a);
                    // Match and select by the lower-handle car's local point.
                    man_collide(m, &ba, &bb, fresh, thr, quirk::COMPOUND_REFRESH_FIRST, wpa, wpb, true, fresh_a);
                }
                static_assert(CAR_CAR_SLOTS == MANIFOLD_CAPACITY, "one slot per manifold point");
                const int held = (int)m.n;
                if (held) {
                    snap[ia].q = ba.rot;
                    snap[ib].q = bb.rot;
                    any_car_car = true;
                }
                // only the held points are written (the readers stop at pair_man[p].n): writing the four slots of every pair
                // every tick was 15 pairs x 64 local stores at 3v3, for a pair that holds a point 0.5% of the ticks
                for (int q = 0; q < held; q++) {
                    ContactPoint& cp = car_car[p][q];
                    cp.ok = true;
                    cp.a = 1 + ia;
                    cp.b = 1 + ib;
                    cp.pb = wpb[q];
                    cp.n = m.p[q].n;
                    cp.d = m.p[q].d;
                    cp.pa = wpa[q];  // the manifold's point on A (Bullet m_positionWorldOnA), not B plus the normal depth
                    cp.friction = c::CAR_CAR_FRICTION;
                    cp.restitution = c::CAR_CAR_RESTITUTION;
                    cp.special = false;
                }
            }
        }
    }

    RLMAX_PROF(10)
    // ---- ball sleeping (island rule: no active car's AABB overlaps the ball's)
    if (bs.sleeping == BALL_WANTS_DEACTIVATION) {
        bool island_active = false;
        V3 bmin = bodies[0].pos - v3s(k.ball_radius + 0.08f), bmax = bodies[0].pos + v3s(k.ball_radius + 0.08f);
        for (int i = 0; i < NC; i++) {
            if (!bodies[1 + i].contact_response) continue;
            Obb ob = car_obb(bodies[1 + i], geo[i]);
            V3 fh = obb_full_half(ob);
            V3 ext = v3(dot(vabs(v3(ob.r.c0.x, ob.r.c1.x, ob.r.c2.x)), fh), dot(vabs(v3(ob.r.c0.y, ob.r.c1.y, ob.r.c2.y)), fh),
                        dot(vabs(v3(ob.r.c0.z, ob.r.c1.z, ob.r.c2.z)), fh));
            V3 cmin = ob.c - ext, cmax = ob.c + ext;
            bool overlap = !(cmin.x > bmax.x || cmax.x < bmin.x || cmin.y > bmax.y || cmax.y < bmin.y ||
                             cmin.z > bmax.z || cmax.z < bmin.z);
            island_active = island_active || overlap;
        }
        if (!island_active) {
            bs.sleeping = BALL_SLEEPING;
            bs.sleep_inv_i = bodies[0].inv_i;
            bodies[0].vel = bodies[0].ang = v3s(0.0f);
            bodies[0].active = false;
        }
    }

    RLMAX_PROF(11)
    // ---- solve
    const bool ball_was_sleeping = bs.sleeping == BALL_SLEEPING;
    work.split_ball_was_sleeping = ball_was_sleeping;
    const bool has_solver_points = known_solver_points || any_car_car;
    if (has_solver_points) {
        const RowSource<NC> src{bodies, predictive, ball_world, ball_world_n, car_world, car_ball, car_ball_n, car_car, any_car_car};
        if (s.seed_mask) solve_contacts_ext<NC, true>(work.solver, s, bodies, src, sw_order, sw_n, wheels_woke_ball, ball_was_sleeping);
        else solve_contacts_ext<NC, false>(work.solver, s, bodies, src, sw_order, sw_n, wheels_woke_ball, ball_was_sleeping);
    } else {
        for (int i = 0; i < NB; i++) {
            Body& b = bodies[i];
            if (!b.active && !(i == 0 && wheels_woke_ball)) continue;
            // Match the empty solver's additions, including their signed-zero behavior.
            b.vel = (b.vel + v3s(0.0f)) + b.acc_lin;
            b.ang = (b.ang + v3s(0.0f)) + b.acc_ang;
            if (i == 0 && ball_was_sleeping && (!is_zero(b.vel) || !is_zero(b.ang))) {
                bs.sleeping = BALL_ACTIVE;
                bs.deactivation_time = 0.0f;
            }
        }
    }
