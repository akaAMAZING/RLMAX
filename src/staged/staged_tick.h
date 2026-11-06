#pragma once
#include "physics/tick.h"
namespace rlmax {
// The world part of one wheel_rays4 result, used only after the staged
// broadphase gate has excluded every dynamic body for all four rays. The
// triangle order, tie rank, leaf check and plane order are the same as above.
// Each wheel has its own ray.
RLMAX_HD WheelHit staged_world_wheel_ray(const ArenaData& ad, V3 from, V3 to, const CarGeo& geo,
                                          int cell, V3 rmn, V3 rmx) {
    const V3 d = to - from;
    const float l2 = len2(d);
    const float min_frac2 = quirk::WHEEL_RAY_MIN_HIT_DIST ? wheel_ray_min_dist_sq(geo) / l2 : 0.0f;
    float best = 1.0f;
    V3 best_n = v3s(0);
    int best_body = -2;
    uint32_t best_rank = 0;
    for (uint32_t k = ad.cell_off[cell]; k < ad.cell_off[cell + 1]; ++k) {
        Tri tri = ad.tris[ad.cell_idx[k]];
        V3 org = v3s(0.0f);
        if (tri.flags & TF_ORIGIN_GOAL) {
            V3 lv[3];
            goal_tri_local(tri, ad.goal_ly, lv);
            org = tri_origin(tri);
            tri.p0 = lv[0]; tri.p1 = lv[1]; tri.p2 = lv[2];
        }
        const V3 nraw = tri_raw_normal(tri);
        RayHit h{1.0f, v3s(0), false};
        ray_tri(from - org, to - org, tri, nraw, &h);
        if (!h.ok || !(h.frac * h.frac >= min_frac2)) continue;
        const uint32_t rank = tri_ray_rank(tri);
        if (!(h.frac < best || (h.frac == best && rank < best_rank))) continue;
        if (!ray_leaf_box(from - org, to - org, ad.tri_box + 6u * ad.cell_idx[k])) continue;
        best = h.frac; best_n = h.n; best_body = -1; best_rank = rank;
    }
    auto take = [&](RayHit& h) {
        if (h.ok && h.frac * h.frac >= min_frac2 && h.frac < best) {
            best = h.frac; best_n = h.n; best_body = -1;
        }
    };
    const float arena_x = c::ARENA_X_UU * c::UU_TO_BT;
    const float arena_z = c::ARENA_Z_UU * c::UU_TO_BT;
    const float edge_x = arena_x - arena_x * (1.0f / 64.0f);
    const float edge_z = arena_z - arena_z * (1.0f / 64.0f);
    const bool floor_only = rmn.x > -edge_x && rmx.x < edge_x && rmn.y > -2.0f * arena_x &&
                            rmx.y < 2.0f * arena_x && rmn.z > -arena_x && rmx.z < edge_z;
    if (floor_only) {
        RayHit h{1.0f, v3s(0), false};
        ray_plane_game(from, to, v3s(0.0f), v3(0.0f, 0.0f, 1.0f), &h);
        take(h);
    } else {
        for (int p = 0; p < NUM_PLANES; ++p) {
            V3 po, pn;
            ray_plane_at(p, &po, &pn);
            RayHit h{1.0f, v3s(0), false};
            ray_plane_game(from, to, po, pn, &h);
            take(h);
        }
    }
    WheelHit out{false, v3s(0), v3s(0), -2};
    if (best_body != -2) {
        out.ok = true;
        const float t_ = best, s_ = 1.0f - t_;
        out.point = v3(from.x * s_ + to.x * t_, from.y * s_ + to.y * t_, from.z * s_ + to.z * t_);
        out.normal = best_n;
        out.body = best_body;
    }
    return out;
}

// The staged path has excluded every dynamic body. Retain only values read
// after the per-wheel contact preparation; the hit's body and hard point are local.
struct StagedWheel {
    V3 axle_dir, contact_point, contact_normal, vel_at_contact;
    float susp_len, clipped_inv, susp_rel_vel, extra_pushback, lat_friction, long_friction;
    bool hit;
};

// Independent staged wheels touch only the world. These calculations match the
// corresponding vehicle_update phases; each delta reads the same phase input.
// Wheel ray, suspension and friction phases.
RLMAX_HD void staged_prepare_world_wheel(const Body& rb, const CarGeo& g, const CarTickOut& out,
                                         float handbrake_val, float real_throttle, int i,
                                         V3 hard_point, const WheelHit& hit, StagedWheel& w) {
    w.hit = hit.ok;
    if (!w.hit) { w.extra_pushback = 0.0f; return; }
    w.contact_point = hit.point;
    w.contact_normal = hit.normal;
    const V3 axle = steering_axle(rb.r, i < 2 ? out.steer : 0.0f);
    w.axle_dir = axle;
    const V3 up = rb.r.c2;
    const float trace = dot(hard_point - w.contact_point, up);
    const float max_len = g.rest_len[i] + c::MAX_SUSPENSION_TRAVEL_UU * c::UU_TO_BT;
    w.vel_at_contact = vel_at(rb, w.contact_point - rb.pos);
    const float proj = dot(w.contact_normal, w.vel_at_contact);
    const float denom = dot(w.contact_normal, up);
    if (denom > 0.1f) {
        const float inv = 1.0f / denom;
        w.susp_rel_vel = proj * inv;
        w.clipped_inv = inv;
    } else { w.susp_rel_vel = 0.0f; w.clipped_inv = 10.0f; }
    w.susp_len = fmin_(trace - g.radius[i], max_len);
    w.extra_pushback = 0.0f;
    const float thresh = (g.rest_len[i] + g.radius[i]) - 2.5f * c::UU_TO_BT;
    if (trace < thresh) {
        const float res = resolve_single_collision(rb, nullptr, w.contact_point, w.contact_normal, trace - thresh);
        w.extra_pushback = res / 4.0f;
    }
    const V3 long_dir = cross(axle, w.contact_normal);
    const V3 wheel_delta = hard_point - rb.pos;
    const V3 cv = (cross(rb.ang, wheel_delta) + rb.vel) * c::BT_TO_UU;
    const float base = absf(dot(cv, axle));
    const float fin = base > 5.0f ? base / (absf(dot(cv, long_dir)) + base) : 0.0f;
    float lat_f = curves::LAT_FRICTION.eval(fin);
    float long_f = 1.0f;
    if (handbrake_val != 0.0f) {
        lat_f *= 1.0f + (curves::HANDBRAKE_LAT_FRICTION_FACTOR.eval(fin) - 1.0f) * handbrake_val;
        long_f *= 1.0f + (curves::HANDBRAKE_LONG_FRICTION_FACTOR.eval(fin) - 1.0f) * handbrake_val;
    }
    if (real_throttle == 0.0f) {
        const float ns = curves::NON_STICKY_FRICTION_FACTOR.eval(w.contact_normal.z);
        lat_f *= ns; long_f *= ns;
    }
    w.lat_friction = lat_f;
    w.long_friction = long_f;
}

RLMAX_HD WheelDv staged_suspension_wheel(const Body& rb, const CarGeo& g, const StagedWheel& w, int i) {
    WheelDv out{v3s(0.0f), v3s(0.0f), false};
    if (!w.hit) return out;
    const float force = (g.rest_len[i] - w.susp_len) * c::SUSPENSION_STIFFNESS * w.clipped_inv;
    const float damp = w.susp_rel_vel < 0.0f ? c::WHEELS_DAMPING_COMPRESSION : c::WHEELS_DAMPING_RELAXATION;
    const float sf = (force - damp * w.susp_rel_vel) * g.force_scale[i];
    if (sf <= 0.0f) return out;
    const float bscale = sf * c::DT + w.extra_pushback;
    return wheel_impulse(rb, w.contact_normal * bscale, w.contact_point - rb.pos);
}

RLMAX_HD WheelDv staged_friction_wheel(const Body& rb, const StagedWheel& w, const CarTickOut& out,
                                        int num_contact) {
    WheelDv result{v3s(0.0f), v3s(0.0f), false};
    if (!w.hit) return result;
    const V3 axle = w.axle_dir;
    const V3 cn = cross(w.contact_normal, axle);
    const V3 fwd = cn * (1.0f / sqrtf(len2(cn)));
    const float friction_scale = rb.mass / 3.0f;
    float engine_force = out.engine_force;
    if (num_contact < 3) engine_force /= 4.0f;
    float rolling = 0.0f;
    if (engine_force == 0.0f) {
        if (out.brake_force != 0.0f) {
            const float rel = dot(w.vel_at_contact, fwd);
            rolling = clampf(-rel * c::ROLLING_FRICTION_SCALE, -out.brake_force, out.brake_force) * (c::DT * friction_scale);
        }
    } else rolling = -engine_force * c::DT;
    const float side_imp = resolve_bilateral_fake_ground(rb, w.contact_point, axle);
    const float roll = rolling * w.long_friction;
    const float side = (side_imp * (c::DT * friction_scale)) * w.lat_friction;
    const V3 impulse = v3(roll * fwd.x + side * axle.x, roll * fwd.y + side * axle.y, roll * fwd.z + side * axle.z);
    if (len2(impulse) == 0.0f) return result;
    const V3 off = w.contact_point - rb.pos;
    const V3 rel = off - rb.r.c2 * dot(rb.r.c2, off);
    return wheel_impulse(rb, impulse, rel);
}

// Independent per-body box preparation for the staged path; the ordered
// sweep_step remains unchanged; each body prepares its own proxy.
template <int NC>
RLMAX_HD void sweep_prepare_body(SweepState<NC>& st, SweepIn<NC>& swin, const Body* bodies,
                                 const V3* predict_lin, const V3* predict_ang, const CarGeo* geo,
                                 float ball_radius, const Q* reused_rotations, uint32_t reusable_rotations, int b) {
    const float cbt = c::CONTACT_BREAKING_THRESHOLD;
    const Body& rb = bodies[b];
    V3 nmn, nmx, pmn, pmx;
    if (b == 0) {
        const V3 e = v3s(ball_radius + c::SPHERE_BOUND_PAD_BT);
        nmn = rb.pos - e;
        nmx = rb.pos + e;
        const V3 pp = rb.pos + predict_lin[0] * c::DT;
        pmn = pp - e;
        pmx = pp + e;
        swin.ball_min = nmn;
        swin.ball_max = nmx;
        swin.ball_e = ball_radius + c::SPHERE_BOUND_PAD_BT;
    } else {
        const Obb now = car_obb(rb, geo[b - 1]);
        obb_aabb(now, &nmn, &nmx);
        swin.child_min[b - 1] = nmn;
        swin.child_max[b - 1] = nmx;
        // the box one tick on, in the game's arithmetic: the origin from the uu velocity, the basis by the tick's integration
        Body p;
        p.pos = rb.pos;
        if (reusable_rotations & (1u << b)) {
            p.r = bullet_set_rotation(reused_rotations[b - 1]);
        } else {
            p.r = rb.r;
            integrate_game(p, v3s(0.0f), predict_ang[b], c::DT);
        }
        const V3 pp = rb.pos + (predict_lin[b] * c::BT_TO_UU) * (c::UU_TO_BT * c::DT);
        const V3 off = geo[b - 1].hitbox_off;
        const V3 pc = v3(((pp.x + p.r.c0.x * off.x) + p.r.c1.x * off.y) + p.r.c2.x * off.z, ((pp.y + p.r.c0.y * off.x) + p.r.c1.y * off.y) + p.r.c2.y * off.z, ((pp.z + p.r.c0.z * off.x) + p.r.c1.z * off.y) + p.r.c2.z * off.z);
        obb_aabb(Obb{pc, p.r, geo[b - 1].hitbox_half, geo[b - 1].margin}, &pmn, &pmx);
    }
    const V3 cb = v3s(cbt);
    nmn = nmn - cb;
    nmx = nmx + cb;
    pmn = pmn - cb;
    pmx = pmx + cb;
    const V3 lo = v3(fmin_(nmn.x, pmn.x), fmin_(nmn.y, pmn.y), fmin_(nmn.z, pmn.z));
    const V3 hi = v3(fmax_(nmx.x, pmx.x), fmax_(nmx.y, pmx.y), fmax_(nmx.z, pmx.z));
    st.box[b][0] = lo.x; st.box[b][1] = lo.y; st.box[b][2] = lo.z;
    st.box[b][3] = hi.x; st.box[b][4] = hi.y; st.box[b][5] = hi.z;
    for (int a = 0; a < 3; a++) {
        const uint16_t qmin = sw_quant(vget(lo, a), 0);
        const uint16_t qmax = sw_quant(vget(hi, a), 1);
        swin.qedge[b][a] = (uint32_t)qmin | ((uint32_t)qmax << 16);
    }
    swin.active[b] = rb.active;
    swin.response[b] = rb.contact_response;
}

template <int NC>
RLMAX_COLD void staged_car_car_callbacks(ArenaState<NC>& s, const CarSnap* snap, const CarGeo* geo,
                                  const ContactPoint (*car_car)[CAR_CAR_SLOTS], V3* bumped_vel, EpaScratch& scratch) {
    // a car demolished this tick by its victim may still demolish it but not bump it; a car
    // demolished earlier, or by another car, takes no part
    int demoer_now[NC];
    for (int i = 0; i < NC; i++) demoer_now[i] = -1;
    int p = 0;
    for (int i = 0; i < NC; i++) {
        for (int j = i + 1; j < NC; j++, p++) {
            for (int q = 0; q < (int)s.pair_man[p].n; q++) {  // the pair's held points: the slots written below
                for (int sw = 0; sw < 2; sw++) {
                    int ai = sw ? j : i, vi = sw ? i : j;
                    CarState& att = s.cars[ai];
                    CarState& vic = s.cars[vi];
                    if (fl(vic, CF_IS_DEMOED)) continue;
                    const bool demo_only = fl(att, CF_IS_DEMOED);
                    if (demo_only && demoer_now[ai] != vi) continue;
                    V3 apos = att.pos * c::BT_TO_UU, vpos = vic.pos * c::BT_TO_UU;
                    V3 avel = contact_car_velocity_uu(att), vvel = contact_car_velocity_uu(vic);
                    V3 delta = vpos - apos;
                    if (dot(avel, delta) < 0.0f) continue;
                    V3 vel_dir = normalize_or_zero(avel);
                    V3 dir_to_v = normalize_or_zero(delta);
                    if (dot(avel, dir_to_v) <= dot(vvel, vel_dir)) continue;
                    // Tick-start quantization rebuilds the basis from this quaternion.
                    static_assert(quirk::QUANTIZE_PHYS, "callback basis follows the quantized quaternion");
                    const CarPose ap{att.pos, bullet_set_rotation(snap[ai].q), snap[ai].q, apos},
                                  vp{vic.pos, bullet_set_rotation(snap[vi].q), snap[vi].q, vpos};
                    const BodyPreset& ac = body_preset(att.body), &vc = body_preset(vic.body);
                    const V3 aoff = v3(ac.off_x, 0.0f, ac.off_z), voff = v3(vc.off_x, 0.0f, vc.off_z);
                    // demolition is checked before the bump cooldown, at the poses of first contact
                    bool is_demo = false;
                    if (fl(att, CF_IS_SUPERSONIC)) {
                        const float fs = dot(avel, ue3_rotator_axes(snap[ai].q).c0);
                        if (absf(fs) >= c::SUPERSONIC_MAINTAIN_MIN_SPEED_UU) {
                            CarPose at = ap, vt = vp;
                            time_of_impact_poses(at, geo[ai], avel, contact_car_angular(att), vt, geo[vi], vvel, contact_car_angular(vic), scratch);
                            is_demo = within_forward_cone(at, aoff, vt, voff, fs < 0.0f, c::DEMO_YAW_DEG, c::DEMO_PITCH_DEG);
                        }
                    }
                    if (is_demo) is_demo = att.team != vic.team;
                    if (is_demo) {
                        set_fl(vic, CF_IS_DEMOED, true);
                        vic.demo_respawn_timer = c::DEMO_RESPAWN_TIME;
                        demoer_now[vi] = ai;
                        continue;
                    }
                    if (demo_only) continue;
                    if (att.bump_cooldown > 0.0f && (!fl(att, CF_HAS_BUMP_VICTIM) || att.bump_victim == (uint32_t)vi)) continue;
                    const bool bumper = within_forward_cone(ap, aoff, vp, voff, false, c::BUMP_YAW_DEG, c::BUMP_PITCH_DEG);
                    if (bumper) {
                        // the impulse in momentum units along the attacker's velocity (its reciprocal speed), then
                        // through 1/50 and 1/mass; on the ground the victim's up is its rotator's
                        const float speed = sqrtf(len2(avel));
                        const V3 dir = avel * (1.0f / speed);
                        auto curve = [speed](float y0, float y1, float y2) {
                            if (speed < 1400.0f) return y0 + (y1 - y0) * (speed / 1400.0f);
                            if (speed < 2200.0f) return y1 + (y2 - y1) * ((speed - 1400.0f) / 800.0f);
                            return y2;
                        };
                        const V3 impulse = fl(vic, CF_ON_GROUND) ? dir * curve(150.0f, 200000.0f, 275000.0f) + ue3_up_axis(snap[vi].q) * curve(50.0f, 50000.0f, 75000.0f)
                                                                 : dir * curve(150.0f, 250000.0f, 350000.0f);
                        bumped_vel[vi] += (impulse * (1.0f / 50.0f)) * (1.0f / c::CAR_MASS_BT);
                    }
                    // the next bump needs the physics time strictly past the cooldown (half a tick longer); the
                    // cooldown starts without a bump too
                    att.bump_cooldown = c::BUMP_COOLDOWN_TIME + c::DT * 0.5f;
                    att.bump_victim = (uint32_t)vi;
                    set_fl(att, CF_HAS_BUMP_VICTIM, true);
                }
            }
        }
    }
}

template <int NC> struct StagedTick {
    GeoCache<NC> geo;
    uint32_t geo_valid;
    Body bodies[NC + 1];
    V3 pre_q_pos[NC + 1], ball_snap_vel_uu;
    CarTickOut outs[NC];
    bool pre_ticked[NC], car_independent[NC];
    V3 predict_lin[NC + 1], predict_ang[NC + 1];
    V3 ray_from[NC][4], ray_to[NC][4], ray_min[NC], ray_max[NC];
    int ray_cell[NC];
    bool wheels_woke_ball, independent;
    // Predictive results are written by independent body lanes before staged finish.
    PredictiveHit predictive[NC + 1];
    Q sweep_rotations[NC];
    uint32_t predictive_mask, reusable_rotations;
    CarSnap ball_stage_snap[NC];
    ContactPoint ball_stage_points[NC][CAR_BALL_SLOTS];
    int ball_stage_counts[NC];
    bool ball_stage_touch[NC];
    SweepIn<NC> prepared_sweep;
    V3 post_bumped_vel[NC];
    bool split_car_world_hit[NC], split_any_car_car, split_ball_was_sleeping;
    V3 split_car_world_normal[NC];
    ContactPoint world_stage_points[NC][CAR_WORLD_SLOTS];
    int world_stage_counts[NC];
    // Per-arena storage for the contact solve, so the kernel frame holds no solver rows.
    TickSolver<NC> solver;
    // EPA's working storage, in global memory so no kernel frame carries it: the world stage uses one per lane (four
    // lanes a car, concurrent); the stages that follow never overlap it and use the first.
    EpaScratch epa[NC * 4];
    ContactPoint split_ball_world[BALL_WORLD_SLOTS];
    ContactPoint split_car_car[NC * (NC - 1) / 2 > 0 ? NC * (NC - 1) / 2 : 1][CAR_CAR_SLOTS];
};

template <int NC>
RLMAX_HD void staged_begin(ArenaState<NC>& s, const ArenaData& ad, StagedTick<NC>& work) {
    static_assert(quirk::VEHICLE_IN_STEP, "parallel pre-ticks require vehicles inside the step");
    GeoCache<NC>& gc = work.geo;
    if (!work.geo_valid) { gc.reset(); work.geo_valid = 1; }
#define RLMAX_STAGED_REFERENCES
#include "physics/tick_setup.inc"
#undef RLMAX_STAGED_REFERENCES
    // Respawn consumes arena RNG in visit order, before independent car pre-ticks.
    // Each car can prepare its own state.
    update_visit_order<NC>(s);
    for (int vk = 0; vk < NC; ++vk) {
        const int i = visit_slot<NC>(s.visit, vk);
        CarState& cs = s.cars[i];
        work.pre_ticked[i] = false;
        if (fl(cs, CF_IS_DEMOED)) {
            cs.demo_respawn_timer = fmax_(cs.demo_respawn_timer - c::DT, 0.0f);
            if (cs.demo_respawn_timer != 0.0f) {
                bodies[1 + i].active = false;
                bodies[1 + i].contact_response = false;
                continue;
            }
            if (s.seed_mask) s.seed_mask &= ~((1u << (1 + i)) | (1u << (21 + i)));
            respawn_car<NC>(cs, s.rng);
#ifdef RLMAX_FULL_STATE
            fs_respawn_override<NC>(cs, i);
#endif
            Body& spawned = bodies[1 + i];
            spawned = make_body(cs.pos, cs.rot, cs.vel, cs.ang_vel, c::CAR_MASS_BT, geo[i].inv_inertia_local);
            spawned.inv_i = world_inv_inertia(respawn_inertia_basis(cs.team), spawned.inv_i_local);
            const V3 spawn_pos = spawned.pos;
#if !defined(RLMAX_FULL_STATE)
            pre_q_pos[1 + i] = spawn_pos;
#endif
            if (quirk::QUANTIZE_PHYS) quantize_body(spawned, spawned.vel_uu);
        }
        work.pre_ticked[i] = true;
    }
}

// The boxes and body response flags are unchanged by car pre-tick and gravity.
// Each car can compute its own four-ray hull and independence while its
// pre-tick lane is active; the next arena stage combines the results.
// Exclude interacting vehicles from the independent path.
template <int NC>
RLMAX_HD bool staged_vehicle_independent_car(int i, const ArenaState<NC>& s, const ArenaData& ad, StagedTick<NC>& work) {
    if (!quirk::VEHICLE_IN_STEP || !(s.sweep.hdr & 2u)) return false;
    const Body& body = work.bodies[1 + i];
    const CarGeo& geo = work.geo.g[i];
    V3 from[4], to[4];
    for (int r = 0; r < 4; ++r) {
        from[r] = body.pos + body.r * geo.wheel_conn[r];
        to[r] = from[r] - body.r.c2 * geo.ray_len[r];
        if (!(len2(to[r] - from[r]) < FLT_BIG)) return false;
        work.ray_from[i][r] = from[r];
        work.ray_to[i][r] = to[r];
    }
    V3 lo = from[0], hi = from[0];
    for (int r = 0; r < 4; ++r) {
        lo = v3(fmin_(lo.x, fmin_(from[r].x, to[r].x)), fmin_(lo.y, fmin_(from[r].y, to[r].y)), fmin_(lo.z, fmin_(from[r].z, to[r].z)));
        hi = v3(fmax_(hi.x, fmax_(from[r].x, to[r].x)), fmax_(hi.y, fmax_(from[r].y, to[r].y)), fmax_(hi.z, fmax_(from[r].z, to[r].z)));
    }
    work.ray_min[i] = lo;
    work.ray_max[i] = hi;
    work.ray_cell[i] = cell_of(ad, body.pos + body.r * geo.hitbox_off);
    for (int b = 0; b <= NC; ++b) {
        if (b == i + 1 || (b && !work.bodies[b].contact_response)) continue;
        const float* a = s.sweep.box[b];
        if (a[3] < lo.x || hi.x < a[0] || a[4] < lo.y || hi.y < a[1] || a[5] < lo.z || hi.z < a[2]) continue;
        // The admitted proxy can lag its body. Use the same current-body tests
        // as the ordered wheel path; a possible hit keeps the ordered fallback.
        for (int r = 0; r < 4; ++r) {
            if (b == 0) {
                RayHit hit{1.0f, v3s(0), false};
                ray_sphere(from[r], to[r], work.bodies[0].pos, tick_consts().ball_radius, &hit);
                if (hit.ok) return false;
            } else if (ray_car_box_overlap(from[r], to[r], work.bodies[b], work.geo.g[b - 1])) return false;
        }
    }
    return true;
}

template <int NC>
RLMAX_HD void staged_pre_car(int i, ArenaState<NC>& s, const ArenaData& ad, StagedTick<NC>& work) {
    if (!work.pre_ticked[i]) { work.car_independent[i] = true; return; }
    CarState& cs = s.cars[i];
    const TickConsts k = tick_consts();
    CarCtx<NC> ctx{&ad, work.geo.g, work.bodies, k.ball_radius, (s.sweep.hdr & 2u) ? s.sweep.box : nullptr};
    car_pre_tick<NC>(i, cs, ctx, work.outs[i]);
    // Gravity reads only this car's force and inertia; the ball remains in
    // the ordered arena stage. Keep Bullet's operation order for each body.
    Body& b = work.bodies[1 + i];
    if (b.active) {
        const V3 total = b.force + v3(0.0f, 0.0f, (c::GRAVITY_Z_UU * c::UU_TO_BT) * b.mass);
        b.acc_lin += (total * b.inv_mass) * c::DT;
        if (!is_zero(b.torque)) b.acc_ang += (b.inv_i * b.torque) * c::DT;
    }
    work.car_independent[i] = staged_vehicle_independent_car(i, s, ad, work);
}

template <int NC>
RLMAX_HD void staged_vehicle(ArenaState<NC>& s, const ArenaData& ad, StagedTick<NC>& work) {
    RLMAX_PROF_BEGIN
    constexpr int NB = NC + 1;
    BallState& bs = s.ball;
    Body (&bodies)[NB] = work.bodies;
    CarTickOut (&outs)[NC] = work.outs;
    bool (&pre_ticked)[NC] = work.pre_ticked;
    const CarGeo* geo = work.geo.g;
    const TickConsts k = tick_consts();
    CarCtx<NC> ctx{&ad, geo, bodies, k.ball_radius, (s.sweep.hdr & 2u) ? s.sweep.box : nullptr};
#define RLMAX_STAGED_REFERENCES
#include "physics/tick_vehicle.inc"
#undef RLMAX_STAGED_REFERENCES
}

// Each independent wheel ray and its contact preparation read only that car
// and the world, and return the lane's prepared wheel values.
template <int NC>
RLMAX_HD StagedWheel staged_vehicle_wheel(int i, int wheel, ArenaState<NC>& s, const ArenaData& ad, StagedTick<NC>& work) {
    const WheelHit hit = staged_world_wheel_ray(ad, work.ray_from[i][wheel], work.ray_to[i][wheel],
                                                work.geo.g[i], work.ray_cell[i], work.ray_min[i], work.ray_max[i]);
    const CarState& cs = s.cars[i];
    StagedWheel prepared{};
    staged_prepare_world_wheel(work.bodies[1 + i], work.geo.g[i], work.outs[i], cs.handbrake_val,
                               work.outs[i].real_throttle, wheel, work.ray_from[i][wheel], hit, prepared);
    return prepared;
}

template <int NC>
RLMAX_HD void staged_finish_pre_integrate(ArenaState<NC>& s, const ArenaData& ad, EpaScratch& scratch, StagedTick<NC>& work) {
    RLMAX_PROF_BEGIN
    constexpr int NB = NC + 1, NP = NC * (NC - 1) / 2;
    BallState& bs = s.ball;
    Body (&bodies)[NB] = work.bodies;
    V3 (&pre_q_pos)[NB] = work.pre_q_pos;
    CarTickOut (&outs)[NC] = work.outs;
    V3 (&predict_lin)[NB] = work.predict_lin;
    V3 (&predict_ang)[NB] = work.predict_ang;
    const bool wheels_woke_ball = work.wheels_woke_ball;
    const V3 ball_snap_vel_uu = work.ball_snap_vel_uu;
    const CarGeo* geo = work.geo.g;
    const TickConsts k = tick_consts();
    CarSnap* snap = work.ball_stage_snap;
#include "physics/tick_finish_pre.inc"
}
template <int NC>
RLMAX_HD void staged_finish_callbacks(ArenaState<NC>& s, const ArenaData& ad, EpaScratch& scratch, StagedTick<NC>& work) {
    RLMAX_PROF_BEGIN
    constexpr int NB = NC + 1, NP = NC * (NC - 1) / 2;
    BallState& bs = s.ball;
    Body (&bodies)[NB] = work.bodies;
    V3 (&pre_q_pos)[NB] = work.pre_q_pos;
    CarTickOut (&outs)[NC] = work.outs;
    V3 (&predict_lin)[NB] = work.predict_lin;
    V3 (&predict_ang)[NB] = work.predict_ang;
    const bool wheels_woke_ball = work.wheels_woke_ball;
    const V3 ball_snap_vel_uu = work.ball_snap_vel_uu;
    const CarGeo* geo = work.geo.g;
    const TickConsts k = tick_consts();
    CarSnap* snap = work.ball_stage_snap;
    bool* car_ball_touch = work.ball_stage_touch;
    bool* car_world_hit = work.split_car_world_hit;
    V3* car_world_normal = work.split_car_world_normal;
    ContactPoint (*car_car)[CAR_CAR_SLOTS] = nullptr;
    const bool any_car_car = work.split_any_car_car;
    const bool ball_was_sleeping = work.split_ball_was_sleeping;
#include "physics/tick_finish_callbacks.inc"
}
template <int NC>
RLMAX_HD void staged_car_post(int i, ArenaState<NC>& s, StagedTick<NC>& work) {
    Body* bodies = work.bodies;
    CarTickOut* outs = work.outs;
    V3* bumped_vel = work.post_bumped_vel;
#include "physics/tick_finish_car_post_body.inc"
}
template <int NC>
RLMAX_HD void staged_finish_final(ArenaState<NC>& s, const ArenaData& ad, EpaScratch& scratch, StagedTick<NC>& work) {
    RLMAX_PROF_BEGIN
    constexpr int NB = NC + 1, NP = NC * (NC - 1) / 2;
    BallState& bs = s.ball;
    Body (&bodies)[NB] = work.bodies;
    V3 (&pre_q_pos)[NB] = work.pre_q_pos;
    CarTickOut (&outs)[NC] = work.outs;
    V3 (&predict_lin)[NB] = work.predict_lin;
    V3 (&predict_ang)[NB] = work.predict_ang;
    const bool wheels_woke_ball = work.wheels_woke_ball;
    const V3 ball_snap_vel_uu = work.ball_snap_vel_uu;
    const CarGeo* geo = work.geo.g;
    const TickConsts k = tick_consts();
#include "physics/tick_finish_final.inc"
}

}  // namespace rlmax
