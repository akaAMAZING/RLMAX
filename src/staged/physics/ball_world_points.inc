        // a sleeping untouched ball's manifolds are kept but give the solver no points this tick
#pragma unroll
        for (int q = 0; q < BALL_WORLD_MANIFOLDS; q++) {
            if (!awake || q >= (int)w.n) break;
#pragma unroll
            for (int pi = 0; pi < MANIFOLD_CAPACITY; pi++) {
                if (pi >= (int)w.m[q].n) break;
                const V3 org = mesh_origin(w.obj[q]);
                const bool pl = w.obj[q] >= 0x100u;
                const PlaneXf px = plane_xf(pl ? (int)(w.obj[q] - 0x100u) : 0);
                const Contact bw = man_point(w.m[q], nullptr, pi, pl ? &px : nullptr, w.obj[q] - 4u < 4u ? &org : nullptr);
                ContactPoint& cp = ball_world[ball_world_n++];
                cp.ok = true;
                cp.a = 0;
                cp.b = -1;
                cp.pb = bw.pb;
                cp.n = bw.n;
                cp.d = bw.d;
                cp.pa = man_point_a(bodies[0], w.m[q], pi);  // the manifold's point on A (Bullet m_positionWorldOnA)
                cp.friction = fmin_(c::BALL_FRICTION, c::ARENA_FRICTION);
                cp.restitution = fmax_(c::BALL_RESTITUTION, c::ARENA_RESTITUTION);
                cp.special = true;
            }
        }
