        if (awake) {
            const Body& rb = bodies[0];
            const float thr = k.ball_threshold;
            // a goal triangle's point on the world is kept as an offset from the goal body's origin
            auto add = [&](const Contact& f, V3 org, V3 on_ball) {
                if (!f.ok || f.d > thr) return;
                Manifold* m = wman_get(w, f.group);
                if (m) man_add_point(*m, MPoint{man_to_local(&rb, on_ball), f.pb - org, f.n, f.d}, thr);
            };
            V3 bc = rb.pos;
            for (int p = 0; p < NUM_PLANES + 2; p++) {  // the goal floors too: the game's planes 104 and 105
                Contact cp = sphere_plane(bc, rb.r, k.ball_radius, plane_xf(p), thr);
                cp.group = 0x100u + (uint32_t)p;
                add(cp, plane_xf(p).o, cp.pb + cp.n * cp.d);
            }
            // the convex-concave callback takes only triangles that touch the sphere's bounding box (TestTriangleAgainstAabb2)
            const V3 br = v3s(k.ball_radius + c::SPHERE_BOUND_PAD_BT);
            int cell = cell_of(ad, bc);
            for (uint32_t t = ad.cell_off[cell]; t < ad.cell_off[cell + 1]; t++) {
                const Tri& tri = ad.tris[ad.cell_idx[t]];
                if (quirk::TRI_AABB_GATE_BALL) {
                    // a goal triangle is tested in its body's frame, the game's local vertex y against the centre less the origin
                    if (tri.flags & TF_ORIGIN_GOAL) {
                        V3 lv[3];
                        goal_tri_local(tri, ad.goal_ly, lv);
                        const float cy = bc.y - tri_origin(tri).y;
                        if (!tri_aabb_overlap(lv[0], lv[1], lv[2], v3(bc.x - br.x, cy - br.y, bc.z - br.z), v3(bc.x + br.x, cy + br.y, bc.z + br.z))) continue;
                    } else if (!tri_aabb_overlap(tri.p0, tri.p1, tri.p2, bc - br, bc + br)) {
                        continue;
                    }
                }
                Contact cc = sphere_tri(tri, bc, k.ball_radius, thr, ad.goal_ly);
                cc.group = tri_mesh(tri);
                // the edge adjustment runs as the point is added (contact-added callback), before the manifold sees it
                // the point on the ball is kept as it was found: the adjustment turns the normal and moves only the point on the world
                const V3 on_ball = cc.pb + cc.n * cc.d;
                if (cc.ok) adjust_internal_edge(tri, &cc, ad.goal_ly);
                add(cc, tri_origin(tri), on_ball);
            }
            V3 wpa[MANIFOLD_CAPACITY], wpb[MANIFOLD_CAPACITY];
            for (int q = 0; q < BALL_WORLD_MANIFOLDS; q++)
                if (q < (int)w.n) {
                    const V3 org = mesh_origin(w.obj[q]);
                    // a plane's point comes back through the plane's transform, whose rounding is what leaves a rolling ball a distance
                    const bool pl = w.obj[q] >= 0x100u;
                    const PlaneXf px = plane_xf(pl ? (int)(w.obj[q] - 0x100u) : 0);
                    man_refresh(w.m[q], &rb, nullptr, thr, wpa, wpb, pl ? &px : nullptr, w.obj[q] - 4u < 4u ? &org : nullptr);
                }
            wman_release_empty(w);
        }
