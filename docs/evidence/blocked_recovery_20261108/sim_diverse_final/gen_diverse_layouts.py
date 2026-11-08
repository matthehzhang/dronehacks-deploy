#!/usr/bin/env python3
"""Experimental full-inventory topology randomization, separate from the default catalog.

Reuses the existing generic chain sampler but requires all three maze dividers,
all four chicane panels, and all five boulders. Acceptance preserves the dense
catalog's clearance, window, barrier, pocket, inventory, and density criteria.
No global rotations/reflections are used to claim diversity.
Route length is recorded separately; these are adversarial bugfinding worlds,
not a matched-difficulty evaluation.
"""
import argparse, json, time, collections
from pathlib import Path
import numpy as np
from gen_layouts import *

def generate_full_chain(rng, tier):
    """One canonical layout: barrier along x at y=yb, start area north, landing area south. Or None.
    tier: close (official template, jittered) | moderate (generic chain, official flow) | wild (anything valid)."""
    if tier == "close":
        return generate_sparse_close_legacy(rng)
    mod = tier == "moderate"
    slot = int(rng.choice([0, 3])) if mod else int(rng.integers(0, 4))
    L = Layout()
    L.tier = tier
    yb = rng.uniform(-0.45, 0.85)
    panels = ["B1", "B2", "B3"]
    rng.shuffle(panels)
    order = panels[:slot] + ["TUBE"] + panels[slot:]
    x = -H
    for p in order:
        w = LEN[p]
        if p == "TUBE":
            L.tube = {"c": [x + w / 2, yb], "u": [0.0, 1.0]}
        else:
            L.walls.append({"id": p, "a": [x, yb], "b": [x + w, yb], "role": "barrier"})
        x += w
    xt = L.tube["c"][0]

    # ---- start area: three independently placed dividers; reject incomplete inventory later
    for _ in range(200):
        p = np.array([rng.uniform(-H + 0.75, H - 0.75), rng.uniform(yb + 0.75, H - 0.75)])
        if mod and (p[0] * L.tube["c"][0] > 0 or abs(p[0]) < 0.5):   # start at the far end from the tube
            continue
        if L.tube_rect_dist(p[None])[0] > 0.75 and np.linalg.norm(p - L.entrance) > 1.6:
            L.launch = p
            break
    else:
        return None
    n_div = 3
    for did in rng.choice(["M1", "M2", "M3"], size=n_div, replace=False):
        for _ in range(60):
            Ld = LEN[did]
            if rng.random() < 0.6:   # one end against the net or the barrier
                side = rng.choice(["N", "W", "E", "S"])
                if side == "N":
                    a, n = np.array([rng.uniform(-H + 0.3, H - 0.3), H - 0.055]), np.array([0, -1.0])
                elif side == "W":
                    a, n = np.array([-H + 0.055, rng.uniform(yb + 0.3, H - 0.3)]), np.array([1.0, 0])
                elif side == "E":
                    a, n = np.array([H - 0.055, rng.uniform(yb + 0.3, H - 0.3)]), np.array([-1.0, 0])
                else:
                    a, n = np.array([rng.uniform(-H + 0.3, H - 0.3), yb + 0.106]), np.array([0, 1.0])
                ang = math.atan2(n[1], n[0]) + math.radians(rng.choice([-45, -30, 0, 0, 30, 45]))
                b = a + Ld * unit(ang)
            else:
                c = np.array([rng.uniform(-H + 0.9, H - 0.9), rng.uniform(yb + 0.9, H - 0.9)])
                ang = math.radians(rng.choice([0, 45, 90, 135]))
                a, b = c - Ld / 2 * unit(ang), c + Ld / 2 * unit(ang)
            S = seg_samples(a, b)
            if (np.abs(S) > H - 0.05).any() or (S[:, 1] < yb + 0.1).any():
                continue
            if seg_dist(L.launch[None], a, b)[0] < 0.85 or L.tube_rect_dist(S).min() < 1.0:
                continue
            if any(seg_dist(S, w["a"], w["b"]).min() < 0.3 for w in L.walls if w["role"] == "loose"):
                continue
            L.walls.append({"id": str(did), "a": a.tolist(), "b": b.tolist(), "role": "loose"})
            break

    # ---- landing area: chicane chain from one boundary to another with one gap (Gate B) and the window wall
    off = 0.055
    yface = yb - WALL_HT - off
    bnd = [(np.array([-H + off, -H + off]), np.array([H - off, -H + off])),
           (np.array([-H + off, -H + off]), np.array([-H + off, yface])),
           (np.array([H - off, -H + off]), np.array([H - off, yface]))]
    if xt - TUBE_HW - 0.05 > -H + off + 0.3:
        bnd.append((np.array([-H + off, yface]), np.array([xt - TUBE_HW - 0.05, yface])))
    if xt + TUBE_HW + 0.05 < H - off - 0.3:
        bnd.append((np.array([xt + TUBE_HW + 0.05, yface]), np.array([H - off, yface])))
    lens = np.array([np.linalg.norm(b - a) for a, b in bnd])

    chain = None
    for _ in range(400):
        k = 3
        seq = ["C2a"] + list(rng.choice(["C2b", "A1", "C1"], size=k, replace=False))
        rng.shuffle(seq)
        gi = int(rng.integers(0, len(seq)))
        gap = rng.uniform(1.0, 1.35)
        elems = seq[:gi] + ["GAP"] + seq[gi:]
        bi = rng.choice(len(bnd), p=lens / lens.sum())
        s0, s1 = bnd[bi]
        start = s0 + rng.uniform(0.05, 0.95) * (s1 - s0)
        tdir = (s1 - s0) / np.linalg.norm(s1 - s0)
        inward = np.array([-tdir[1], tdir[0]])
        if inward @ (np.zeros(2) - start) < 0 and bi < 3:
            inward = -inward
        if bi >= 3:
            inward = np.array([0, -1.0])
        h = math.atan2(inward[1], inward[0]) + math.radians(rng.choice([-60, -45, -30, 0, 30, 45, 60]))
        cur, out, ok = start.copy(), [], True
        for j, e in enumerate(elems):
            if j > 0:
                h += math.radians(rng.choice([0, 0, 30, -30, 45, -45, 60, -60, 90, -90]))
            Le = gap if e == "GAP" else LEN[e]
            if j == len(elems) - 1:   # close onto a boundary
                cands = []
                for a_, b_ in bnd:
                    for q in circle_seg_hits(cur, Le, a_, b_):
                        hq = math.atan2(q[1] - cur[1], q[0] - cur[0])
                        turn = abs((hq - h + math.pi) % (2 * math.pi) - math.pi)
                        if turn <= math.radians(100) and np.linalg.norm(q - start) > 0.5:
                            cands.append((q, hq))
                if not cands:
                    ok = False
                    break
                q, h = cands[rng.integers(len(cands))]
                nxt = np.array(q)
            else:
                nxt = cur + Le * unit(h)
            out.append((e, cur.copy(), nxt.copy()))
            cur = nxt
        if not ok:
            continue
        pan = [(e, a, b) for e, a, b in out if e != "GAP"]
        S_all = [seg_samples(a, b) for _, a, b in pan]
        if any((np.abs(S) > H - 0.04).any() or (S[:, 1] > yb - 0.09).any() for S in S_all):
            continue
        if any(L.tube_rect_dist(S).min() < 0.2 for S in S_all):
            continue
        if any(np.linalg.norm(S - L.exit, axis=1).min() < 1.1 for S in S_all):
            continue
        bad = False
        for i in range(len(pan)):
            for j in range(i + 1, len(pan)):
                adj = np.allclose(pan[i][2], pan[j][1]) or np.allclose(pan[j][2], pan[i][1])
                if not adj and seg_dist(S_all[i], pan[j][1], pan[j][2]).min() < 0.35:
                    bad = True
        if bad:
            continue
        g = next((a, b) for e, a, b in out if e == "GAP")
        chain = (pan, g)
        break
    if chain is None:
        return None
    pan, g = chain
    for e, a, b in pan:
        L.walls.append({"id": e, "a": a.tolist(), "b": b.tolist(), "role": "chicane"})
    L.gate = (g[0], g[1])

    far = ((GX * xt < 0) & (np.abs(GX - xt) >= 2.8)) if mod else None
    if not place_pad(L, rng, far):
        return None
    if "C1" not in [e for e, _, _ in pan]:
        place_c1(L, rng, yb)
    if not place_boulders(L, rng, yb, south_only=mod):
        return None
    return L


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seed',type=int,default=918246100)
    ap.add_argument('--seconds',type=float,default=120)
    ap.add_argument('--attempts',type=int,help='Exact candidate count for reproducing a saved timed run')
    ap.add_argument('--min-route',type=float,default=10.)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    reference=json.loads((ROOT/'layouts/procgen/layouts.json').read_text())['reference_metrics']
    candidates=[]; counts=collections.Counter(); start=time.monotonic(); attempt=0
    while (attempt < args.attempts if args.attempts is not None else time.monotonic()-start < args.seconds):
        seed=args.seed+attempt;attempt+=1;rng=np.random.default_rng(seed)
        layout=generate_full_chain(rng,'wild')
        if layout is None:counts['generation']+=1;continue
        if len(layout.walls)!=10 or len(layout.boulders)!=5:counts['inventory']+=1;continue
        density=density_metrics(layout,reference)
        if density['max_open_radius_m']>reference['max_open_radius_m'] or density['high_clearance_area_m2']>1.1*reference['high_clearance_area_m2']:counts['density']+=1;continue
        ok,metrics=check(layout,rng,.36)
        if not ok or metrics['route_len_m']<args.min_route:counts['route']+=1;continue
        metrics.update(density)
        metrics['route_length_ratio_to_original']=metrics['route_len_m']/reference['route_len_m']
        entry=to_json(layout,f'X{len(candidates)+1:02}',seed,metrics,'none')
        entry.update(tier='wild',topology_family='random_full_chain',chain='Random boundary chain containing all four chicane panels and three independently anchored dividers')
        candidates.append(entry)
        print('accepted',seed,round(metrics['route_len_m'],2),flush=True)
    if len(candidates)<2:raise RuntimeError(f'Only {len(candidates)} accepted from {attempt}; {counts}')
    def xy(entry):
        ps={p['id']:p for p in entry['pieces']}
        return np.array([[ps[k]['x'],ps[k]['y']] for k in sorted(SLUG)])
    def distance(a,b):
        # Remove global D4 equivalence before measuring layout diversity.
        best=float('inf')
        for k in range(4):
            ang=k*np.pi/2;R=np.array([[np.cos(ang),-np.sin(ang)],[np.sin(ang),np.cos(ang)]])
            for mirror in (np.eye(2),np.diag([-1,1])):
                best=min(best,float(np.sqrt(np.mean(np.sum((a-b@(R@mirror).T)**2,axis=1)))))
        return best
    pair=max(((distance(xy(a),xy(b)),i,j) for i,a in enumerate(candidates) for j,b in enumerate(candidates) if i<j))
    selected=[candidates[pair[1]],candidates[pair[2]]]
    for i,e in enumerate(selected):e['name']=f'DIVERSE{i+1:02}'
    result={'generator':'tools/procgen/gen_diverse_layouts.py','base_seed':args.seed,'attempts':attempt,'rejections':dict(counts),'candidate_count':len(candidates),'selection':'maximum D4-aligned wall-centre RMS separation among seeded valid candidates; no policy outcomes used','wall_center_rms_separation_m':pair[0],'minimum_route_m':args.min_route,'reference_metrics':reference,'layouts':selected,'candidate_seeds':[e['seed'] for e in candidates]}
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+'\n')
    print('selected',[(e['name'],e['seed'],e['metrics']['route_len_m']) for e in selected],pair[0],flush=True)

if __name__=='__main__':main()
