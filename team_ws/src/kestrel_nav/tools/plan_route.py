#!/usr/bin/env python3
"""Plan the course from the course file + published map. Writes plans/ (never route.json).

  python3 tools/plan_route.py                      # both chicane options, PNG + JSON
  python3 tools/plan_route.py --start X Y YAW_DEG  # anchor at a measured start pose (e.g. the localizer's)
  python3 tools/plan_route.py --legs 1             # also write a refnav route with only the first N legs
Always writes plans/route_<chicane>.json: the whole course in refnav format (params from the course file).
No ROS, no simulator: pure geometry.
"""
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))   # run from the source tree without colcon
from kestrel_nav.coursenav.course import load_course, plan_grids, resolve  # noqa: E402
from kestrel_nav.coursenav.objmap import CourseMap                         # noqa: E402
from kestrel_nav.coursenav import planner                                  # noqa: E402


def draw(cmap, obst, blocked, plans, out_png, doc, px_per_m=110):
    """Top-down PNG with OpenCV (already installed; no plotting dependency)."""
    import cv2
    import numpy as np
    from kestrel_nav.coursenav.viz import TopView
    tv = TopView(obst.origin, obst.occ.shape, cmap.res, px_per_m)
    img = tv.base(obst.occ, blocked.occ)
    colors = {"window": (40, 40, 220), "gate_b": (200, 110, 30)}
    pad = doc["sequence"][-1]
    cv2.circle(img, tv.px(pad["xy"]), int(pad.get("radius_m", 0.76) * px_per_m), (120, 120, 120), 1, cv2.LINE_AA)
    y = 22
    for key, p in plans.items():
        col = colors.get(key, (0, 0, 0))
        xy = [w["xy"] for w in p["waypoints"]]
        cv2.polylines(img, [np.array([tv.px(q) for q in xy], np.int32)], False, col, 2, cv2.LINE_AA)
        for w in p["waypoints"]:
            cv2.circle(img, tv.px(w["xy"]), 3, col, -1, cv2.LINE_AA)
            if not w["name"].startswith("via"):
                tv.label(img, w["name"], w["xy"], col)
        tv.text(img, f"{key}: {p['length_m']:.1f} m, {len(xy) - 1} legs, min clear {p['min_clearance_m']:.2f} m",
                (10, y), col)
        y += 20
    cv2.drawMarker(img, tv.px(doc["start"]["xy"]), (0, 150, 0), cv2.MARKER_TRIANGLE_UP, 16, 2)
    tv.text(img, f"z={doc['cruise_z_m']} m band; dark=obstacles, grey=inflated+keep-outs; green=start prior",
            (10, img.shape[0] - 10), (60, 60, 60))
    cv2.imwrite(str(out_png), img)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", nargs=3, type=float, metavar=("X", "Y", "YAW_DEG"))
    ap.add_argument("--chicane", choices=["auto", "window", "gate_b", "both"], default="auto",
                    help="auto: plan every option, score them and choose (the competition start does this)")
    ap.add_argument("--legs", type=int, default=0, help="also write plans/route_first<N>.json (refnav format)")
    ap.add_argument("--out", default=str(ROOT / "plans"))
    a = ap.parse_args()
    doc = load_course()
    cmap = CourseMap.load(resolve(doc, "map"))
    obst, blocked, clearance = plan_grids(cmap, doc)
    sx, sy = (a.start[0], a.start[1]) if a.start else doc["start"].get("spawn_xy", doc["start"]["xy"])
    syaw = math.radians(a.start[2] if a.start else doc["start"]["yaw_deg"])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    plans, chosen = {}, None
    if a.chicane in ("auto", "both"):
        res = planner.plan_options(doc, obst, blocked, clearance, start_xy=(sx, sy), cmap=cmap)
        print(f"chicane choice: {res['chicane']}\n  {res['reason']}")
        plans = res["plans"] if a.chicane == "both" else {res["chicane"]: res["plan"]}
        chosen = res["chicane"]
    else:
        try:
            plans[a.chicane] = planner.plan(doc, obst, blocked, clearance, start_xy=(sx, sy), chicane=a.chicane, cmap=cmap)
            chosen = a.chicane
        except planner.PlanError as e:
            raise SystemExit(f"{a.chicane}: NO PLAN: {e}")
    for k, p in plans.items():
        p["start"] = {"xy": [sx, sy], "yaw_deg": math.degrees(syaw)}
        p["route_forward_left"] = planner.to_route(p["waypoints"], (sx, sy), syaw)
        p["legs"] = planner.to_legs(p["waypoints"])
        (out / f"plan_{k}.json").write_text(json.dumps(p, indent=1))
        print(f"{k}: {len(p['waypoints'])} waypoints, {p['length_m']:.2f} m, min clearance "
              f"{p['min_clearance_m']:.2f} m (need {p['required_clearance_m']:.2f}), estimated {p['time_estimate_s']:.0f} s"
              + (f", moved leg ends {p['moved_leg_ends']}" if p["moved_leg_ends"] else "")
              + (f", LEG-END VIOLATIONS {p['leg_end_violations']}" if p["leg_end_violations"] else ""))
        for w in p["waypoints"]:
            extra = ""
            if w.get("align"):
                extra = f"  align: {w['align']['width_m']:.2f} m wide at {w['align']['dist_m']:.2f} m" + \
                        ("" if w["align"]["visible"] else " (edges NOT in view)")
            if w.get("passage", {}).get("is_passage"):
                extra += f"  passage {w['passage']['length_m']:.2f} m"
            print(f"   {w['name']:14s} ({w['xy'][0]:6.2f},{w['xy'][1]:6.2f})  clear {w['clearance_m']:.2f}"
                  f"  facing {w.get('facing_m')}{extra}")
    rn = doc.get("refnav_params", {})
    z = doc["cruise_z_m"]
    from kestrel_nav.coursenav.course import Grid2D
    band_m = rn.get("band_m", 0.25)
    band = Grid2D(cmap.band(z - band_m, z + band_m), obst.origin, cmap.res)      # refnav's obstacle band
    for k, p in plans.items():
        # offline replay of refnav's blocking rule on the map (+ 3 cm for map-cell rounding): predicts false stops
        hits = planner.refnav_block_check(p["waypoints"], band, rn.get("corridor_half_m", 0.32) + 0.03,
                                          planner.effective_stop_m(rn), rn.get("leg_end_margin_m", 0.25))
        p["predicted_false_blocks"] = hits
        print(f"{k}: refnav rule replayed on the map -> {'OK, no predicted false blocks' if not hits else hits}")
    draw(cmap, obst, blocked, plans, out / "course_plan.png", doc)
    params = {k: v for k, v in doc.get("refnav_params", {}).items() if not k.startswith("_")}
    for k, p in plans.items():
        # the whole course in refnav format (ends with the landing on the pad)
        n = len(p["route_forward_left"])
        doc_r = {"route": p["route_forward_left"], "legs": p["legs"], "params": params,
                 "_note": f"all {n} legs of plan_{k} from start {p['start']}; ends over the pad, then refnav lands. "
                          f"Generated by tools/plan_route.py (a planner OUTPUT for review; fly_course.py plans at start)."}
        (out / f"route_{k}.json").write_text(json.dumps(doc_r, indent=1))
    if a.legs:
        p = plans[chosen]
        route, legs = p["route_forward_left"][:a.legs], p["legs"][:a.legs]
        doc_r = {"route": route, "legs": legs, "params": params,
                 "_note": f"first {a.legs} leg(s) of plan_{chosen} from start {p['start']}; course-frame targets "
                          f"{[w['xy'] for w in p['waypoints'][1:a.legs + 1]]}. Generated by tools/plan_route.py."}
        path = out / (f"route_{chosen}_full.json" if a.legs >= len(p["route_forward_left"]) else f"route_first{a.legs}.json")
        path.write_text(json.dumps(doc_r, indent=1))
        print(f"wrote {path}: {route}")
    print(f"wrote {out / 'course_plan.png'}")


if __name__ == "__main__":
    main()
