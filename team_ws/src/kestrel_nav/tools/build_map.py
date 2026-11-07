#!/usr/bin/env python3
"""Build maps/course_v3.npz from the published course OBJ (no simulator, no ROS)."""
import argparse
import hashlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # run from the source tree without colcon
from kestrel_nav.coursenav import objmap                      # noqa: E402
from kestrel_nav.coursenav.course import load_course, resolve  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--course", default=None)
    ap.add_argument("--res", type=float, default=0.05)
    a = ap.parse_args()
    doc = load_course(a.course) if a.course else load_course()
    obj, out = resolve(doc, "obj"), resolve(doc, "map")
    t = time.time()
    groups = objmap.load_obj(obj)
    cmap = objmap.build(groups, res=a.res)
    cmap.meta["obj_sha256"] = hashlib.sha256(obj.read_bytes()).hexdigest()
    out.parent.mkdir(parents=True, exist_ok=True)
    cmap.save(out)
    print(f"{len(groups)} groups -> {cmap.shape} voxels @ {a.res} m, {int(cmap.occupied.sum())} occupied, "
          f"{time.time() - t:.1f} s -> {out}")


if __name__ == "__main__":
    main()
