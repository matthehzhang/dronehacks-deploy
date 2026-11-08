#!/usr/bin/env python3
"""Replay a skybeta rosbag on the course map: writes one self-contained HTML file and opens it.

    python replay/replay_bag.py <bag>                    # bag dir, .mcap file, or group154 run dir
    python replay/replay_bag.py <bag> --open             # also open it in the browser
    python replay/replay_bag.py <bag> --start -2.515 0.914 90 --landed-z 0.10
    python replay/replay_bag.py <bag> --map replay/maps/<other_map> --csv

Pure standard library (Python >= 3.8; zstd-compressed bags need Python 3.14 or `pip install zstandard`).
The HTML embeds the map and the flight, so it can be copied or mailed and opened offline.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import webbrowser
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from skyreplay.course_map import DEFAULT_MAP, course_start, load_map  # noqa: E402
from skyreplay.extract import divergence_summary, extract  # noqa: E402


def resolve_bag(path: Path) -> tuple[Path, Path | None]:
    """(bag path, run dir). A group154 run dir holds bag/ + course_v3.json + drone.env."""
    if path.is_dir() and (path / "bag").is_dir():
        return path / "bag", path
    if path.is_dir():
        return path, path.parent if (path.parent / "course_v3.json").exists() else None
    if path.suffix == ".mcap":
        parent = path.parent
        if parent.name == "bag" and (parent.parent / "course_v3.json").exists():
            return path, parent.parent
        return path, None
    raise SystemExit(f"{path}: not a bag directory, .mcap file or group154 run directory")


def default_output(bag: Path, run_dir: Path | None) -> Path:
    if run_dir is not None:
        return run_dir / "replay.html"
    if bag.is_dir():
        return bag / "replay.html"
    return bag.with_name(bag.stem + "_replay.html")


def write_csv(result: dict, path: Path):
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["estimate", "t_s", "x_m", "y_m", "z_m", "yaw_deg"])
        for k, tr in result["traces"].items():
            for t, x, y, z, yaw in zip(tr["t"], tr["x"], tr["y"], tr["z"], tr["yaw"]):
                w.writerow([k, t, x, y, "" if z is None else z, "" if yaw is None else round(math.degrees(yaw), 2)])


def render(map_doc: dict, result: dict) -> str:
    template = (HERE / "skyreplay" / "viewer.html").read_text(encoding="utf-8")

    def clean(o):                       # JSON has no NaN/Infinity: a stray one would break the page
        if isinstance(o, float):
            return o if math.isfinite(o) else None
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        return o

    def js(obj):
        return json.dumps(clean(obj), separators=(",", ":"), allow_nan=False).replace("</", "<\\/")

    data = f"const MAP={js(map_doc)};\nconst RUN={js(result)};"
    title = Path(result["bag"]).name
    return template.replace("/*__DATA__*/", data).replace("__TITLE__", title.replace("<", ""))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag", type=Path, help="rosbag2 directory, .mcap file, or group154 run directory")
    ap.add_argument("--map", type=Path, default=DEFAULT_MAP, help=f"map package directory (default {DEFAULT_MAP.name})")
    ap.add_argument("--start", nargs=3, type=float, metavar=("X", "Y", "YAW_DEG"),
                    help="launch pose in the course frame; overrides the fixture recorded in the bag")
    ap.add_argument("--landed-z", type=float, default=0.10,
                    help="body origin above the floor when landed, m (refnav_skybeta default 0.10)")
    ap.add_argument("--max-hz", type=float, default=30.0, help="downsample high-rate estimates to this rate")
    ap.add_argument("-o", "--output", type=Path, help="HTML file to write (default: next to the bag)")
    ap.add_argument("--csv", action="store_true", help="also write the course-frame traces as CSV")
    ap.add_argument("--open", action="store_true", help="open the replay in the default browser")
    a = ap.parse_args(argv)

    bag, run_dir = resolve_bag(a.bag.resolve())
    map_doc = load_map(a.map)

    fallback, fallback_src = None, None
    for cand in ([run_dir / "course_v3.json"] if run_dir else []) + [Path(a.map) / "course_v3.json"]:
        if cand.exists():
            fallback = course_start(json.loads(cand.read_text()))
            if fallback:
                fallback_src = f"start block of {cand}"
                break
    fixture = None
    if a.start:
        fixture = (a.start[0], a.start[1], math.radians(a.start[2]))

    t = time.time()
    result = extract(bag, fixture=fixture, fixture_source="--start" if fixture else None,
                     fallback_fixture=fallback, fallback_source=fallback_src,
                     landed_z=a.landed_z, max_hz=a.max_hz,
                     progress=lambda f: print(f"  read {f.name}", file=sys.stderr))
    result["generated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    result["run_dir"] = str(run_dir) if run_dir else None

    out = a.output or default_output(bag, run_dir)
    out.write_text(render(map_doc, result), encoding="utf-8")

    print(f"bag       {bag}  ({result['duration_s']:.1f} s, {sum(result['topics'].values())} messages, "
          f"{time.time() - t:.1f} s to read)")
    if result["truncated_files"]:
        print(f"note      {', '.join(result['truncated_files'])} ends mid-record (recorder did not finalise); "
              f"shown up to the cut")
    fx = result["fixture"]
    print("fixture   " + (f"({fx['x']:.3f}, {fx['y']:.3f}) m, {fx['yaw_deg']:.1f} deg  <- {fx['source']}"
                          if fx else "NONE: VIO/PX4 cannot be placed on the map; pass --start X Y YAW_DEG"))
    print("estimates")
    print("\n".join(divergence_summary(result)) or "  none found")
    if result["occupancy"]:
        print(f"live map  {len(result['occupancy'])} /coursenav/occupancy frames")
    print(f"events    {len(result['events'])}")
    if a.csv:
        cpath = out.with_suffix(".csv")
        write_csv(result, cpath)
        print(f"csv       {cpath}")
    print(f"replay    {out}")
    if a.open:
        webbrowser.open(out.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
