"""Load a course map package (OBJ + MTL + course/manifest JSON) into a compact dict for the viewer.

Course frame (from the package README): metres, x east, y north, z up, origin at the enclosure floor
centre. The OBJ is drawn top-down and also sliced at the drone's height in the viewer, so it keeps
3-D vertices.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_MAP = HERE.parent / "maps" / "new_course_origin"

MARK_PREFIXES = ("Launch mark", "Landing pad", "Approach ring")
SKIP_GROUPS = ("Net roof",)               # covers the whole course top-down; hides everything below it


def group_kind(name: str) -> str:
    if name.startswith("Floor"):
        return "floor"
    if name.startswith(MARK_PREFIXES):
        return "mark"
    if name.startswith("Net"):
        return "net"
    return "obstacle"


def _read_mtl(path: Path) -> dict[str, list[float]]:
    colors, cur = {}, None
    if not path.exists():
        return colors
    for ln in path.read_text().splitlines():
        parts = ln.split()
        if not parts:
            continue
        if parts[0] == "newmtl":
            cur = parts[1]
        elif parts[0] == "Kd" and cur:
            colors[cur] = [float(v) for v in parts[1:4]]
    return colors


def load_obj(obj_path: Path) -> dict:
    obj_path = Path(obj_path)
    verts: list[float] = []
    groups: list[dict] = []
    mtl_colors: dict = {}
    cur = None
    for ln in obj_path.read_text().splitlines():
        if not ln or ln[0] == "#":
            continue
        parts = ln.split()
        tag = parts[0]
        if tag == "v":
            verts.extend(round(float(v), 4) for v in parts[1:4])
        elif tag == "mtllib":
            mtl_colors.update(_read_mtl(obj_path.parent / " ".join(parts[1:])))
        elif tag in ("g", "o"):
            cur = {"name": " ".join(parts[1:]), "material": None, "faces": []}
            groups.append(cur)
        elif tag == "usemtl" and cur is not None:
            cur["material"] = parts[1]
        elif tag == "f":
            if cur is None:
                cur = {"name": "default", "material": None, "faces": []}
                groups.append(cur)
            idx = []
            for tok in parts[1:]:
                i = int(tok.split("/")[0])
                idx.append(i - 1 if i > 0 else len(verts) // 3 + i)
            for k in range(1, len(idx) - 1):           # fan-triangulate polygons
                cur["faces"].extend((idx[0], idx[k], idx[k + 1]))
    out = []
    for g in groups:
        if not g["faces"] or g["name"].startswith(SKIP_GROUPS):
            continue
        zs = [verts[3 * i + 2] for i in g["faces"]]
        out.append({"name": g["name"], "kind": group_kind(g["name"]),
                    "color": mtl_colors.get(g["material"], [0.5, 0.5, 0.5]),
                    "zmin": min(zs), "zmax": max(zs), "faces": g["faces"]})
    return {"verts": verts, "groups": out}


def load_map(map_dir: str | Path = DEFAULT_MAP) -> dict:
    """Everything the viewer draws: mesh, launch start prior, semantic goals, caveats."""
    d = Path(map_dir)
    objs = sorted(d.glob("*.obj"))
    if not objs:
        raise FileNotFoundError(f"no .obj in map directory {d}")
    mesh = load_obj(objs[0])
    manifest = json.loads((d / "manifest.json").read_text()) if (d / "manifest.json").exists() else {}
    course = json.loads((d / "course_v3.json").read_text()) if (d / "course_v3.json").exists() else {}
    goals = [{"name": s["name"], "xy": s["xy"], "radius_m": s.get("radius_m")}
             for s in manifest.get("resolved_sequence", []) if "xy" in s]
    start = manifest.get("start") or course.get("start")
    vehicle = course.get("vehicle", {})
    return {
        "name": d.name,
        "obj": objs[0].name,
        "mesh": mesh,
        "start": start,
        "goals": goals,
        "vehicle_radius_m": vehicle.get("radius_m", 0.294),
        "cruise_z_m": course.get("cruise_z_m"),
        "limitations": manifest.get("limitations", []),
    }


def course_start(course_doc: dict) -> tuple[float, float, float] | None:
    """(x, y, yaw_rad) of a course_v3.json start block; spawn_xy wins over xy as loc_node does."""
    import math
    st = (course_doc or {}).get("start")
    if not st:
        return None
    xy = st.get("spawn_xy", st.get("xy"))
    if xy is None or "yaw_deg" not in st:
        return None
    return float(xy[0]), float(xy[1]), math.radians(float(st["yaw_deg"]))
