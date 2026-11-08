"""Turn a skybeta rosbag into course-frame trajectories, events and live-map frames for the viewer.

Course frame: metres, x east, y north, z up, origin at the enclosure floor centre (the map's frame).

Every onboard position estimate in the bag is put into that frame, so they can be compared with each
other and with the walls:

  pf    Localizer (coursenav particle filter). /coursenav/pose, else /coursenav/status x/y/yaw_deg.
        Already in the course frame: the map-matched answer to "where is the drone".
  nav   Navigator belief: PX4's local position, placed with the launch fixture and shifted by the
        latest /coursenav/drift correction. This is the frame refnav/fly_course convert their
        course-frame paths into, so it is where the flight code thought the drone was.
  px4   PX4 EKF2 local position placed at the launch fixture, no map correction.
  vio   cuVSLAM odometry placed at the launch fixture (pure dead reckoning from the start pose).

Frame chain (checked against refnav/node.py and coursenav/loc_node.py):
  cuVSLAM odom (ENU, x = nose at power-on)  ->  course:  fixture (+) odom      (SE(2) compose)
  PX4 local NED                             ->  odom:    (x, y) = (y_ned, x_ned), yaw = pi/2 - heading
  the fixture is the declared start pose: /coursenav/drift "fixture", else the run's course_v3.json.

Heights: body origin above the floor = landed_z + (rise since the first sample). Bags from the
group154 recorder start on the ground, which this assumes.

No ground truth is recorded. The estimates can be compared with each other and with the walls; they
cannot be compared with where the drone really was unless an external reference (video, tape marks)
is added.
"""
from __future__ import annotations

import base64
import bisect
import json
import math
import re
from collections import Counter
from pathlib import Path

from .cdr import Decoder
from .mcap_reader import McapReader, bag_files

RE_LPOS = re.compile(r"^/fmu/out/vehicle_local_position(_v\d+)?$")
RE_STATUS = re.compile(r"^/fmu/out/vehicle_status(_v\d+)?$")
RE_LAND = re.compile(r"^/fmu/out/vehicle_land_detected(_v\d+)?$")
WANTED = {
    "/visual_slam/tracking/odometry", "/visual_slam/status", "/coursenav/status", "/coursenav/drift",
    "/coursenav/pose", "/coursenav/occupancy", "/refnav/status", "/skybeta/state", "/rosout",
}
VIO_JUMP_M = 0.5
ROSOUT_MIN_LEVEL = 30                      # WARN and above
MAX_ROSOUT_EVENTS = 300
MAX_OCC_FRAMES = 150

TRACE_STYLE = {
    "pf": ("Localizer (map match)", "#e8590c",
           "coursenav particle filter: depth scans matched against the course map. Course frame directly."),
    "nav": ("Navigator belief", "#1c7ed6",
            "PX4 local position + launch fixture + latest /coursenav/drift correction: where the flight "
            "code thought the drone was."),
    "px4": ("PX4 EKF2 (uncorrected)", "#2b8a3e",
            "PX4 local position placed at the launch fixture, no map correction."),
    "vio": ("cuVSLAM VIO (dead reckoning)", "#862e9c",
            "Visual odometry placed at the launch fixture, no map correction."),
}


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def compose(T, p):
    c, s = math.cos(T[2]), math.sin(T[2])
    return (T[0] + c * p[0] - s * p[1], T[1] + s * p[0] + c * p[1], wrap(T[2] + p[2]))


def yaw_of(q) -> float:
    return math.atan2(2 * (q["w"] * q["z"] + q["x"] * q["y"]), 1 - 2 * (q["y"] ** 2 + q["z"] ** 2))


def ned_to_odom(x_ned, y_ned, heading=None):
    yaw = None if heading is None or not math.isfinite(heading) else wrap(math.pi / 2 - heading)
    return y_ned, x_ned, yaw


class Trace:
    def __init__(self, key, max_hz):
        self.key = key
        self.min_dt = 1.0 / max_hz if max_hz else 0.0
        self.t, self.x, self.y, self.z, self.yaw = [], [], [], [], []
        self.extra: dict[str, list] = {}

    def add(self, t, x, y, z, yaw, force=False, **extra):
        if not (math.isfinite(x) and math.isfinite(y)):
            return
        if self.t and not force and t - self.t[-1] < self.min_dt:
            return
        self.t.append(round(t, 3))
        self.x.append(round(x, 4))
        self.y.append(round(y, 4))
        self.z.append(None if z is None or not math.isfinite(z) else round(z, 3))
        self.yaw.append(None if yaw is None or not math.isfinite(yaw) else round(yaw, 4))
        for k, v in extra.items():
            self.extra.setdefault(k, [None] * (len(self.t) - 1)).append(v)
        for k, lst in self.extra.items():
            if len(lst) < len(self.t):
                lst.append(None)

    def to_json(self):
        label, color, desc = TRACE_STYLE[self.key]
        return {"key": self.key, "label": label, "color": color, "desc": desc, "t": self.t, "x": self.x,
                "y": self.y, "z": self.z, "yaw": self.yaw, **({"extra": self.extra} if self.extra else {})}


class StepSeries:
    """Zero-order-hold lookup of a value by time."""

    def __init__(self):
        self.t, self.v = [], []

    def add(self, t, v):
        self.t.append(t)
        self.v.append(v)

    def at(self, t, max_age=None):
        i = bisect.bisect_right(self.t, t) - 1
        if i < 0:
            return None
        if max_age is not None and t - self.t[i] > max_age:
            return None
        return self.v[i]


def _json(msg):
    try:
        d = json.loads(msg.get("data", ""))
        return d if isinstance(d, dict) else None
    except (ValueError, TypeError):
        return None


def _finite(v):
    return isinstance(v, (int, float)) and math.isfinite(v)


def extract(bag: str | Path, fixture=None, fixture_source=None, fallback_fixture=None, fallback_source=None,
            landed_z=0.10, max_hz=30.0, progress=None) -> dict:
    """fixture: (x, y, yaw_rad) launch pose that overrides everything.
    Otherwise the bag's /coursenav/drift fixture, else fallback_fixture (e.g. the run's course_v3.json)."""
    files = bag_files(bag)
    decoders: dict[int, Decoder] = {}
    counts: Counter = Counter()
    truncated = []
    t0_ns = None

    odom, lpos, events = [], [], []
    pf_status, pf_pose, drift, refnav, skystate = [], [], [], [], []
    vo_changes, occ_msgs = [], []
    px4_status, px4_land = [], []
    rosout_n = 0

    for f in files:
        reader = McapReader(f)
        for m in reader.messages():
            topic = m.channel.topic
            counts[topic] += 1
            if t0_ns is None:
                t0_ns = m.log_time_ns
            if not (topic in WANTED or RE_LPOS.match(topic) or RE_STATUS.match(topic) or RE_LAND.match(topic)):
                continue
            if m.schema is None or m.schema.encoding != "ros2msg" or m.channel.message_encoding != "cdr":
                continue
            dec = decoders.get(m.channel.id)
            if dec is None:
                dec = decoders[m.channel.id] = Decoder(m.schema.name, m.schema.data.decode("utf-8", "replace"))
            try:
                d = dec(m.data)
            except Exception:                    # noqa: BLE001  one corrupt message must not kill the replay
                counts["!decode_errors"] += 1
                continue
            t = (m.log_time_ns - t0_ns) * 1e-9
            if topic == "/visual_slam/tracking/odometry":
                p, q = d["pose"]["pose"]["position"], d["pose"]["pose"]["orientation"]
                odom.append((t, p["x"], p["y"], p["z"], yaw_of(q)))
            elif RE_LPOS.match(topic):
                if d.get("xy_valid"):
                    lpos.append((t, d["x"], d["y"], d["z"] if d.get("z_valid") else math.nan, d.get("heading", math.nan),
                                 d.get("dist_bottom", math.nan)))
            elif topic == "/visual_slam/status":
                vo = d.get("vo_state")
                if not vo_changes or vo_changes[-1][1] != vo:
                    vo_changes.append((t, vo))
            elif topic == "/coursenav/pose":
                p, q = d["pose"]["pose"]["position"], d["pose"]["pose"]["orientation"]
                cov = d["pose"]["covariance"]
                pf_pose.append((t, p["x"], p["y"], yaw_of(q), math.sqrt(max(cov[0], cov[7], 0.0))))
            elif topic == "/coursenav/occupancy":
                occ_msgs.append((t, d))
            elif topic == "/rosout":
                if d.get("level", 0) >= ROSOUT_MIN_LEVEL and rosout_n < MAX_ROSOUT_EVENTS:
                    rosout_n += 1
                    lvl = {30: "WARN", 40: "ERROR", 50: "FATAL"}.get(d["level"], str(d["level"]))
                    events.append({"t": round(t, 3), "kind": "log", "text": f"[{lvl}] {d.get('name', '')}: {d.get('msg', '')}"[:400]})
            elif RE_STATUS.match(topic):
                px4_status.append((t, d.get("arming_state"), d.get("nav_state")))
            elif RE_LAND.match(topic):
                px4_land.append((t, d.get("landed")))
            else:
                j = _json(d)
                if j is None:
                    continue
                {"/coursenav/status": pf_status, "/coursenav/drift": drift, "/refnav/status": refnav,
                 "/skybeta/state": skystate}.get(topic, []).append((t, j))
        if reader.truncated:
            truncated.append(f.name)
        if progress:
            progress(f)

    if t0_ns is None:
        raise ValueError(f"{bag}: no messages")

    # ---------------------------------------------------------------- launch fixture
    fixture_drift = next(((tuple(j["fixture"])) for _, j in drift if len(j.get("fixture") or []) == 3), None)
    if fixture is None:
        if fixture_drift is not None:
            fixture, fixture_source = fixture_drift, "/coursenav/drift fixture (the localizer's declared start)"
        elif fallback_fixture is not None:
            fixture, fixture_source = tuple(fallback_fixture), fallback_source or "course file start"
        else:
            fixture_source = "none"
    nav_fixture = fixture_drift or fixture

    traces = {}
    # ---------------------------------------------------------------- VIO
    vio_z0 = odom[0][3] if odom else 0.0
    vio_z = StepSeries()
    if odom:
        tr = Trace("vio", max_hz)
        prev = None
        for t, x, y, z, yaw in odom:
            zc = landed_z + (z - vio_z0)
            vio_z.add(t, zc)
            if prev is not None and math.hypot(x - prev[0], y - prev[1]) > VIO_JUMP_M:
                events.append({"t": round(t, 3), "kind": "vio",
                               "text": f"VIO pose jumped {math.hypot(x - prev[0], y - prev[1]):.2f} m (odometry reset)"})
            prev = (x, y)
            if fixture is not None:
                cx, cy, cyaw = compose(fixture, (x, y, yaw))
                tr.add(t, cx, cy, zc, cyaw)
        if tr.t:
            traces["vio"] = tr
    for i, (t, vo) in enumerate(vo_changes):
        if i == 0 and vo == 1:
            continue
        txt = {0: "unknown", 1: "tracking", 2: "LOST"}.get(vo, str(vo))
        events.append({"t": round(t, 3), "kind": "vio", "text": f"cuVSLAM vo_state -> {vo} ({txt})"})

    # ---------------------------------------------------------------- drift correction (navigator)
    delta = StepSeries()
    last_resets = None
    for t, j in drift:
        dl = j.get("delta")
        if isinstance(dl, list) and len(dl) == 3 and all(_finite(v) for v in dl):
            delta.add(t, (dl[0], dl[1], dl[2], j.get("sigma_xy"), bool(j.get("uncorrected"))))
        if last_resets is not None and j.get("resets") != last_resets:
            events.append({"t": round(t, 3), "kind": "loc", "text": f"drift EKF reset (resets={j.get('resets')})"})
        last_resets = j.get("resets")
    first_corr = next((t for t, v in zip(delta.t, delta.v) if not v[4]), None)
    if first_corr is not None:
        events.append({"t": round(first_corr, 3), "kind": "loc", "text": "map correction active (drift EKF accepted the localizer)"})

    # ---------------------------------------------------------------- PX4
    if lpos:
        z0 = next((r[3] for r in lpos if math.isfinite(r[3])), 0.0)
        tr_px4 = Trace("px4", max_hz)
        tr_nav = Trace("nav", max_hz)
        for t, xn, yn, zn, hd, _db in lpos:
            ox, oy, oyaw = ned_to_odom(xn, yn, hd)
            zc = landed_z - (zn - z0) if math.isfinite(zn) else None
            yaw_in = oyaw if oyaw is not None else 0.0
            if fixture is not None:
                cx, cy, cyaw = compose(fixture, (ox, oy, yaw_in))
                tr_px4.add(t, cx, cy, zc, cyaw if oyaw is not None else None)
            dv = delta.at(t)
            if dv is not None and nav_fixture is not None:
                nx, ny, nyaw = compose(nav_fixture, (ox, oy, yaw_in))
                tr_nav.add(t, nx + dv[0], ny + dv[1], zc, wrap(nyaw + dv[2]) if oyaw is not None else None,
                           corrected=not dv[4])
        if tr_px4.t:
            traces["px4"] = tr_px4
        if tr_nav.t:
            traces["nav"] = tr_nav

    # ---------------------------------------------------------------- localizer
    tr_pf = Trace("pf", 0)
    if pf_pose:
        for t, x, y, yaw, sig in pf_pose:
            tr_pf.add(t, x, y, vio_z.at(t), yaw, sigma_xy=round(sig, 3))
    else:
        seen_t = None
        for t, j in pf_status:
            if not all(_finite(j.get(k)) for k in ("x", "y")) or j.get("t") == seen_t:
                continue
            seen_t = j.get("t")
            yaw = math.radians(j["yaw_deg"]) if _finite(j.get("yaw_deg")) else None
            tr_pf.add(t, j["x"], j["y"], vio_z.at(t), yaw, sigma_xy=j.get("sigma_xy"),
                      updates=j.get("updates"), beams=j.get("beams_depth"))
    if tr_pf.t:
        traces["pf"] = tr_pf

    # ---------------------------------------------------------------- planned route (refnav targets)
    def nominal_to_course(xn, yn, t):
        if nav_fixture is None:
            return None
        ox, oy, _ = ned_to_odom(xn, yn)
        cx, cy, _ = compose(nav_fixture, (ox, oy, 0.0))
        dv = delta.at(t)
        if dv is not None:
            cx, cy = cx + dv[0], cy + dv[1]
        return [round(cx, 4), round(cy, 4)]

    plans, last_targets = [], None
    setpoints = Trace("nav", 0)
    last_phase = None
    for t, j in refnav:
        tg = j.get("targets") or []
        if tg and tg != last_targets:
            pts = [nominal_to_course(p[0], p[1], t) for p in tg if len(p) >= 2]
            pts = [p for p in pts if p is not None]
            if pts:
                plans.append({"t": round(t, 3), "points": pts})
            last_targets = tg
        sp = j.get("setpoint")
        if isinstance(sp, list) and len(sp) >= 2 and _finite(sp[0]) and _finite(sp[1]):
            c = nominal_to_course(sp[0], sp[1], t)
            if c is not None:
                setpoints.add(t, c[0], c[1], None, None, force=True)
        ph = j.get("phase")
        if ph != last_phase:
            why = j.get("why") or ""
            events.append({"t": round(t, 3), "kind": "phase",
                           "text": f"refnav phase -> {ph}" + (f" ({why})" if why else "")
                                   + (f" leg {j.get('leg')}/{j.get('legs')}" if j.get("legs") else "")})
            last_phase = ph

    # ---------------------------------------------------------------- vehicle state events
    if skystate:
        prev = {}
        for t, j in skystate:
            for key, fmt in (("armed", lambda v: "ARMED" if v else "disarmed"),
                             ("landed", lambda v: "landed" if v else "airborne"),
                             ("nav_state_name", lambda v: f"PX4 mode {v}"),
                             ("px4_link", lambda v: "PX4 link up" if v else "PX4 link LOST")):
                v = j.get(key)
                if v is None or prev.get(key) == v:
                    continue
                if key in prev or v not in (False, "-1"):
                    events.append({"t": round(t, 3), "kind": "state", "text": fmt(v)})
                prev[key] = v
    else:
        prev_arm = prev_nav = prev_land = None
        for t, arm, nav in px4_status:
            if arm != prev_arm:
                events.append({"t": round(t, 3), "kind": "state",
                               "text": {2: "ARMED", 1: "disarmed"}.get(arm, f"arming_state {arm}")})
                prev_arm = arm
            if nav != prev_nav:
                events.append({"t": round(t, 3), "kind": "state", "text": f"PX4 nav_state {nav}"})
                prev_nav = nav
        for t, landed in px4_land:
            if landed != prev_land:
                events.append({"t": round(t, 3), "kind": "state", "text": "landed" if landed else "airborne"})
                prev_land = landed

    # ---------------------------------------------------------------- live occupancy frames
    occ = _occupancy_frames(occ_msgs)

    events = collapse_repeats(sorted(events, key=lambda e: e["t"]))
    duration = max([t for tr in traces.values() for t in tr.t[-1:]] + [e["t"] for e in events] + [0.0])
    return {
        "bag": str(Path(bag)),
        "files": [f.name for f in files],
        "truncated_files": truncated,
        "t0_unix": t0_ns * 1e-9,
        "duration_s": round(duration, 3),
        "fixture": None if fixture is None else {"x": fixture[0], "y": fixture[1],
                                                 "yaw_deg": round(math.degrees(fixture[2]), 3),
                                                 "source": fixture_source},
        "nav_fixture_differs": bool(fixture_drift is not None and fixture is not None
                                    and any(abs(a - b) > 1e-6 for a, b in zip(fixture_drift, fixture))),
        "landed_z": landed_z,
        "traces": {k: v.to_json() for k, v in traces.items()},
        "plans": plans,
        "setpoints": {"t": setpoints.t, "x": setpoints.x, "y": setpoints.y} if setpoints.t else None,
        "events": events,
        "occupancy": occ,
        "topics": dict(sorted(counts.items())),
    }


_DIGITS = re.compile(r"[-+]?\d+(\.\d+)?")


def collapse_repeats(events: list[dict], window_s: float = 3.0) -> list[dict]:
    """Fold bursts of the same log line (numbers ignored) into one event with a count."""
    out, last = [], {}
    for e in events:
        if e["kind"] != "log":
            out.append(e)
            continue
        key = _DIGITS.sub("#", e["text"])
        prev = last.get(key)
        if prev is not None and e["t"] - prev["t_last"] <= window_s:
            prev["count"] = prev.get("count", 1) + 1
            prev["t_last"] = e["t"]
            continue
        e = dict(e, t_last=e["t"])
        last[key] = e
        out.append(e)
    for e in out:
        t_last = e.pop("t_last", None)
        if e.get("count", 1) > 1:
            e["text"] += f"  (x{e['count']} until t={t_last:.1f} s)"
    return out


def _occupancy_frames(msgs):
    if not msgs:
        return None
    step = max(1, math.ceil(len(msgs) / MAX_OCC_FRAMES))
    keep = msgs[::step]
    if keep[-1] is not msgs[-1]:
        keep.append(msgs[-1])
    frames = []
    for t, d in keep:
        info = d["info"]
        o = info["origin"]
        data = bytes((v + 256) % 256 for v in d["data"])          # int8 -> byte (-1 unknown = 255)
        frames.append({"t": round(t, 3), "w": info["width"], "h": info["height"],
                       "known": sum(1 for v in data if v != 255), "occupied": sum(1 for v in data if 50 <= v <= 100),
                       "res": info["resolution"], "ox": o["position"]["x"], "oy": o["position"]["y"],
                       "oyaw": yaw_of(o["orientation"]), "frame": d["header"]["frame_id"],
                       "data": base64.b64encode(data).decode("ascii")})
    return frames


def divergence_summary(result: dict) -> list[str]:
    """Plain-text comparison of the estimates: how far apart they ended and how far apart they got."""
    tr = result["traces"]
    lines = []
    for k, v in tr.items():
        if v["t"]:
            lines.append(f"  {v['label']:<30} {len(v['t']):6d} samples, ends at "
                         f"({v['x'][-1]:+.2f}, {v['y'][-1]:+.2f}) m")
    keys = [k for k in ("pf", "nav", "px4", "vio") if k in tr]
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            gap = max_gap(tr[a], tr[b])
            if gap:
                lines.append(f"  max gap {tr[a]['label']} vs {tr[b]['label']}: {gap[0]:.2f} m at t={gap[1]:.1f} s")
    return lines


def interp_xy(tr: dict, t: float, max_span: float = 1.0):
    """Linear interpolation of a trace at t; None outside it or across a gap longer than max_span."""
    ts = tr["t"]
    j = bisect.bisect_right(ts, t)
    if j == 0 or j == len(ts):
        return (tr["x"][-1], tr["y"][-1]) if ts and j == len(ts) and t - ts[-1] < 1e-6 else None
    t0, t1 = ts[j - 1], ts[j]
    if t1 - t0 > max_span:
        return None
    f = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
    return (tr["x"][j - 1] + f * (tr["x"][j] - tr["x"][j - 1]), tr["y"][j - 1] + f * (tr["y"][j] - tr["y"][j - 1]))


def max_gap(A: dict, B: dict):
    """(metres, t) of the largest horizontal disagreement, at the sparser trace's samples."""
    if len(A["t"]) > len(B["t"]):
        A, B = B, A
    worst = None
    for t, x, y in zip(A["t"], A["x"], A["y"]):
        p = interp_xy(B, t)
        if p is None:
            continue
        dd = math.hypot(p[0] - x, p[1] - y)
        if worst is None or dd > worst[0]:
            worst = (dd, t)
    return worst
