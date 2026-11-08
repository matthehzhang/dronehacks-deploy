"""Derive one bounded diagnostic lane and its native transit from the current map.

Pure planning: no ROS, truth, vehicle commands, or saved-run coordinates. The
configured tunnel semantic supplies the axis. Map clearance is an assumption
about the published prior, not a physical survey or hardware qualification.
"""
from __future__ import annotations

import copy
import math

import numpy as np
from scipy.spatial import cKDTree

from . import planner
from .course import load_course, plan_grids, resolve
from .navigator import Navigator
from .navplan import smooth_params_for, stops_and_legs
from .objmap import CourseMap


TRANSIT_CEILING_MPS = .15


class MapClearanceCertificate:
    """Distance to occupied cell squares, with a conservative segment bound."""

    def __init__(self, grid):
        self.grid = grid
        cells = np.argwhere(grid.occ)
        self.tree = cKDTree(grid.xy(cells)) if len(cells) else None
        self.half_diagonal = grid.res / math.sqrt(2)

    def points(self, xy):
        q = np.atleast_2d(np.asarray(xy, float))
        if q.shape[1] != 2 or not np.isfinite(q).all():
            raise ValueError("clearance points must be finite XY coordinates")
        distance = (self.tree.query(q)[0] - self.half_diagonal
                    if self.tree is not None else np.full(len(q), np.inf))
        # Unmapped space is not certified free, including a disk crossing the
        # edge when its centre happens to be inside the map.
        lo = self.grid.origin
        hi = lo + np.asarray(self.grid.occ.shape) * self.grid.res
        boundary = np.minimum(q - lo, hi - q).min(axis=1)
        return np.minimum(distance, boundary)

    def line(self, a, b):
        length = float(np.linalg.norm(np.asarray(b) - a))
        n = max(2, int(math.ceil(length / (self.grid.res / 2))) + 1)
        # Distance to a closed obstacle set is 1-Lipschitz: every point between
        # samples is at most half an interval from a checked sample.
        return float(self.points(np.linspace(a, b, n)).min() - length / (n - 1) / 2)


def _settings(doc, landed_z):
    params = {k: copy.deepcopy(v) for k, v in doc.get("refnav_params", {}).items() if not k.startswith("_")}
    nav = {k: copy.deepcopy(v) for k, v in doc.get("navigation", {}).items() if not k.startswith("_")}
    params.update({k: v for k, v in nav.pop("refnav", {}).items() if not k.startswith("_")})
    nav.pop("mode", None)
    # The diagnostic's single slow transit keeps flow-corners' angle-only arc selection: arc_min_tube_m
    # (course mission, group154) turns its maze bends into stops, which this planner's one-stop transit
    # contract does not accept. The diagnostic mission is not what main.py flies on this branch.
    nav.pop("arc_min_tube_m", None)
    speed = min(float(params.get("speed_mps", TRANSIT_CEILING_MPS)), TRANSIT_CEILING_MPS)
    if not math.isfinite(speed) or speed <= 0:
        raise planner.PlanError("diagnostic transit speed must be positive")
    params.update(speed_mps=speed, track_mode="path", correction_max_m=0.0,
                  height_m=float(doc["cruise_z_m"]) - landed_z)
    for key in ("tunnel_speed_mps", "side_min_mps"):
        params[key] = min(float(params.get(key, speed)), speed)
    nav["cruise_mps"] = min(float(nav.get("cruise_mps", speed)), speed)
    nav["slow_min_mps"] = min(float(nav.get("slow_min_mps", nav["cruise_mps"])), nav["cruise_mps"])
    if not all(math.isfinite(v) and v > 0 for v in
               (params["height_m"], nav["cruise_mps"], nav["slow_min_mps"], params["tunnel_speed_mps"], params["side_min_mps"])):
        raise planner.PlanError("diagnostic height and speed must be positive")
    return params, nav


def _approach_goals(doc, feature):
    """Retain configured course goals preceding this semantic tube's entry."""
    goals = []
    entry_name = f"{feature}_entry"
    for goal in doc.get("sequence", []):
        if goal.get("name") == entry_name:
            return goals
        if "xy" not in goal or goal.get("choose") or goal.get("straight") or goal.get("land"):
            raise planner.PlanError("diagnostic approach contains an unsupported crossing or landing")
        goals.append(dict(name=goal["name"], xy=list(goal["xy"]), mode="cruise"))
    raise planner.PlanError("configured tube entry is missing from the resolved course sequence")


def plan_diagnostic(start_pose, landed_z, doc=None, cmap=None, *, max_flight_s=170.0, speeds=(.1,)):
    """Return ``(plan, route_doc)`` for this measured course ``(x,y,yaw_deg)``.

    The transit uses normal navigator paths and stops at the lane's first open
    endpoint. The diagnostic adapter then owns the paired experiment. No pad
    goal, saved physical configuration, or automatic return is inherited.
    """
    from refnav.diagnostic_protocol import validate_config
    from .course import land_columns

    start = np.asarray(start_pose, float)
    if start.shape != (3,) or not np.isfinite(start).all() or not math.isfinite(landed_z):
        raise planner.PlanError("finite measured start pose and landed height are required")
    if landed_z < 0:
        raise planner.PlanError("landed body height cannot be negative")
    doc = copy.deepcopy(load_course() if doc is None else doc)
    cmap = CourseMap.load(resolve(doc, "map")) if cmap is None else cmap
    need = float(doc["vehicle"]["radius_m"]) + float(doc["vehicle"]["margin_m"])
    if not math.isfinite(need) or need < .40 - 1e-9:
        raise planner.PlanError("diagnostic prior planning requires at least 0.40 m clearance")
    params, nav_cfg = _settings(doc, float(landed_z))
    # The diagnostic's certificate below is defined on the voxel raster ("occupied-cell-centre
    # distance minus half-cell diagonal"), so it keeps the raster field. On the surface-derived
    # field (course.surface_clearance) its navigator transit dips to 0.397 m < 0.414 m beside M1
    # and every candidate is refused (matthew930am, 2026-11-07); the course mission does not use it.
    obst, blocked, clear = plan_grids(cmap, doc, surface=False)
    # Every configured opening stays closed during transit and this diagnostic
    # lane; passing through a gate would require its separate crossing protocol.
    gate_lines = [[np.asarray(p, float).tolist() for p in planner.gate_segment(g)]
                  for g in doc.get("chicane_options", {}).values()]
    gated = planner.with_walls(blocked, gate_lines)
    cert = MapClearanceCertificate(obst)
    feature = "tunnel"
    try:
        tunnel = doc["derived"][feature]
        centre = np.asarray(tunnel["axis_xy"], float)
        axis = np.asarray(tunnel["entry"], float) - np.asarray(tunnel["exit"], float)
        length = float(tunnel["length_m"])
        if centre.shape != (2,) or axis.shape != (2,) or not np.isfinite([*centre, *axis, length]).all() or length <= 0 or np.linalg.norm(axis) <= 0:
            raise ValueError("invalid axis")
    except (KeyError, TypeError, ValueError) as exc:
        raise planner.PlanError("valid configured tunnel semantic geometry is required") from exc
    axis /= np.linalg.norm(axis)
    mouth_start, mouth_end = centre + axis * length / 2, centre - axis * length / 2
    approach = _approach_goals(doc, feature)
    # The protocol limits total lane length to eight metres. Enumerate endpoint
    # offsets at half map resolution, beginning one complete planning clearance
    # beyond each mouth; no body-centre-only 'outside' shortcut is accepted.
    offsets = np.arange(need, 8.0 - length - need + 1e-9, obst.res / 2)
    starts = [(mouth_start + axis * d, float(d)) for d in offsets]
    ends = [(mouth_end - axis * d, float(d)) for d in offsets]
    starts = [(q, d) for q, d in starts if cert.points(q)[0] >= .65 and planner.segment_free(gated, q, q)]
    ends = [(q, d) for q, d in ends if cert.points(q)[0] >= .65 and planner.segment_free(gated, q, q)]
    if not starts or not ends:
        raise planner.PlanError("no mapped open turning zones outside the tube mouths")
    best = None
    for a, start_offset in starts:
        goals = approach + [dict(name="diagnostic_lane_start", xy=a.tolist(), mode="align")]
        try:
            plan = planner.plan(doc, obst, gated, clear, start_xy=start[:2], goals=goals, cmap=cmap)
        except planner.PlanError:
            continue
        # Passing the gated grid explicitly also protects simplification and
        # moved via points; preserve the same exclusion lines for live replans.
        plan["gate_lines"] = gate_lines
        route, legs, nav_legs = stops_and_legs(plan, start[:2], math.radians(start[2]))
        if len(route) != 1 or not nav_legs[0]["path"] or legs[0].get("align") or legs[0].get("tunnel"):
            raise planner.PlanError("diagnostic transit must end in one ordinary path stop")
        navigator = Navigator(obst, gated, clear, need, smooth_params_for(nav_cfg, params, need))
        try:
            path = navigator.plan(start[:2], nav_legs[0]["goals"])
        except planner.PlanError:
            continue
        # A waypoint route and a navigator path are distinct artifacts. Check
        # the path the native adapter will actually request, including keepouts.
        if any(not planner.segment_free(gated, x, y) or
               planner.segment_clearance(obst, clear, x, y) < need - 1e-9
               for x, y in zip(path.pts[:-1], path.pts[1:])):
            continue
        transit_estimate = max(float(plan["length_m"]) / params["speed_mps"], path.duration()) + 5.0
        for b, end_offset in ends:
            lane_length = float(np.linalg.norm(b-a))
            if not 2 <= lane_length <= 8 or not planner.segment_free(gated, a, b):
                continue
            half_width = cert.line(a, b)
            if half_width < .5:
                continue
            lane = dict(name="published_semantic_axis", kind="tube", start_xy=a.tolist(), end_xy=b.tolist(),
                        height_m=float(doc["cruise_z_m"]), half_width_m=min(3.0, half_width),
                        start_turn_radius_m=min(3.0, float(cert.points(a)[0])),
                        end_turn_radius_m=min(3.0, float(cert.points(b)[0])), start_open=True, end_open=True)
            try:
                config = validate_config(dict(start_course=start.tolist(), landed_z_m=float(landed_z),
                                              speeds_mps=list(speeds), time_limit_s=float(max_flight_s),
                                              stationkeeping_policy='native_position_hold_v1',
                                              turn_excursion_reference='captured_hold',
                                              minimum_clearance_m=need,
                                              candidates=[lane], map_only=True))
            except ValueError:
                continue
            total = transit_estimate + config["estimated_lane_s"]
            if total > config["time_limit_s"] or (best is not None and total >= best[0]):
                continue
            config["estimated_total_s"] = total
            # Match coursenav.nav_node's native PX4/VIO frame contract. This is
            # the declared ENU heading used by the existing workspace bridge,
            # not a lane heading or fitted course-coordinate correction.
            config["declared_heading_rad"] = math.pi / 2
            config["budget_note"] = "Startup feasibility estimate, not an upper bound; live elapsed-time limit still applies."
            config["geometry_basis"] = "Published prior map only; not a physical survey or hardware qualification."
            geometry = dict(required_prior_clearance_m=need, mouth_start_xy=mouth_start.tolist(),
                            mouth_end_xy=mouth_end.tolist(), start_outside_m=start_offset, end_outside_m=end_offset,
                            lane_length_m=lane_length, certified_half_width_m=half_width,
                            map_resolution_m=obst.res, height_band_m=[doc["cruise_z_m"]-doc["plan_band_m"], doc["cruise_z_m"]+doc["plan_band_m"]],
                            certificate="Occupied-cell-centre distance minus half-cell diagonal; lane additionally subtracts half sample interval.",
                            transit_path_length_m=path.length, transit_profile_duration_s=path.duration(),
                            transit_min_prior_grid_clearance_m=float(path.clearance.min()),
                            transit_speed_ceiling_mps=TRANSIT_CEILING_MPS)
            plan["diagnostic_geometry"] = geometry
            route_doc = dict(route=route, legs=legs, nav_legs=nav_legs, params=params,
                             land_columns=land_columns(doc, cmap, start[:2].tolist(), float(start[2])),
                             navigation=nav_cfg, gate_lines=gate_lines, diagnostic=config)
            best = total, copy.deepcopy(plan), copy.deepcopy(route_doc)
    if best is None:
        raise planner.PlanError("no certified lane and gated transit fit the diagnostic startup budget")
    return best[1], best[2]
