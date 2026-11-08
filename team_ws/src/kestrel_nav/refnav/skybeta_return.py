"""Confirmed pad contact, still-armed climb, circular return, final landing/disarm.

There is no rearm or mode-request path. Touchdown uses a fresh PX4 landed edge
after the pad descent starts; altitude or elapsed descent time cannot substitute.
"""
from dataclasses import replace
import math

import numpy as np

from . import core as C


def return_arc(start, finish, offset_m=.6096, side=1, speed_mps=.3, spacing_m=.025):
    """Circular arc with exactly offset_m sagitta from its endpoint chord, in NE.

    Positive side bends along (-east, north) relative to the direction of travel.
    Include the midpoint explicitly so the requested two-foot bulge is represented.
    """
    a, b = np.asarray(start, float), np.asarray(finish, float)
    h, speed = float(offset_m), float(speed_mps)
    length = float(np.linalg.norm(b-a))
    if (a.shape != (2,) or b.shape != (2,) or not np.isfinite([*a, *b, h, speed]).all()
            or h <= 0 or speed <= 0 or length <= 2*h or side not in (-1, 1)):
        raise ValueError('invalid circular return geometry; endpoint spacing must exceed twice the arc offset')
    normal = np.array([-(b-a)[1], (b-a)[0]]) / length
    radius = length*length/(8*h) + h/2
    centre = (a+b)/2 - side*(radius-h)*normal
    angle = 4*math.atan2(2*h, length)
    segments = max(20, int(math.ceil(radius*angle/spacing_m)))
    segments += segments % 2
    theta0 = math.atan2(a[1]-centre[1], a[0]-centre[0])
    theta = theta0 - side*np.linspace(0., angle, segments+1)
    points = centre + radius*np.column_stack((np.cos(theta), np.sin(theta)))
    points[0], points[-1] = a, b
    distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    speeds = np.minimum(speed, np.sqrt(np.minimum(distance, distance[-1]-distance)))
    return points, speeds


class SkybetaReturnMission(C.Mission):
    def __init__(self, params, spec):
        super().__init__(params)
        self.spec = spec if isinstance(spec, dict) else {}
        self.stage = 'outbound'
        self.launch_pose = None
        self.return_started = False
        self.pad_touched = False
        self.pad_descent_t = None
        self.landed_report = None
        self.touchdown_t = None
        self.return_height_since = None

    @property
    def accepts_outbound_paths(self):
        return self.stage == 'outbound'

    @property
    def touch_and_go_pending(self):
        return (self.stage == 'outbound' and self.phase == 'land' and self.land_reason == 'route complete'
                and self.land_request is None)

    def on_pad_descent(self, now):
        if self.touch_and_go_pending and self.pad_descent_t is None:
            self.pad_descent_t = now

    def on_landed(self, now, landed, source_stamp):
        previous = self.landed_report
        if previous is not None and source_stamp <= previous[2]:
            return  # a repeated old ground report cannot prove new contact
        self.landed_report = (now, bool(landed), source_stamp)
        if (self.touch_and_go_pending and self.pad_descent_t is not None and now >= self.pad_descent_t
                and previous is not None and previous[1] is False and landed):
            self.touchdown_t = now
            self.pad_touched = True

    def on_correction(self, *args, **kwargs):
        if self.accepts_outbound_paths:
            super().on_correction(*args, **kwargs)

    def on_path(self, *args, **kwargs):
        return super().on_path(*args, **kwargs) if self.accepts_outbound_paths else False

    def _begin_touch_and_go(self, now, v):
        height = float(self.spec['height_above_floor_m'])
        speed = float(self.spec.get('speed_mps', .3))
        if not math.isfinite(height) or height <= 0:
            raise ValueError('invalid return height')
        points, speeds = return_arc(v.pos[:2], self.launch_pose[:2],
                                    self.spec.get('arc_offset_m', .6096), self.spec.get('arc_side', 1), speed)
        target_d = self.launch_pose[2] + self.p.landed_z_m - height
        params = replace(self.p, route=[(0., 0.)],
                         legs=[dict(name='original_launch', mode='return', hold=True)],
                         height_m=v.pos[2]-target_d, track_mode='path', speed_mps=speed,
                         climb_tol_m=.05, correction_max_m=0., correction_rate_mps=0., tunnel_mode=False,
                         blend_turn_deg=0., hover_s=0., excite_yaw_deg=0., first_turn_gate_s=0.,
                         land_timeout_s=0., land_columns=None, safe_land=False, arrive_m=.15)
        replacement = C.Mission(params)
        for key in ('health', 'depth_ready_since', 'depth_t', 'frame', 'depth_pts', 'last_valid_frac'):
            setattr(replacement, key, getattr(self, key))
        replacement.home = (*v.pos, v.heading)
        replacement.sp = [*v.pos, v.heading]
        replacement.targets = [tuple(self.launch_pose[:2])]
        replacement.on_path(now, 0, points, speeds, pid=1)
        replacement.phase, replacement.offboard_seen = 'climb', True
        replacement.t_phase = replacement.t_started = replacement.t_last = now
        events = [f'!!!!!!!!!! PAD TOUCH CONFIRMED: remaining armed; climb vertically to {height:.2f} m, '
                  'then follow the curved return !!!!!!!!!!']
        out = C.Output('climb', setpoint=tuple(replacement.sp), velocity=(0., 0.), stream=True, events=events)
        self.__dict__.update(replacement.__dict__)
        self.stage, self.return_started, self.pad_touched = 'return', True, True
        return out

    def _next_leg(self, now, out, v, blended=False):
        if self.stage == 'return' and self.phase == 'climb':
            if self.return_height_since is None:
                self.return_height_since = now
            if now-self.return_height_since < 1.:
                return
            out.events.append('!!!!!!!!!! RETURN HEIGHT READY: vertical climb settled; starting curved return !!!!!!!!!!')
        return super()._next_leg(now, out, v, blended=blended)

    def step(self, now, v, wall_now=None):
        if self.stage == 'return':
            if v.armed is False and self.phase not in ('land', 'done'):
                out = C.Output('done')
                self._end(now, out, 'return stopped: PX4 disarmed; no rearm will be sent')
                self.stage = 'finished'
                return out
            if self.path is not None:
                self.path['t'] = now  # locally generated curve does not depend on navigator heartbeats
            if self.phase == 'climb':
                target = self.home[2]-self.p.height_m
                if v.pos is None or abs(v.pos[2]-target) >= self.p.climb_tol_m:
                    self.return_height_since = None
        report_fresh = self.landed_report is not None and 0 <= now-self.landed_report[0] <= 1.
        if (self.touch_and_go_pending and self.touchdown_t is not None and report_fresh
                and self.landed_report[1] and v.landed is True and v.armed is True
                and v.nav_state == C.NAV_OFFBOARD and v.pos_ok and v.pos is not None and v.heading is not None):
            try:
                return self._begin_touch_and_go(now, v)
            except Exception as exc:
                # An invalid optional return completes the pad landing/disarm instead.
                self.stage = 'finish_pad'
                out = super().step(now, v)
                out.events.append(f'!!!!!!!!!! RETURN SKIPPED: {exc}; completing pad landing and disarm !!!!!!!!!!')
                return out
        out = super().step(now, v)
        if self.stage == 'outbound' and self.launch_pose is None and self.home is not None:
            self.launch_pose = tuple(self.home)
        if out.phase == 'done':
            was_returning = self.stage == 'return'
            if self.stage == 'outbound' and self.result == 'complete':
                out.events.append('!!!!!!!!!! TOUCH AND GO SKIPPED: PX4 already disarmed; no rearm will be sent !!!!!!!!!!')
            self.stage = 'finished'
            if was_returning and self.result == 'complete':
                out.events.append('!!!!!!!!!! RETURN COMPLETE: landed and disarmed at original launch !!!!!!!!!!')
        return out

    def status(self, now):
        return dict(super().status(now), mission_stage=self.stage, return_started=self.return_started,
                    pad_touched=self.pad_touched, launch_pose=self.launch_pose,
                    return_height_m=self.spec.get('height_above_floor_m'))
