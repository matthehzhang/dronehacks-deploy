"""One automatic return on skybeta: pad disarm, vertical climb, original launch, disarm.

Uses the existing flight controller and its input handling. The return has no
battery/localizer/map-confidence/time-budget qualification from the native-PX4
return workflow. Its destination is this run's remembered launch position.
"""
from dataclasses import replace
import math

from . import core as C


class SkybetaReturnMission(C.Mission):
    def __init__(self, params, spec):
        super().__init__(params)
        self.spec = spec if isinstance(spec, dict) else {}
        self.stage = 'outbound'
        self.launch_pose = None
        self.return_started = False
        self.outbound_complete = False
        self.return_height_since = None

    @property
    def accepts_outbound_paths(self):
        return self.stage == 'outbound'

    def on_correction(self, *args, **kwargs):
        if self.accepts_outbound_paths:
            super().on_correction(*args, **kwargs)

    def on_path(self, *args, **kwargs):
        return super().on_path(*args, **kwargs) if self.accepts_outbound_paths else False

    def _begin_return(self, now, v, events):
        # Keep the original floor reference even when the pad is raised or the
        # second takeoff has a different local Z. Never reinterpret 2.1 m as a
        # climb of 2.1 m from the pad or use a new map-localized launch target.
        height = float(self.spec['height_above_floor_m'])
        speed = float(self.spec.get('speed_mps', .3))
        if not math.isfinite(height) or height <= 0 or not math.isfinite(speed) or speed <= 0:
            raise ValueError('invalid return height/speed')
        target_d = self.launch_pose[2] + self.p.landed_z_m - height
        dn, de = self.launch_pose[0] - v.pos[0], self.launch_pose[1] - v.pos[1]
        c, s = math.cos(v.heading), math.sin(v.heading)
        params = replace(self.p, route=[(dn*c + de*s, dn*s - de*c)],
                         legs=[dict(name='original_launch', mode='return', hold=True)],
                         height_m=v.pos[2] - target_d, track_mode='pose',
                         speed_mps=speed, climb_tol_m=.05,
                         correction_max_m=0., correction_rate_mps=0., tunnel_mode=False,
                         blend_turn_deg=0., hover_s=0., excite_yaw_deg=0., first_turn_gate_s=0.,
                         accept_armed_start=False, land_timeout_s=0., land_columns=None,
                         safe_land=False, arrive_m=.1)
        retained = {key: getattr(self, key) for key in
                    ('health', 'depth_ready_since', 'depth_t', 'frame', 'depth_pts', 'last_valid_frac')}
        replacement = C.Mission(params)
        for key, value in retained.items():
            setattr(replacement, key, value)
        replacement.home = (*v.pos, v.heading)
        replacement.sp = [*v.pos, v.heading]
        replacement.targets = [tuple(self.launch_pose[:2])]
        replacement.phase = 'arm'
        replacement.t_phase = replacement.t_started = replacement.t_last = now
        events.append(f'!!!!!!!!!! RETURN START: rearm, hold XY, climb to {height:.2f} m above floor; '
                      'fly to original launch only after measured height settles !!!!!!!!!!')
        out = C.Output('arm', setpoint=tuple(replacement.sp), velocity=(0., 0.), stream=True, events=events)
        # Prepare everything before replacing the completed outbound mission.
        self.__dict__.update(replacement.__dict__)
        self.stage, self.return_started = 'return', True
        return out

    def _next_leg(self, now, out, v, blended=False):
        if self.stage == 'return' and self.phase == 'climb':
            if self.return_height_since is None:
                self.return_height_since = now
            if now - self.return_height_since < 1.:
                return
            out.events.append('!!!!!!!!!! RETURN HEIGHT READY: 2.1 m climb settled; starting return transit !!!!!!!!!!')
        return super()._next_leg(now, out, v, blended=blended)

    def step(self, now, v, wall_now=None):
        if self.stage == 'return' and self.phase == 'climb':
            target = self.home[2] - self.p.height_m
            if v.pos is None or abs(v.pos[2] - target) >= self.p.climb_tol_m:
                self.return_height_since = None
        out = super().step(now, v)
        if self.stage == 'outbound':
            if self.launch_pose is None and self.home is not None:
                self.launch_pose = tuple(self.home)
            if out.phase == 'done':
                if (self.result == 'complete' and self.land_reason == 'route complete'
                        and v.armed is False and v.landed is True and v.pos_ok
                        and v.pos is not None and v.heading is not None):
                    self.outbound_complete = True
                    try:
                        return self._begin_return(now, v, out.events)
                    except Exception as exc:
                        # This boundary runs only after successful landing/disarm.
                        # An optional-return setup failure preserves that finished run.
                        out.events.append(f'!!!!!!!!!! RETURN SKIPPED: {exc}; outbound landing complete !!!!!!!!!!')
                self.stage = 'finished'
        elif self.stage == 'return' and out.phase == 'done':
            self.stage = 'finished'
            if self.result == 'complete':
                out.events.append('!!!!!!!!!! RETURN COMPLETE: landed and disarmed at original launch !!!!!!!!!!')
        return out

    def status(self, now):
        return dict(super().status(now), mission_stage=self.stage, return_started=self.return_started,
                    outbound_complete=self.outbound_complete, launch_pose=self.launch_pose,
                    return_height_m=self.spec.get('height_above_floor_m'))
