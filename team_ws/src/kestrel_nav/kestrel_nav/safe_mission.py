#!/usr/bin/env python3
"""Fly the DroneHacks course on skybeta with the SAFE policy (livenav.safe): stop-turn-go segments,
depth-centred gate crossings, autonomous descent onto the pad. Everything else (the skybeta contract,
input watchdogs, raw PX4 land checks, host depth, PF localizer, live grid) is live_mission's.

    publish    /skybeta/setpoint/twist   ENU world velocity + yaw rate (twist only: a pose setpoint is
                                         flown at PX4's own limits, MPC_XY_VEL_MAX 12 m/s, and never expires)

Two differences from live_mission at the node level:
  * bounded corrective motion (station keeping, cross-track, gate side-steps, pad centring; all capped at
    SafeParams.creep_mps = 0.06 m/s) does not need the forward depth corridor; any faster command does.
  * on the final descent over the pad the depth stream is waived (the camera looks at the floor from
    under 0.4 m, inside stereo minimum range); odometry, tracking, interface state and PX4 land state
    stay required. Touchdown: PX4's land detector, then COM_DISARM_LAND (2 s) disarms.
"""
from __future__ import annotations

import math

import numpy as np
import rclpy

from .live_mission import LiveMissionNode
from .livenav.replanner import ReplanParams
from .coursenav.scan import level_from_px4_q
from .livenav.pad import level_cam_rotation
from .livenav.safe import REACH_M, SafeMission, SafeParams
from .livenav import square_pad


class SafeMissionNode(LiveMissionNode):
    def __init__(self):
        self._corridor_wait = None
        super().__init__(node_name='safe_mission')

    def make_mission(self, mp, fp, course_file):
        dp = self.declare_parameter
        sp = SafeParams(
            v_cap=float(dp('v_cap', SafeParams.v_cap).value),
            v_gate=float(dp('v_gate_window', SafeParams.v_gate).value),
            turn_rate=math.radians(float(dp('turn_rate_deg', math.degrees(SafeParams.turn_rate)).value)),
            land_vz=float(dp('land_vz', SafeParams.land_vz).value),
            gate_requires_measurement=bool(dp('gate_requires_measurement', True).value),
        )
        if not 0 < sp.v_gate <= sp.v_cap <= 0.5:
            raise ValueError('speeds must satisfy 0 < v_gate_window <= v_cap <= 0.5 m/s')
        if not 0 < sp.turn_rate <= math.radians(45):
            raise ValueError('turn_rate_deg must be in (0, 45] (PX4 MPC_YAWRAUTO_MAX 45 deg/s)')
        if not 0 < sp.land_vz < 0.25:
            raise ValueError('land_vz must be below PX4 LNDMC_Z_VEL_MAX (0.25 m/s)')
        self.sp = sp
        return SafeMission(mp, None, ReplanParams(), sp, course_file=course_file)

    def announce_mission(self, mp, fp):
        plan, sp = self.mission.plan, self.sp
        self.get_logger().info(
            f"SAFE policy ready: chicane {plan.get('chicane')}, {len(self.mission.goals)} goals, {plan['length_m']} m; "
            f"stop-turn-go, tube speeds <= {sp.v_cap} m/s (window <= {sp.v_gate}), turns {math.degrees(sp.turn_rate):.0f} deg/s "
            f"in place; prop reach {REACH_M:.3f} m; gates {'measured or refused' if sp.gate_requires_measurement else 'map fallback'}; "
            f"lands on the pad. Waiting for Offboard.")

    CORRIDOR_WAIT_S = 5.0   # as 930am's depth-loss tolerance: stop and wait, latch only after this long

    def motion_corridor_error(self, cmd):
        """Bounded corrective motion needs no observed corridor. Faster motion into an unobserved
        corridor is STOPPED (horizontal command zeroed, the vehicle holds) and waits for coverage;
        only 5 s of continuous low coverage latches the fault. Gazebo @ 90021ff: rays past see-through
        nets into far background read as unobserved and a latched fault ended the run 13 s in."""
        if math.hypot(cmd.vx, cmd.vy) <= self.sp.creep_mps + 1e-6:
            return None
        reason = super().motion_corridor_error(cmd)
        if reason is None:
            self._corridor_wait = None
            return None
        now = self.now_s()
        if self._corridor_wait is None:
            self._corridor_wait = now
            self.get_logger().warning(f'stopping until the motion corridor is observed: {reason}')
        if now - self._corridor_wait >= self.CORRIDOR_WAIT_S:
            return f'{reason} for {self.CORRIDOR_WAIT_S:.0f} s'
        cmd.vx = cmd.vy = 0.0
        cmd.note = f'waiting for corridor depth: {reason}'
        return None

    # ------------------------------------------------------------------ venue pad (blue square, colour)
    def mono_info_cb(self, m):
        if m.k[0] > 0 and self.K_mono is None:
            self.K_mono = np.array(m.k, float).reshape(3, 3)
            self.mono_hw = (m.height, m.width)
            self.pad_detector = 'square'

    def mono_cb(self, m):
        """Square blue pad on a COLOUR image (pad_image via mono_topic). The skybeta stack streams only the
        mono OV9282 pair: on a grey image there is no hue, nothing is detected, and the landing uses the
        course-file pad position (logged as CAD FALLBACK)."""
        self.counts['mono'] += 1
        if self.pad_detector is None or self.mission.state not in ('FLY', 'PAD', 'DESCEND'):
            return
        rp = self.mission.replanner
        if rp.k < len(self.mission.goals) - 2:
            return                                                  # window crossing onward: pad in view
        now = self.now_s()
        if now - self.last_pad_run < 1.0 / self.pad_hz:
            return
        self.last_pad_run = now
        try:
            if m.encoding not in ('bgr8', 'rgb8'):
                self.info_once('pad_mono', f'pad image is {m.encoding}: no colour, landing will use CAD pad coordinates')
                return
            img = np.frombuffer(m.data, np.uint8).reshape(m.height, m.step // 3, 3)[:, :m.width]
            if m.encoding == 'rgb8':
                img = img[..., ::-1]
            img = np.ascontiguousarray(img)
        except ValueError:
            return
        if not square_pad.is_colour(img):
            self.info_once('pad_grey', 'pad image has equal channels (mono sensor): no hue, landing will use CAD pad coordinates')
            return
        R_body_cam, t_body_cam = self.camera_pose(m.header.frame_id)
        roll = pitch = 0.0
        if self.att is not None:
            Rl = level_from_px4_q(self.att)
            pitch, roll = -math.asin(max(-1.0, min(1.0, Rl[2, 0]))), math.atan2(Rl[2, 1], Rl[2, 2])
        R_level_cam = level_cam_rotation(R_body_cam, roll, pitch)
        det = square_pad.detect(img, self.K_mono, R_level_cam, np.asarray(t_body_cam, float), self.body_height())
        fix = square_pad.fix_from_detection(det, np.asarray(t_body_cam, float)[:2])
        if fix is not None and self.position is not None:
            fix["odom"] = (float(self.position[0]), float(self.position[1]), float(self.yaw))
            self.pad_fix, self.pad_fix_t = fix, now
            self.counts['pad'] += 1

    def tick(self):
        self.safety.waived = frozenset({'depth'}) if self.mission.state == 'DESCEND' else frozenset()
        super().tick()


def main(args=None):
    rclpy.init(args=args)
    node = SafeMissionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
