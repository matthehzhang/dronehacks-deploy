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

import rclpy

from .live_mission import LiveMissionNode
from .livenav.navgrid import NavParams
from .livenav.replanner import ReplanParams
from .livenav.safe import REACH_M, SafeMission, SafeParams


class SafeMissionNode(LiveMissionNode):
    def __init__(self):
        super().__init__(node_name='safe_mission')

    def make_mission(self, mp, fp, course_file):
        dp = self.declare_parameter
        sp = SafeParams(
            v_open=float(dp('v_open', SafeParams.v_open).value),
            v_tight=float(dp('v_tight', SafeParams.v_tight).value),
            v_gate=float(dp('v_gate_window', SafeParams.v_gate).value),
            turn_rate=math.radians(float(dp('turn_rate_deg', math.degrees(SafeParams.turn_rate)).value)),
            land_vz=float(dp('land_vz', SafeParams.land_vz).value),
            gate_requires_measurement=bool(dp('gate_requires_measurement', True).value),
        )
        if not 0 < sp.v_gate <= sp.v_tight <= sp.v_open <= 0.5:
            raise ValueError('speeds must satisfy 0 < v_gate_window <= v_tight <= v_open <= 0.5 m/s')
        if not 0 < sp.land_vz < 0.25:
            raise ValueError('land_vz must be below PX4 LNDMC_Z_VEL_MAX (0.25 m/s)')
        self.sp = sp
        return SafeMission(mp, NavParams(), ReplanParams(), sp, course_file=course_file)

    def announce_mission(self, mp, fp):
        plan, sp = self.mission.plan, self.sp
        self.get_logger().info(
            f"SAFE policy ready: chicane {plan.get('chicane')}, {len(self.mission.goals)} goals, {plan['length_m']} m; "
            f"stop-turn-go at {sp.v_tight}/{sp.v_open} m/s (window {sp.v_gate}), turns {math.degrees(sp.turn_rate):.0f} deg/s "
            f"in place; prop reach {REACH_M:.3f} m; gates {'measured or refused' if sp.gate_requires_measurement else 'map fallback'}; "
            f"lands on the pad. Waiting for Offboard.")

    def motion_corridor_error(self, cmd):
        if math.hypot(cmd.vx, cmd.vy) <= self.sp.creep_mps + 1e-6:
            return None
        return super().motion_corridor_error(cmd)

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
