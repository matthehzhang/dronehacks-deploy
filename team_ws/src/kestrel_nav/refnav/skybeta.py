"""refnav on the skybeta drone: refnav.core's output, unchanged, published through the organizer's flight
interface instead of /fmu/in. Added for deploy branch kestral_nav934am; not part of upstream group119.

The flight interface owns motion setpoints, arming and mode changes. The
pilot arms and switches to Position; skybeta_flight_interface enters Offboard (one attempt, needing a team
setpoint within 1 s) and turns /skybeta/setpoint/twist (ENU world velocity + yaw rate, expires after 1 s,
norm capped at 1.5 m/s) into PX4 setpoints. So:
  * the pilot and interface start the outbound flight. This adapter never sends
    arm or mode commands; a touch-and-go stays armed and in Offboard;
  * refnav's (n, e, d, yaw) setpoint + velocity feed-forward becomes a velocity command, the same loop
    PX4's position controller closes:  v = v_ff + kp * (sp - pos),  yawrate = yawspeed_ff + k_yaw * (yaw_sp - yaw);
    acceleration feed-forward cannot be carried by a twist and is dropped;
  * refnav's "land" (PX4 AUTO_LAND on the lab) has no interface equivalent: a 0.30 m/s descent with xy held
    where the landing started. After fresh PX4 touchdown confirmation, explicitly send a normal
    COMPONENT_ARM_DISARM command (param1=0, param2=0), retrying until PX4 reports disarmed.
    A planned touch-and-go continues upward on confirmed pad contact; only final
    landings and aborted returns explicitly disarm.
Frames: PX4 local NED (refnav) <-> the interface's ENU world frame (it applies enu_to_ned: (y, x, -z)).
"""
import math

KP_XY = 1.0          # PX4 MPC_XY_P default 0.95: the same proportional loop
KP_Z = 1.0           # PX4 MPC_Z_P default 1.0
KP_YAW = 2.0
V_XY_MAX = 0.6       # m/s; refnav's legs fly <= 0.30
V_Z_MAX = 0.5
YAWRATE_MAX = math.radians(45.0)   # PX4 MPC_YAWRAUTO_MAX
PAD_SNAP_M = 0.5     # the route ended this close to its last target (the pad): centre on it before descending
PAD_TOL_M = 0.03     # venue pad ~0.64 m square; 7-inch footprint 0.498 x 0.408 m: ~7 cm per side when centred and square
PAD_YAW_TOL = math.radians(3.0)
PAD_SETTLE_S = 1.0
PAD_ALIGN_TIMEOUT_S = 8.0
FOOTPRINT_CENTRE_X_M = -0.035   # rotor-disc footprint centre, body x (rotors (0.08,+-0.16), (-0.15,+-0.14), 7 in)
LAND_VZ = 0.30       # m/s down. PX4 1.16 MulticopterLandDetector: in_descend needs the setpoint vz >= 1.1 *
                     # LNDMC_Z_VEL_MAX (0.25 on the fleet) = 0.275, and ground_contact requires in_descend; at 0.15
                     # the drone sat armed on its legs (Gazebo, f30d131). PX4's own land speed default is 0.7.


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def twist_from_setpoint(sp, vel_ff, yawspeed_ff, pos, heading):
    """-> (vx_e, vy_n, vz_up, yawrate_enu) for the interface from a NED setpoint and the NED estimate."""
    n, e, d, yaw = sp
    pn, pe, pd = pos
    vn = (vel_ff[0] if vel_ff is not None else 0.0) + KP_XY * (n - pn)
    ve = (vel_ff[1] if vel_ff is not None else 0.0) + KP_XY * (e - pe)
    h = math.hypot(vn, ve)
    if h > V_XY_MAX:
        vn, ve = vn * V_XY_MAX / h, ve * V_XY_MAX / h
    vd = max(-V_Z_MAX, min(V_Z_MAX, KP_Z * (d - pd)))
    yr = 0.0
    if yaw is not None and math.isfinite(yaw) and heading is not None:
        yr = (yawspeed_ff or 0.0) + KP_YAW * _wrap(yaw - heading)
    yr = max(-YAWRATE_MAX, min(YAWRATE_MAX, yr))
    return ve, vn, -vd, -yr                     # ENU: x east, y north, z up; yaw rate about up


class SkybetaOutput:
    def __init__(self, node):
        from geometry_msgs.msg import TwistStamped
        from px4_msgs.msg import VehicleCommand
        self.node = node
        self.TwistStamped = TwistStamped
        self.VehicleCommand = VehicleCommand
        self.pub = node.create_publisher(TwistStamped, "/skybeta/setpoint/twist", 10)
        try:
            from skybeta_flight_interface.topics import resolve_endpoints
            command_topic = resolve_endpoints()['vehicle_command']
        except ImportError:
            version = getattr(VehicleCommand, 'MESSAGE_VERSION', 0)
            command_topic = '/fmu/in/vehicle_command' + (f'_v{version}' if version else '')
        self.command_pub = node.create_publisher(VehicleCommand, command_topic, 10)
        self.land_xy = None                     # NED (n, e) held during the landing descent
        self.land_yaw = None
        self.land_t0 = None
        self.centred_since = None
        self.descending = False
        self.disarm_sent_t = None
        self.touch_and_go_active = False

    def _disarm(self, now):
        """Normal PX4-checked disarm only; never use the force-disarm magic value."""
        previous = getattr(self, 'disarm_sent_t', None)
        if previous is not None and now - previous < 1.0:
            return
        command = self.VehicleCommand()
        command.timestamp = self.node._sys_us()
        command.command = self.VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM
        command.param1, command.param2 = 0.0, 0.0
        command.target_system = command.source_system = getattr(self.node, 'px4_system_id', 1)
        command.target_component = command.source_component = 1
        command.from_external = True
        self.command_pub.publish(command)
        self.disarm_sent_t = now
        self.node.get_logger().warning(
            "!!!!!!!!!! DISARM COMMAND SENT: PX4 reports landed; waiting for DISARMED confirmation !!!!!!!!!!")

    def _send(self, vx, vy, vz, wz):
        t = self.TwistStamped()
        t.header.stamp = self.node.get_clock().now().to_msg()
        t.header.frame_id = "odom"
        t.twist.linear.x, t.twist.linear.y, t.twist.linear.z = float(vx), float(vy), float(vz)
        t.twist.angular.z = float(wz)
        self.pub.publish(t)
        return dict(vx_e=vx, vy_n=vy, vz_up=vz, yawrate_enu=wz)

    def _plan_landing(self, now, v):
        """Where to land. At the end of the route (the last target, the pad, within PAD_SNAP_M): on that
        target, the footprint centre over it, facing the nearest pad side (pad axes = the course axes;
        the start heading faced along one). Elsewhere (an abort): straight down where it is."""
        m = self.node.mission
        self.land_t0, self.centred_since, self.descending = now, None, False
        self.disarm_sent_t = None
        targets = getattr(m, "targets", None) or []
        home = getattr(m, "home", None)
        if targets and math.dist(v.pos[:2], targets[-1]) <= PAD_SNAP_M and home is not None and v.heading is not None:
            q = math.pi / 2
            yaw = home[3] + round(_wrap(v.heading - home[3]) / q) * q
            n, e = targets[-1]
            self.land_xy = (n - FOOTPRINT_CENTRE_X_M * math.cos(yaw), e - FOOTPRINT_CENTRE_X_M * math.sin(yaw))
            self.land_yaw = _wrap(yaw)
            self.node.get_logger().info(f"landing on the pad: centring the footprint within {PAD_TOL_M} m, facing a "
                                        f"pad side ({math.degrees(self.land_yaw):.0f} deg NED), then {LAND_VZ} m/s down")
        else:
            self.land_xy, self.land_yaw, self.descending = (v.pos[0], v.pos[1]), v.heading, True
            self.node.get_logger().info(f"landing in place at {LAND_VZ} m/s until PX4 disarms")

    def publish(self, now, out, v):
        """One tick of refnav output -> at most one twist. Returns what was published (for the trace)."""
        returning = getattr(self.node.mission, 'stage', None) == 'return'
        if returning and not getattr(self, 'touch_and_go_active', False):
            # Confirmed contact starts the climb; release the pad descent latch.
            self.land_xy = self.land_yaw = self.land_t0 = self.centred_since = None
            self.descending = self.disarm_wait_logged = False
            self.disarm_sent_t = None
            self.touch_and_go_active = True
        if "land" in out.commands and v.pos is not None and self.land_xy is None:
            self._plan_landing(now, v)
        if self.land_xy is not None:
            if v.armed is False or v.pos is None:
                return None
            if now - self.land_t0 >= 60.0 and not getattr(self, 'disarm_wait_logged', False):
                self.disarm_wait_logged = True
                self.node.get_logger().warning(
                    "!!!!!!!!!! DISARM NOT CONFIRMED: PX4 has not reported disarmed; continuing landing !!!!!!!!!!")
            landed_t = getattr(self.node, '_land_received_ros_s', None)
            if (self.descending and v.armed is True and getattr(v, 'landed', None) is True
                    and landed_t is not None and 0 <= now - landed_t <= 1.0
                    and not getattr(self.node.mission, 'touch_and_go_pending', False)):
                self._disarm(now)
            sp = (self.land_xy[0], self.land_xy[1], v.pos[2], self.land_yaw)
            vx, vy, _, wz = twist_from_setpoint(sp, None, None, v.pos, v.heading)
            if not self.descending:
                off = math.dist(v.pos[:2], self.land_xy)
                yerr = abs(_wrap(self.land_yaw - v.heading)) if v.heading is not None else math.inf
                still = v.vel is None or math.hypot(v.vel[0], v.vel[1]) < 0.05
                ok = off <= PAD_TOL_M and yerr <= PAD_YAW_TOL and still
                self.centred_since = (self.centred_since or now) if ok else None
                if (self.centred_since is not None and now - self.centred_since >= PAD_SETTLE_S) \
                        or now - self.land_t0 >= PAD_ALIGN_TIMEOUT_S:
                    self.descending = True
                    self.node.get_logger().info(f"landing: descending at {LAND_VZ} m/s ({off:.3f} m, "
                                                f"{math.degrees(yerr):.1f} deg from the target)")
                else:
                    return self._send(vx, vy, 0.0, wz)
            notify_descent = getattr(self.node.mission, 'on_pad_descent', None)
            if notify_descent is not None:
                notify_descent(now)
            return self._send(vx, vy, -LAND_VZ, wz)
        if out.stream and out.setpoint is not None and v.pos is not None:
            return self._send(*twist_from_setpoint(out.setpoint, out.velocity, getattr(out, "yawspeed", None),
                                                   v.pos, v.heading))
        return None
