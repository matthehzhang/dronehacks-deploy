"""Measured-lane diagnostic through the organizer's twist interface only."""
import json
import math
from pathlib import Path
import time

import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu

from .live_mission import LiveMissionNode
from .livenav.tuning import TuningProgram, validate_config



def _json_safe(value):
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_safe(x) for x in value]
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    return value


class TuningNode(LiveMissionNode):
    def __init__(self):
        self.trace_file = None
        self.trace_error = None
        self.dropped_count = 0
        self.imu_sample = None
        self.imu_frame = None
        super().__init__(node_name='tune_mission')
        topic = str(self.declare_parameter('imu_topic', '/oak/imu/data').value)
        self.create_subscription(Imu, topic, self.imu_cb, qos_profile_sensor_data)
        directory = Path(str(self.declare_parameter('log_directory', '~/team_ws/logs/tuning').value)).expanduser()
        directory.mkdir(parents=True, exist_ok=True)
        self.trace_path = directory/f'tuning_{time.time_ns()}.jsonl'
        self.trace_file = self.trace_path.open('x', buffering=1024*1024)
        self.record('header', config=self.mission.config, imu_topic=topic, hardware_qualified=False)

    def make_mission(self, mp, fp, course_file):
        path = str(self.declare_parameter('tuning_config', '').value)
        if not path:
            raise ValueError('tuning_config must name measured lane geometry; no default lane is safe')
        config = validate_config(json.loads(Path(path).expanduser().read_text()))
        if config.get('map_only') is True:
            raise ValueError('map_only geometry is not measured physical clearance; refused by ROS adapter')
        return TuningProgram(config, mp, fp, course_file)

    def announce_mission(self, mp, fp):
        self.get_logger().info(f'bounded tuning: {self.mission.estimated_total_s:.1f} s estimated; waiting for qualified pilot handover')

    def record(self, kind, source_stamp=None, frame_id='odom', **data):
        if self.trace_file is None:
            return
        st = self.mission.controller.status()
        row = dict(schema=1, kind=kind, t_ros=self.now_s(), t_mono=self.monotonic_s(),
                   source_stamp=source_stamp, frame_id=frame_id,
                   **{k: st[k] for k in ('phase', 'candidate', 'trial', 'trial_id', 'split', 'target_speed')})
        row.update(data)
        row['trial_id'] = f"{row['candidate']}:{row['trial']}"
        row['split'] = 'train' if row['trial'] % 2 == 0 else 'validation'
        try:
            self.trace_file.write(json.dumps(_json_safe(row), allow_nan=False)+'\n')
        except (OSError, ValueError) as exc:
            self.dropped_count += 1
            self.trace_error = f'logging failed: {exc}'
            self.safety.trip(self.trace_error)

    def odom_cb(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        before = self.safety.samples.get('odom')
        if msg.header.frame_id != 'odom':
            self.safety.reject('odom', 'diagnostic requires odom ENU frame for commands and fitting')
        else:
            super().odom_cb(msg)
        accepted = self.safety.samples.get('odom') != before
        self.record('odom', self.stamp_s(msg), msg.header.frame_id,
                    position=[p.x, p.y, p.z], quaternion_xyzw=[q.x, q.y, q.z, q.w],
                    covariance=list(msg.pose.covariance), accepted=accepted)

    def imu_cb(self, msg):
        stamp, now, mono = self.stamp_s(msg), self.now_s(), self.monotonic_s()
        a, g = msg.linear_acceleration, msg.angular_velocity
        values = (a.x, a.y, a.z, g.x, g.y, g.z)
        self.record('imu', stamp, msg.header.frame_id, acceleration=list(values[:3]), angular_velocity=list(values[3:]))
        if (not all(math.isfinite(x) for x in values) or not 0 <= now-stamp <= .3
                or not msg.header.frame_id or (self.imu_frame and self.imu_frame != msg.header.frame_id)):
            self.imu_sample = None
            if self.safety.active:
                self.safety.trip('invalid/stale/changed-frame IMU')
        elif self.imu_sample is None or stamp > self.imu_sample[0]:
            self.imu_sample = (stamp, mono)
            self.imu_frame = msg.header.frame_id

    def depth_cb(self, msg):
        super().depth_cb(msg)
        coverage, rays = (self.depth_frame.corridor_coverage(0., .40, .25, 1.)
                          if self.depth_frame is not None else (0., 0))
        self.record('depth', self.stamp_s(msg), msg.header.frame_id, coverage=coverage,
                    rays=rays, valid_fraction=self.depth_frame.valid_frac if self.depth_frame else 0.)

    def state_cb(self, msg):
        super().state_cb(msg)
        self.record('state', offboard_active=self.offboard_active, armed=self.armed,
                    clamped=self.clamped, safety=self.safety.action)

    def tracking_cb(self, msg):
        super().tracking_cb(msg)
        self.record('tracking', self.stamp_s(msg), vo_state=msg.vo_state)

    def send_twist(self, vx, vy, vz, yaw_rate):
        stamp = super().send_twist(vx, vy, vz, yaw_rate)
        published = float(stamp.sec)+float(stamp.nanosec)*1e-9
        self.record('command', published, velocity_enu=[vx, vy, vz], yaw_rate=yaw_rate,
                    offboard_active=self.offboard_active, clamped=self.clamped,
                    health=self.safety.action == 'run', safety=self.safety.action,
                    fault=self.safety.fault)
        if self.trace_file:
            try:
                self.trace_file.flush()
            except OSError as exc:
                self.trace_error = str(exc)
                self.safety.trip('diagnostic log flush failed')

    def tick(self):
        now, mono = self.now_s(), self.monotonic_s()
        sample = self.imu_sample
        imu_ok = sample is not None and 0 <= now-sample[0] <= .3 and 0 <= mono-sample[1] <= .3
        self.mission.imu_ok, self.mission.clamped = imu_ok, bool(self.clamped)
        self.mission.frame = self.depth_frame
        if not imu_ok:
            if self.safety.active or self.offboard_active:
                self.safety.trip('diagnostic IMU missing or stale')
            else:
                # No readiness stream until IMU qualification joins the inherited checks.
                self.safety.healthy_since = None
                return
        if self.clamped:
            self.safety.trip('organizer clamped diagnostic command')
        super().tick()
        while self.mission.controller.events:
            event = self.mission.controller.events.pop(0)
            self.record('event', **event)

    def destroy_node(self):
        if self.trace_file:
            self.record('summary', dropped_count=self.dropped_count, logging_error=self.trace_error,
                        result=self.mission.controller.status())
            self.trace_file.close()
            self.trace_file = None
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = TuningNode()
        rclpy.spin(node)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
