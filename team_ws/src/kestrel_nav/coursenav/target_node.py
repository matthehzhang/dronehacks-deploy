"""Coloured landing-target detector (ROS). Colour camera + depth -> /coursenav/target (JSON).

Publishes detections only; the navigator decides whether to move the final waypoint.
Pose and height come from the localizer's /coursenav/drift message (VIO pose in the
course frame and the sill-proof height), calibration from camera_info and TF.
No colour stream: the node idles and publishes nothing.
"""
from __future__ import annotations

import json
import math

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

from refnav.depth import decode_depth, quat_to_mat
from .scan import level_from_px4_q
from .target import depth_body_points, hsv_mask, largest_blob, level_to_course, locate_target

PX4_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                     history=HistoryPolicy.KEEP_LAST, depth=1)


def decode_rgb(m):
    enc = m.encoding.lower()
    if enc not in ('rgb8', 'bgr8', 'rgba8', 'bgra8'):
        raise ValueError(f'unsupported colour encoding {m.encoding}')
    ch = 4 if enc.endswith('a8') else 3
    a = np.frombuffer(bytes(m.data), np.uint8).reshape(m.height, m.step)[:, :m.width * ch].reshape(m.height, m.width, ch)
    a = a[..., :3]
    return a[..., ::-1] if enc.startswith('bgr') else a


class TargetNode(Node):
    def __init__(self):
        super().__init__('coursenav_target')
        dp = self.declare_parameter
        self.base_frame = dp('base_frame', 'base_link').value
        self.hsv = tuple(float(dp(k, v).value) for k, v in (('hue_lo', 95.), ('hue_hi', 125.), ('sat_min', 90.), ('val_min', 60.)))
        self.min_px = int(dp('min_blob_px', 40).value)
        self.period = 1. / float(dp('rate_hz', 4.).value)
        self.K_rgb = self.K_d = None
        self.cal = {}
        self.depth = None              # (stamp, metres array, frame)
        self.att = None
        self.pose = None               # (stamp, (x, y, yaw), height)
        self.last_t = -math.inf
        self.counts = dict(frames=0, blobs=0, published=0, rejected=0)
        from tf2_ros import Buffer, TransformListener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        from px4_msgs.msg import VehicleAttitude
        self.create_subscription(VehicleAttitude, '/fmu/out/vehicle_attitude', self._on_att, PX4_QOS)
        self.create_subscription(CameraInfo, dp('rgb_info_topic', '/oak/rgb/camera_info').value, self._on_rgb_info, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, dp('depth_info_topic', '/kestrel_nav/depth/camera_info').value, self._on_depth_info, qos_profile_sensor_data)
        self.create_subscription(Image, dp('depth_topic', '/kestrel_nav/depth').value, self._on_depth, qos_profile_sensor_data)
        self.create_subscription(Image, dp('rgb_topic', '/oak/rgb/image').value, self._on_rgb, qos_profile_sensor_data)
        self.create_subscription(String, dp('drift_topic', '/coursenav/drift').value, self._on_drift, 10)
        self.pub = self.create_publisher(String, dp('target_topic', '/coursenav/target').value, 10)
        self.get_logger().info(f'target detector: HSV {self.hsv}, min blob {self.min_px} px')

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_att(self, m):
        self.att = tuple(float(c) for c in m.q)

    def _on_rgb_info(self, m):
        self.K_rgb = (m.k[0], m.k[4], m.k[2], m.k[5])

    def _on_depth_info(self, m):
        self.K_d = (m.k[0], m.k[4], m.k[2], m.k[5])

    def _on_drift(self, m):
        try:
            d = json.loads(m.data)
            if d.get('height_m') is not None:
                self.pose = (float(d['stamp']), tuple(map(float, d['nominal'][:3])), float(d['height_m']))
        except (ValueError, KeyError, TypeError):
            pass

    def _on_depth(self, m):
        try:
            self.depth = (m.header.stamp.sec + m.header.stamp.nanosec * 1e-9,
                          decode_depth(m.encoding, m.height, m.width, m.step, bytes(m.data)), m.header.frame_id)
        except ValueError:
            pass

    def _tf(self, frame):
        if frame not in self.cal:
            from rclpy.time import Time
            tf = self.tf_buffer.lookup_transform(self.base_frame, frame, Time())
            r, t = tf.transform.rotation, tf.transform.translation
            self.cal[frame] = (quat_to_mat(r.w, r.x, r.y, r.z), np.array([t.x, t.y, t.z]))
        return self.cal[frame]

    def _on_rgb(self, m):
        now = self._now()
        if now - self.last_t < self.period or self.K_rgb is None or self.att is None or self.pose is None:
            return
        self.last_t = now
        self.counts['frames'] += 1
        try:
            rgb = decode_rgb(m)
            R_rgb, t_rgb = self._tf(m.header.frame_id)
        except Exception as e:                                   # no TF yet / encoding: try next frame
            self.get_logger().warning(f'target frame skipped: {e}', throttle_duration_sec=10.)
            return
        blob, npx = largest_blob(hsv_mask(rgb, *self.hsv), self.min_px)
        if blob is None:
            return
        self.counts['blobs'] += 1
        stamp = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        pts = np.zeros((0, 3))
        if self.depth is not None and self.K_d is not None and abs(self.depth[0] - stamp) <= .2:
            try:
                R_d, t_d = self._tf(self.depth[2])
                pts = depth_body_points(self.depth[1], self.K_d, R_d, t_d)
            except Exception:
                pass
        pose_t, pose, height = self.pose
        if abs(pose_t - stamp) > .5:
            return
        xy, info = locate_target(blob, self.K_rgb, R_rgb, t_rgb, pts, level_from_px4_q(self.att), height)
        info['blob_px'] = npx
        if xy is None:
            self.counts['rejected'] += 1
            return
        course = level_to_course(xy, pose)
        self.counts['published'] += 1
        self.pub.publish(String(data=json.dumps(dict(stamp=stamp, xy_course=course.round(4).tolist(), **info))))


def main(args=None):
    rclpy.init(args=args)
    node = TargetNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
