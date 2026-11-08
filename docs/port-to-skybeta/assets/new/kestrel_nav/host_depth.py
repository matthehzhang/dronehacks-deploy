#!/usr/bin/env python3
"""Host-side stereo depth from the OAK's mono pair, on the Jetson CPU. Touches nothing in the camera
pipeline: it only subscribes to the images the stack already publishes.

Why host-side: on-device StereoDepth through the depthai v3 driver raised the mono-image latency
into cuVSLAM from 37 ms to 220 ms on skybeta-14 (organizers' latency_probe, 2026-11-07 A/B), and
EKF2 on this airframe is tuned for a 60 ms vision delay. A subscriber cannot change that.

Lifted from the team's local_planner_viz sandbox (rectification from camera_info + the stack's
left<->right TF, OpenCV StereoSGBM 3-way), reduced to a publisher:

  in   /oak/left/image_mono, /oak/right/image_mono   mono8 (GPU-converted by the stack), 30 Hz
       /oak/{left,right}/camera_info, TF
  out  <depth_topic>                sensor_msgs/Image 16UC1 millimetres, rectified left frame
       <depth_topic>/camera_info    rectified intrinsics (P1) at the working scale
       TF static  oak_left_camera_optical_frame -> oak_left_rect_frame   (R1^T, no translation)

Rate-capped (max_hz) and niced; one pair in flight at a time (latest wins), so a slow frame drops
rather than queues.
"""
from __future__ import annotations

import array
import math
import threading
import time

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image


def quat_to_R(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def R_to_quat(R):
    """-> (x, y, z, w)"""
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        return ((R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s)
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
    q = [0.0, 0.0, 0.0, 0.0]
    q[i] = 0.25 * s
    q[j] = (R[j, i] + R[i, j]) / s
    q[k] = (R[k, i] + R[i, k]) / s
    q[3] = (R[k, j] - R[j, k]) / s
    return tuple(q)


def img_to_gray(msg):
    h, w = msg.height, msg.width
    buf = np.frombuffer(msg.data, np.uint8)
    if msg.encoding in ('mono8', '8UC1'):
        return buf.reshape(h, msg.step)[:, :w]
    if msg.encoding in ('bgr8', 'rgb8'):
        return cv2.cvtColor(buf.reshape(h, msg.step // 3, 3)[:, :w], cv2.COLOR_BGR2GRAY)
    raise ValueError(f'unsupported encoding {msg.encoding}')


class HostDepth(Node):
    def __init__(self):
        super().__init__('host_depth')
        dp = self.declare_parameter
        self.scale = float(dp('scale', 0.5).value)              # 1024x640 -> 512x320
        self.max_hz = float(dp('max_hz', 8.0).value)
        self.num_disp = max(16, int(dp('num_disp', 64).value) // 16 * 16)
        self.block = int(dp('block', 5).value) | 1
        self.min_m = float(dp('min_range_m', 0.3).value)
        self.max_m = float(dp('max_range_m', 6.0).value)
        self.threads = int(dp('cv_threads', 2).value)
        left = str(dp('left_topic', '/oak/left/image_mono').value)
        right = str(dp('right_topic', '/oak/right/image_mono').value)
        self.depth_topic = str(dp('depth_topic', '/kestrel_nav/depth').value)
        self.rect_frame = str(dp('rect_frame', 'oak_left_rect_frame').value)
        cv2.setNumThreads(self.threads)
        img_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=2)
        self.info = {}
        self.create_subscription(CameraInfo, '/oak/left/camera_info', lambda m: self.info.__setitem__('l', m), img_qos)
        self.create_subscription(CameraInfo, '/oak/right/camera_info', lambda m: self.info.__setitem__('r', m), img_qos)
        sl = Subscriber(self, Image, left, qos_profile=img_qos)
        sr = Subscriber(self, Image, right, qos_profile=img_qos)
        self.sync = ApproximateTimeSynchronizer([sl, sr], 4, 0.008)
        self.sync.registerCallback(self.on_pair)
        import tf2_ros
        self.tf_buf = tf2_ros.Buffer()
        self.tf_lis = tf2_ros.TransformListener(self.tf_buf, self)
        self.tf_static = tf2_ros.StaticTransformBroadcaster(self)
        self.pub_depth = self.create_publisher(Image, self.depth_topic, qos_profile_sensor_data)
        self.pub_info = self.create_publisher(CameraInfo, self.depth_topic + '/camera_info', qos_profile_sensor_data)
        self.lock = threading.Lock()
        self.pair = None
        self.evt = threading.Event()
        self.rect = None
        self.sgbm = cv2.StereoSGBM_create(
            minDisparity=0, numDisparities=self.num_disp, blockSize=self.block,
            P1=8 * self.block * self.block, P2=32 * self.block * self.block, disp12MaxDiff=1, uniquenessRatio=10,
            speckleWindowSize=100, speckleRange=2, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        self.n_out = 0
        self.t_proc = 0.0
        self.last_valid = 0.0
        threading.Thread(target=self.worker, daemon=True).start()
        self.create_timer(5.0, self.report)
        self.get_logger().info(f'host_depth: {left} + {right} -> {self.depth_topic} at <= {self.max_hz} Hz, scale {self.scale}, '
                               f'{self.num_disp} disparities, {self.threads} threads')

    # ------------------------------------------------------------------ input
    def on_pair(self, l, r):
        with self.lock:
            self.pair = (l, r)
        self.evt.set()

    def tf_mat(self, target, source):
        t = self.tf_buf.lookup_transform(target, source, rclpy.time.Time())
        q, p = t.transform.rotation, t.transform.translation
        return quat_to_R(q.x, q.y, q.z, q.w), np.array([p.x, p.y, p.z])

    # ------------------------------------------------------------------ rectification (from local_planner_viz)
    def build_rect(self):
        il, ir = self.info.get('l'), self.info.get('r')
        if il is None or ir is None:
            return False
        fl, fr = il.header.frame_id, ir.header.frame_id
        try:
            R_lr, t_lr = self.tf_mat(fl, fr)          # pose of the right camera in the left camera
        except Exception as e:  # noqa: BLE001
            self.get_logger().info(f'waiting for TF {fl} <- {fr}: {e}', throttle_duration_sec=5.0)
            return False
        swap = t_lr[0] < 0                            # the "right" topic is physically on the left
        if swap:
            il, ir, fl, fr = ir, il, fr, fl
            R_lr, t_lr = self.tf_mat(fl, fr)
        W, H = il.width, il.height
        size = (int(W * self.scale), int(H * self.scale))

        def kd(ci):
            K = np.array(ci.k, float).reshape(3, 3)
            D = np.array(ci.d, float)
            if D.size not in (4, 5, 8, 12, 14):
                D = np.zeros(5)
            return K, D
        K1, D1 = kd(il)
        K2, D2 = kd(ir)
        R = R_lr.T
        T = -R_lr.T @ t_lr
        R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(K1, D1, K2, D2, (W, H), R, T, flags=cv2.CALIB_ZERO_DISPARITY,
                                                   alpha=0, newImageSize=size)
        m1 = cv2.initUndistortRectifyMap(K1, D1, R1, P1, size, cv2.CV_16SC2)
        m2 = cv2.initUndistortRectifyMap(K2, D2, R2, P2, size, cv2.CV_16SC2)
        f = float(P1[0, 0])
        B = float(-P2[0, 3] / P2[0, 0])
        info = CameraInfo()
        info.header.frame_id = self.rect_frame
        info.width, info.height = size[0], size[1]
        info.distortion_model = 'plumb_bob'
        info.d = [0.0] * 5
        info.k = [f, 0.0, float(P1[0, 2]), 0.0, float(P1[1, 1]), float(P1[1, 2]), 0.0, 0.0, 1.0]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [float(v) for v in P1.ravel()]
        # static TF: rectified frame expressed in the raw left optical frame (P_opt = R1^T P_rect)
        tfm = TransformStamped()
        tfm.header.stamp = self.get_clock().now().to_msg()
        tfm.header.frame_id = fl
        tfm.child_frame_id = self.rect_frame
        qx, qy, qz, qw = R_to_quat(R1.T)
        tfm.transform.rotation.x, tfm.transform.rotation.y, tfm.transform.rotation.z, tfm.transform.rotation.w = qx, qy, qz, qw
        self.tf_static.sendTransform(tfm)
        self.rect = dict(swap=swap, m1=m1, m2=m2, f=f, B=B, size=size, info=info, left_frame=fl)
        self.get_logger().info(f'rectified {W}x{H} -> {size[0]}x{size[1]}  f={f:.1f}px  baseline={B * 1000:.1f}mm  '
                               f'swap={swap}  min depth {f * B / self.num_disp:.2f} m at {self.num_disp} disparities')
        return True

    # ------------------------------------------------------------------ worker
    def _rate_now(self):
        if self.get_parameter('use_sim_time').value:
            return self.get_clock().now().nanoseconds * 1e-9
        return time.time()

    def _wait_for_rate(self, last):
        period = 1.0 / max(self.max_hz, 0.5)
        if not self.get_parameter('use_sim_time').value:
            # Preserve the physical-drone scheduling path.
            wait = period - (time.time() - last)
            if wait > 0:
                time.sleep(wait)
            return
        # A sleep is in wall seconds; repeatedly check /clock rather than
        # treating a simulation interval as a wall interval at low RTF.
        while rclpy.ok():
            elapsed = self._rate_now() - last
            if elapsed < 0 or elapsed >= period:
                return
            time.sleep(max(0.005, min(0.05, period - elapsed)))

    def worker(self):
        last = 0.0
        while rclpy.ok():
            self.evt.wait(1.0)
            self.evt.clear()
            if self.rect is None and not self.build_rect():
                time.sleep(0.5)
                continue
            self._wait_for_rate(last)
            if not rclpy.ok():
                break
            with self.lock:
                pair, self.pair = self.pair, None
            if pair is None:
                continue
            last = self._rate_now()
            try:
                self.process(*pair)
            except Exception as e:  # noqa: BLE001
                self.get_logger().error(f'process: {e!r}', throttle_duration_sec=5.0)

    def process(self, lmsg, rmsg):
        rc = self.rect
        if rc['swap']:
            lmsg, rmsg = rmsg, lmsg
        t0 = time.perf_counter()
        L = cv2.remap(img_to_gray(lmsg), *rc['m1'], cv2.INTER_LINEAR)
        R = cv2.remap(img_to_gray(rmsg), *rc['m2'], cv2.INTER_LINEAR)
        disp = self.sgbm.compute(L, R).astype(np.float32) / 16.0
        valid = disp > 0.5
        depth_m = np.zeros_like(disp)
        depth_m[valid] = rc['f'] * rc['B'] / disp[valid]
        valid &= (depth_m > self.min_m) & (depth_m < self.max_m)
        depth_mm = np.where(valid, np.clip(depth_m * 1000.0, 0, 65535), 0).astype(np.uint16)
        out = Image()
        out.header.stamp = lmsg.header.stamp               # the pair's capture time, not now
        out.header.frame_id = self.rect_frame
        out.height, out.width = depth_mm.shape
        out.encoding = '16UC1'
        out.is_bigendian = 0
        out.step = out.width * 2
        # The ROS setter accepts a uint8 array directly; bytes triggers a Python loop.
        out.data = array.array('B', depth_mm.tobytes())
        info = rc['info']
        info.header.stamp = out.header.stamp
        self.pub_depth.publish(out)
        self.pub_info.publish(info)
        self.n_out += 1
        self.t_proc = 0.8 * self.t_proc + 0.2 * (time.perf_counter() - t0)
        self.last_valid = float(valid.mean())

    def report(self):
        if self.rect is not None:
            self.get_logger().info(f'depth frames {self.n_out}  proc {self.t_proc * 1000:.0f} ms  valid {self.last_valid * 100:.0f}%',
                                   throttle_duration_sec=4.0)


def main(args=None):
    rclpy.init(args=args)
    node = HostDepth()
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
