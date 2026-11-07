#!/usr/bin/env python3
"""One-shot side-by-side of the OAK left image and the on-device depth, with rate and validity.

Run inside the skybeta container:  python3 /root/team_ws/tools/depth_snapshot.py [--seconds 10] [--out /tmp/depth_snapshot.png]
Prints JSON (rates by header stamp, valid fraction, centre range) and writes the PNG.
"""
import argparse, json, time
import cv2, numpy as np, rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

ap = argparse.ArgumentParser()
ap.add_argument('--seconds', type=float, default=10.0)
ap.add_argument('--out', default='/tmp/depth_snapshot.png')
ap.add_argument('--depth', default='/oak/stereo/image_raw')
ap.add_argument('--left', default='/oak/left/image_raw')
ap.add_argument('--max-m', type=float, default=4.0)
a = ap.parse_args()

S = {'d': [], 'l': [], 'depth': None, 'left': None}
rclpy.init(); n = Node('depth_snapshot')

def on_depth(m):
    S['d'].append(m.header.stamp.sec + m.header.stamp.nanosec * 1e-9)
    if m.encoding in ('16UC1', 'mono16'):
        S['depth'] = np.frombuffer(m.data, np.uint16).reshape(m.height, m.step // 2)[:, :m.width].astype(np.float32) * 1e-3
    elif m.encoding == '32FC1':
        S['depth'] = np.frombuffer(m.data, np.float32).reshape(m.height, m.step // 4)[:, :m.width].copy()

def on_left(m):
    S['l'].append(m.header.stamp.sec + m.header.stamp.nanosec * 1e-9)
    if m.encoding == 'bgr8':
        S['left'] = np.frombuffer(m.data, np.uint8).reshape(m.height, m.step // 3, 3)[:, :m.width].copy()
    else:
        S['left'] = cv2.cvtColor(np.frombuffer(m.data, np.uint8).reshape(m.height, m.step)[:, :m.width], cv2.COLOR_GRAY2BGR)

n.create_subscription(Image, a.depth, on_depth, qos_profile_sensor_data)
n.create_subscription(Image, a.left, on_left, qos_profile_sensor_data)
t0 = time.time()
while time.time() - t0 < a.seconds:
    rclpy.spin_once(n, timeout_sec=0.05)

def hz(ts):
    d = np.diff(ts)
    return round(1.0 / float(np.median(d)), 1) if len(d) > 2 else None

out = {'depth_hz_by_stamp': hz(S['d']), 'depth_msgs_received': len(S['d']), 'left_hz_by_stamp': hz(S['l'])}
if S['depth'] is not None:
    d = S['depth']; valid = d > 0
    h, w = d.shape
    cx, cy = w // 2, h // 2
    patch = d[cy - 5:cy + 5, cx - 5:cx + 5]; pv = patch[patch > 0]
    out.update(valid_frac=round(float(valid.mean()), 3), centre_m=round(float(np.median(pv)), 2) if pv.size else None,
               min_m=round(float(d[valid].min()), 2) if valid.any() else None)
    vis = np.clip(d / a.max_m, 0, 1)
    col = cv2.applyColorMap((255 - vis * 255).astype(np.uint8), cv2.COLORMAP_JET)
    col[~valid] = 0
    cv2.drawMarker(col, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 16, 1)
    cv2.putText(col, f"centre {out['centre_m']} m  {out['depth_hz_by_stamp']} Hz  valid {int(out['valid_frac'] * 100)}%",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    left = S['left']
    if left is not None:
        left = cv2.resize(left, (w, h))
        cv2.drawMarker(left, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 16, 1)
        cv2.imwrite(a.out, np.hstack([left, col]))
    else:
        cv2.imwrite(a.out, col)
    out['png'] = a.out
print(json.dumps(out))
