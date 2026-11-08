#!/usr/bin/env python3
"""Supervise a lightweight rosbag in a separate container; never command the drone."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time


TOPICS = (r'^(/fmu/(out|in)/.*|/rosout|/tf|/tf_static|/oak/imu/data|'
          r'/visual_slam/(tracking/odometry|status)|/skybeta/(state|setpoint/pose|setpoint/twist)|'
          r'/estimator/healthy|/refnav/status|/kestrel_nav/live_status|'
          r'/coursenav/(status|drift|px4_correction))$')


class EndOfRun:
    def __init__(self):
        self.ever_armed = False
        self.done = False
        self.ground_since = None
        self.state_at = None
        self.last_message = None
        self.return_in_progress = False

    def status(self, data, now):
        self.done = data.get('phase') == 'done'
        self.return_in_progress = data.get('mission_stage') == 'return' and not self.done
        self.last_message = now

    def state(self, data, now):
        self.last_message = now
        if self.state_at is None or now - self.state_at > 2:
            self.ground_since = None
        self.state_at = now
        self.ever_armed |= data.get('armed') is True
        ground = data.get('armed') is False and data.get('landed') is True
        if ground and (self.ever_armed or self.done):
            if self.ground_since is None:
                self.ground_since = now
        else:
            self.ground_since = None

    def reason(self, now):
        if (not self.return_in_progress and self.ground_since is not None and self.state_at is not None
                and now - self.state_at < 2 and now - self.ground_since >= 5):
            return 'run ended, disarmed and landed; five-second post-roll saved'
        if self.last_message is not None and now - self.last_message > 60:
            return 'mission/state telemetry absent for 60 seconds'
        return None


def main():
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String

    output = Path('/out')
    reason = 'recorder exited'
    requested = []
    rclpy.init()
    node = Node('group154_bag_supervisor')
    end = EndOfRun()

    def receive(callback, message):
        try:
            data = json.loads(message.data)
            if isinstance(data, dict):
                callback(data, time.monotonic())
        except ValueError:
            pass

    node.create_subscription(String, '/refnav/status', lambda m: receive(end.status, m), 10)
    node.create_subscription(String, '/skybeta/state', lambda m: receive(end.state, m), 10)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda signum, frame: requested.append(signum))
    command = ['ros2', 'bag', 'record', '-s', 'mcap', '-o', str(output / 'bag'),
               '--max-cache-size', '16777216', '-e', TOPICS]
    (output / 'recording.json').write_text(json.dumps({'command': command, 'started_unix': time.time()}, indent=2))
    started = time.monotonic()
    with (output / 'rosbag.log').open('w') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while rclpy.ok() and process.poll() is None:
                rclpy.spin_once(node, timeout_sec=0.2)
                now = time.monotonic()
                if not (output / 'recording.ready').exists() and list((output / 'bag').glob('*.mcap')):
                    (output / 'recording.ready').touch(exist_ok=True)
                if requested:
                    reason = 'stop requested'
                    break
                reason = end.reason(now)
                if reason:
                    break
                if end.last_message is None and now - started > 180:
                    reason = 'no mission/state telemetry within 180 seconds'
                    break
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=5)
            (output / 'recording.ready').unlink(missing_ok=True)
            (output / 'finished.json').write_text(json.dumps({
                'reason': reason or 'rosbag process exited', 'returncode': process.returncode,
                'finished_unix': time.time(), 'metadata_saved': (output / 'bag/metadata.yaml').exists(),
            }, indent=2))
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
    return 0 if (output / 'bag/metadata.yaml').exists() else 1


if __name__ == '__main__':
    raise SystemExit(main())
