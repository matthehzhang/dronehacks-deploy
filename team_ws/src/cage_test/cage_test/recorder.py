"""``ros2 bag record`` (mcap) as a child process, started at the arm, stopped after the disarm.

Never blocks the node's timer: :meth:`stop` sends SIGINT (rosbag2's clean
shutdown) and :meth:`poll` escalates to SIGKILL if it has not exited after
``KILL_AFTER_S``. The child runs under coreutils ``timeout`` so an orphaned
recorder (node killed hard) still stops after ``max_s`` and cannot fill the disk.
"""
import os
import re
import shutil
import signal
import subprocess
import time

KILL_AFTER_S = 15.0

#: Recorded whenever they exist on the graph at the arm.
TOPIC_PATTERNS = [
    r'^/fmu/(in|out)/',                       # every input to and output from PX4
    r'^/skybeta/',                            # setpoints in, interface state out
    r'^/cage_test/',                          # this node's events and segment markers
    r'^/visual_slam/(tracking/odometry|status|tracking/vo_pose_covariance)$',
    r'^/oak/imu/data$',
    r'^/oak/(left|right)/camera_info$',
    r'^/tf(_static)?$',
    r'^/rosout$',                             # every node's log lines
]
#: Recorded even if not on the graph yet (rosbag2 picks them up when they appear).
ALWAYS = [
    '/fmu/in/trajectory_setpoint', '/fmu/in/offboard_control_mode',
    '/fmu/in/vehicle_visual_odometry', '/skybeta/setpoint/pose',
    '/skybeta/setpoint/twist', '/skybeta/state', '/cage_test/event',
    '/visual_slam/tracking/odometry', '/visual_slam/status', '/oak/imu/data', '/rosout',
]
#: ~39 MB/s together at 1024x640 mono8 30 Hz: only with record_images.
IMAGES = ['/oak/left/image_mono', '/oak/right/image_mono']


def select_topics(graph_names, record_images=False):
    names = set(ALWAYS)
    for name in graph_names:
        if any(re.search(p, name) for p in TOPIC_PATTERNS):
            names.add(name)
    if record_images:
        names.update(IMAGES)
    return sorted(names)


class BagRecorder:

    def __init__(self, logger=None):
        self.proc = None
        self.stop_sent_at = None
        self.path = None
        self.log = logger

    @staticmethod
    def available():
        return shutil.which('ros2') is not None

    @property
    def running(self):
        return self.proc is not None and self.proc.poll() is None

    def start(self, path, topics, max_s=900):
        cmd = ['ros2', 'bag', 'record', '-s', 'mcap', '-o', path] + list(topics)
        if shutil.which('timeout'):
            cmd = ['timeout', '--signal=INT', '--kill-after=10', str(int(max_s))] + cmd
        self.path = path
        self.stop_sent_at = None
        self.proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=open(path + '.stderr.log', 'w'),
            start_new_session=True)
        return cmd

    def stop(self):
        if not self.running or self.stop_sent_at is not None:
            return
        self.stop_sent_at = time.monotonic()
        try:
            os.killpg(self.proc.pid, signal.SIGINT)
        except ProcessLookupError:
            pass

    def poll(self):
        """True once there is no recorder left running."""
        if self.proc is None:
            return True
        if self.proc.poll() is not None:
            return True
        if self.stop_sent_at is not None and time.monotonic() - self.stop_sent_at > KILL_AFTER_S:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        return False

    def shutdown(self, wait_s=6.0):
        """Blocking stop, for node shutdown only."""
        if not self.running:
            return
        self.stop()
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline and self.proc.poll() is None:
            time.sleep(0.1)
        if self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
