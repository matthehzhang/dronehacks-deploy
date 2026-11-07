"""One directory per run (arm -> disarm): what was commanded, what every sensor said.

No ROS in here. Layout of ``<log_dir>/<YYYYmmdd-HHMMSS>_<sequence>/``:

    meta.json        parameters, the planned segments, drone identity, result, bag topics
    samples.csv      every node tick: state, segment, the command sent (the input),
                     VIO pose, PX4 pose and yaw, odometry age, battery, flags
    imu_oak.csv      OAK-D IMU (BNO086) at full rate: t (receipt), stamp (header), accel, gyro
    imu_px4.csv      PX4 sensor_combined as it arrives: t (receipt), stamp (FC boot s), accel, gyro
    events.jsonl     every log line and state change, timestamped
    summary.json     per-segment numbers (``analysis.summarize``)
    summary.txt      the same as a table
    bag/             rosbag2 mcap of every input to and output from PX4 (``recorder``)

The first ``PREROLL_S`` seconds before the arm are written from memory, so
the stationary baseline is in the CSVs even though the run starts at the arm.
"""
import csv
import json
import os
import re
import socket
import time

SAMPLE_COLUMNS = [
    't', 't_run', 'state', 'seg', 'seg_name', 'phase',
    'cmd_kind', 'cmd_x', 'cmd_y', 'cmd_z', 'cmd_yaw', 'cmd_vz',
    'vio_x', 'vio_y', 'vio_z', 'vio_yaw', 'vio_vx', 'vio_vy', 'vio_vz',
    'px4_x', 'px4_y', 'px4_z', 'px4_yaw',
    'odom_age', 'armed', 'offboard', 'landed', 'clamped', 'battery_v',
]
IMU_COLUMNS = ['t', 'stamp', 'ax', 'ay', 'az', 'gx', 'gy', 'gz']
PREROLL_S = 10.0


def default_log_dir():
    """``/root/team_ws/runs`` inside the stack (host ``~/team_ws/runs``), else ``~/cage_test_runs``."""
    if os.path.isdir('/root/team_ws'):
        return '/root/team_ws/runs'
    return os.path.join(os.path.expanduser('~'), 'cage_test_runs')


def free_gb(path):
    try:
        st = os.statvfs(path)
    except OSError:
        return None
    return st.f_bavail * st.f_frsize / 1e9


def _fmt(v):
    if v is None:
        return ''
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, float):
        return f'{v:.4f}'
    return v


class RunLog:

    def __init__(self, base_dir, label, meta):
        stamp = time.strftime('%Y%m%d-%H%M%S')
        safe = re.sub(r'[^A-Za-z0-9_.-]+', '-', label).strip('-') or 'run'
        os.makedirs(base_dir, exist_ok=True)
        self.path = os.path.join(base_dir, f'{stamp}_{safe}')
        n = 1
        while os.path.exists(self.path):
            n += 1
            self.path = os.path.join(base_dir, f'{stamp}_{safe}_{n}')
        os.makedirs(self.path)
        self.meta = dict(meta)
        self.meta.update({
            'run_dir': self.path,
            'started_wall': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
            'hostname': socket.gethostname(),
            'ros_domain_id': os.environ.get('ROS_DOMAIN_ID'),
            'drone_id': os.environ.get('DRONE_ID'),
        })
        self._files = {}
        self._writers = {}
        for name, cols in (('samples', SAMPLE_COLUMNS), ('imu_oak', IMU_COLUMNS),
                           ('imu_px4', IMU_COLUMNS)):
            f = open(os.path.join(self.path, f'{name}.csv'), 'w', newline='')
            w = csv.writer(f)
            w.writerow(cols)
            self._files[name] = f
            self._writers[name] = w
        self._events = open(os.path.join(self.path, 'events.jsonl'), 'w')
        self.write_meta()
        self.closed = False

    def write_meta(self):
        tmp = os.path.join(self.path, 'meta.json.tmp')
        with open(tmp, 'w') as f:
            json.dump(self.meta, f, indent=2, default=str)
        os.replace(tmp, os.path.join(self.path, 'meta.json'))

    def sample(self, row):
        self._writers['samples'].writerow([_fmt(row.get(c)) for c in SAMPLE_COLUMNS])

    def imu(self, source, row):
        self._writers[f'imu_{source}'].writerow([_fmt(v) for v in row])

    def event(self, t, kind, text, **fields):
        rec = {'t': round(t, 3), 'kind': kind, 'text': text}
        rec.update(fields)
        self._events.write(json.dumps(rec, default=str) + '\n')
        self._events.flush()

    def flush(self):
        for f in self._files.values():
            f.flush()

    def close(self, result, summary=None, table=None):
        if self.closed:
            return
        self.closed = True
        for f in self._files.values():
            f.close()
        self._events.close()
        self.meta['ended_wall'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
        self.meta['result'] = result
        self.write_meta()
        if summary is not None:
            with open(os.path.join(self.path, 'summary.json'), 'w') as f:
                json.dump(summary, f, indent=2)
        if table:
            with open(os.path.join(self.path, 'summary.txt'), 'w') as f:
                f.write(table + '\n')
        chown_like_parent(self.path)


def chown_like_parent(path):
    """Inside the container everything is written as root; hand it to the owner
    of the team workspace so the host user can read and delete it."""
    parent = os.path.dirname(os.path.dirname(os.path.abspath(path)))
    try:
        st = os.stat(parent)
    except OSError:
        return
    if os.geteuid() != 0 or st.st_uid == 0:
        return
    names = [os.path.dirname(os.path.abspath(path))]
    for root, dirs, files in os.walk(path):
        names += [root] + [os.path.join(root, n) for n in dirs + files]
    for name in names:
        try:
            os.chown(name, st.st_uid, st.st_gid)
        except OSError:
            pass
