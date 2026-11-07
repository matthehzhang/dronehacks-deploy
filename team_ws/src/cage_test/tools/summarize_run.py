#!/usr/bin/env python3
"""Re-compute a run's per-segment summary from its CSVs. No ROS needed.

    python3 tools/summarize_run.py ~/team_ws/runs/20261107-153000_hover-steps
    python3 tools/summarize_run.py RUN_DIR --json       # the full numbers

Useful after copying runs to a laptop (``rsync -a dronehacks@<ip>:~/team_ws/runs .``),
or after changing ``cage_test/analysis.py``. Writes nothing.
"""
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cage_test import analysis  # noqa: E402
from cage_test.tune_plan import Segment  # noqa: E402


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def load(run_dir):
    with open(os.path.join(run_dir, 'meta.json')) as f:
        meta = json.load(f)
    segments = [Segment(**{k: tuple(v) if isinstance(v, list) else v for k, v in s.items()})
                for s in meta.get('segments', [])]
    with open(os.path.join(run_dir, 'samples.csv')) as f:
        samples = [{k: _num(v) for k, v in row.items()} for row in csv.DictReader(f)]
    for row in samples:
        row['seg'] = int(row['seg']) if row.get('seg') not in ('', None) else -1
    imu = {}
    for source in ('oak', 'px4'):
        path = os.path.join(run_dir, f'imu_{source}.csv')
        rows = []
        if os.path.exists(path):
            with open(path) as f:
                for r in csv.DictReader(f):
                    rows.append(tuple(float(r[k]) for k in ('t', 'ax', 'ay', 'az', 'gx', 'gy', 'gz')))
        imu[source] = rows
    return meta, segments, samples, imu


def main(argv):
    if not argv or argv[0] in ('-h', '--help'):
        print(__doc__)
        return 0
    meta, segments, samples, imu = load(argv[0])
    print(f'{argv[0]}\n  sequence {meta.get("sequence")}, result: {meta.get("result")}')
    if not segments:
        print('  no segments: the vehicle never entered Offboard in this run')
        return 0
    summary = analysis.summarize(segments, samples, imu['oak'], imu['px4'])
    if '--json' in argv:
        print(json.dumps(summary, indent=2))
    else:
        print(analysis.format_table(summary))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
