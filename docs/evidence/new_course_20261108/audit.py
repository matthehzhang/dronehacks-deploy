"""Evaluate localization against recorded 5 Hz grader poses; not a full truth replay."""
import json
from pathlib import Path
import sys
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from std_msgs.msg import String

evidence, bag, localizer = map(Path, sys.argv[1:])
reader = rosbag2_py.SequentialReader()
reader.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id='mcap'),
            rosbag2_py.ConverterOptions('', ''))
reader.set_filter(rosbag2_py.StorageFilter(topics=['/grader/status']))
poses = []
while reader.has_next():
    _, data, _ = reader.read_next()
    row = json.loads(deserialize_message(data, String).data)
    if row.get('xyz') is not None:
        poses.append([row['t_sim'], *row['xyz']])
poses = np.asarray(poses)
assert len(poses), 'No recorded grader positions'
poses = poses[np.unique(poses[:, 0], return_index=True)[1]]
np.savez_compressed(evidence / 'grader_pose.npz', poses=poses)
score = json.loads((evidence / 'scorecard.json').read_text())
loc = np.genfromtxt(localizer, delimiter=',', names=True)
loc = loc[(loc['t'] >= score['t_start']) & (loc['t'] <= score['t_end'])]
errors = np.hypot(loc['x'] - np.interp(loc['t'], poses[:, 0], poses[:, 1]),
                  loc['y'] - np.interp(loc['t'], poses[:, 0], poses[:, 2]))
result = dict(localizer_xy_error_m=dict(samples=len(errors), median=float(np.median(errors)),
                                       p95=float(np.percentile(errors, 95)), maximum=float(errors.max())),
              method='Diagnostic only: 5 Hz grader positions rounded to 1 mm, interpolated at localizer capture stamps. Grader update timing adds uncertainty. No evaluation poses were inputs to policy.',
              full_clearance_replay_available=False,
              replay_limitation='Recorded /sim/truth_tf transforms have empty frame IDs, so model identity and orientation cannot be recovered authoritatively. Clearance is the live grader measurement in scorecard.json.')
(evidence / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
