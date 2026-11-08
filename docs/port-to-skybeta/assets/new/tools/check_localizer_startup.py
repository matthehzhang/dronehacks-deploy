#!/usr/bin/env python3
"""Native ROS startup check in isolated DDS domain 77; no vehicle commands.

Run in the sourced Sky Beta environment. Instantiates the complete localizer,
checks parameter override propagation, spins startup timers, then destroys it.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fit', type=float, default=.42)
    parser.add_argument('--seconds', type=float, default=1.2)
    parser.add_argument('--domain', type=int, default=77)
    args = parser.parse_args()
    os.environ['ROS_DOMAIN_ID'] = str(args.domain)
    os.environ['ROS_LOCALHOST_ONLY'] = '1'
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import rclpy
    from coursenav.loc_node import LocNode
    from coursenav.course import default_course_path
    with tempfile.TemporaryDirectory(prefix='localizer-startup-') as logs:
        rclpy.init(args=['--ros-args', '-p', f'min_fit_frac:={args.fit}',
                         '-p', f'log_dir:={logs}', '-p', 'height_source:=local_position',
                         '-p', 'use_sim_time:=false'])
        node = None
        try:
            node = LocNode()
            assert node.pf.p.min_fit_frac == args.fit
            assert node.prior_mon.p.restore_fit == args.fit
            assert node.get_parameter('min_fit_frac').value == args.fit
            deadline = time.monotonic() + args.seconds
            while time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=.1)
            print(json.dumps(dict(result='PASS', fit=args.fit, restore_fit=node.prior_mon.p.restore_fit,
                                  course=str(default_course_path()), domain=args.domain,
                                  live_map=node.live is not None, local_map=node.local is not None)))
        finally:
            if node is not None:
                node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
