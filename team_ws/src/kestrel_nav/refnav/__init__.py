"""refnav - the reference flight package: pure flight logic (core.py, depth.py) behind a thin
ROS 2 node (node.py). Runs unchanged against the simulator, recorded data and hardware; the
launch files in ../launch supply topics, clocks, calibration (TF) and, in the simulator only,
lockstep acknowledgements. Graders live in ../grader and only observe."""
