# Blocked-recovery ROS package

Build in a ROS 2 workspace with the Sky Beta and PX4 overlays sourced:

```bash
colcon build --packages-select kestrel_nav --symlink-install
source install/setup.bash
ros2 launch kestrel_nav refnav_skybeta.launch.py
```

A normal copied `colcon build` is also supported. The installed launch resolves
`share/kestrel_nav/runtime`, which contains the upstream module layout, flight
configuration, new-origin course OBJ/materials, occupancy map, and provenance.
Symlink installs continue to use the source tree. The recovery policy and its
localizer both resolve `course/course_new_origin.json` by default; the course
file points to the bundled new-origin map.

For Gazebo, add `use_sim_time:=true`. Hardware uses the default system clock.
The launch streams Sky Beta setpoints; the pilot remains responsible for arming.

`tests/test_package_install.py` builds a wheel, unpacks its installed payload
outside the checkout, resolves the actual launch runtime path, and plans the
new-origin course from those bundled files. It does not start ROS or arm.
