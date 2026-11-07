"""Ordinary ROS launch arguments shared by hardware and simulation projects."""
import os
from pathlib import Path


def arguments(project, *, height_m, hover_s, ros_args=()):
    project = Path(project).resolve()
    args = ["refnav", "--ros-args", "--params-file", str(project / "flight.yaml")]
    deployment = Path(os.environ.get("DRONE_PARAMS_FILE", "/etc/flyx/mission.yaml"))
    if deployment.is_file():
        args += ["--params-file", str(deployment)]
    # ROS infers CLI parameter types from their text: Python's 4 must reach
    # our DOUBLE declarations as 4.0, just like a fractional height does.
    return args + ["-p", f"route_file:={project / 'route.json'}", "-p", f"height_m:={float(height_m)}",
                   "-p", f"hover_s:={float(hover_s)}", *ros_args]
