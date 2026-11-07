"""livenav: map-primed A* replanning on a live depth occupancy grid + a velocity carrot follower.

Pure Python (numpy/scipy/OpenCV). No ROS imports here: ``kestrel_nav.live_mission`` is the ROS node.

    navgrid     prior obstacle band from the published OBJ + live log-odds grid from depth scans,
                net fence, inflation by the vehicle radius -> blocked grid + clearance field
    replanner   ordered course goals (from coursenav.planner) -> A* path on the live blocked grid,
                replanned when the live grid blocks the path
    follower    carrot (pure-pursuit) velocity follower with corner, clearance and reactive speed
                scheduling; yaw follows the motion; never stops to turn in place next to a surface
    pad         landing-pad ring detector on a ground-plane rectified mono image
    mission     the state machine: WAIT -> CLIMB -> FLY -> PAD -> DESCEND -> DONE
    offline_sim closed-loop test with a kinematic vehicle and ray-cast depth from the course map
"""
