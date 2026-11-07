"""kestrel_nav: team wide-kestrel's course navigation for the skybeta stack.

Ported from the team workspace (github.com/AMMistry18/workspace): the published-OBJ course map,
the A* planner with chicane scoring, the scan-matching particle-filter localizer and their tests.
The old refnav controller (which drove /fmu/in/* itself) is NOT ported: on skybeta the vehicle is
flown only through /skybeta/setpoint/pose, and course_mission.py is the node that does that.
"""
