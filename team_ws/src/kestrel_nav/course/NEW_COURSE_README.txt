NEW COURSE ORIGIN MAP — TEMPORARY GROUP154 MAP POLICY

Open new_course_origin.obj with new_course_origin.mtl beside it. No external texture is needed.
Coordinate units: METRES. Axes: x east, y north, z up. Origin: enclosure floor centre.
This matches the CURRENT local topview_entrance simulation, with M1 against the west AND north perimeter.
The entrance placements are approximations from photos/partial LiDAR and team corrections, NOT a surveyed venue map.
Original downstream tunnel, chicane, landing area and course dimensions remain in the simulation.

Files:
  new_course_origin.obj / .mtl: complete portable collision model plus retained floor marks/net groups.
  course_origin_5cm.npz: legacy coursenav native 3-D surface voxel prior (occupied, labels, JSON meta).
  course_v3.json: legacy group154 course config. First semantic goal: tunnel. No obsolete maze_exit.
  occupancy_2d_080m.npz: planning diagnostic at body z 0.80 m, vertical band +/-0.25 m.
  course_geometry.json / scenario.json: source sim geometry and approximate-layout limitations.
  top_down.png: physical obstacle band and inflated planning exclusion, with semantic goals.
  manifest.json: exact source hashes, replaced groups, baseline commit and clearance requirement.

The native NPZ surface map is indexed [x, y, z]; origin/resolution/groups are in its JSON meta.
The 2-D diagnostic is indexed [x, y]; transpose for image display (origin lower).
Planner uses the same physical vehicle reach and 0.12 m margin; no safety/grader/autopilot setting was reduced.
This is an explicitly hardcoded/map-based temporary solution, separate from the mapless exploration strategy.
The source geometry still contains historical scoring references for the first two checkpoints; these are unqualified.
It is NOT an official updated course export and does not establish flight or competition success.


RECORDED LOCAL SIM TEST: stack_20261108T063119Z
The temporary map policy physically traversed the entrance, tunnel and window,
reached the destination pad, landed and confirmed PX4 disarm. Elapsed graded
time was 78.729154 s, with 0 obstacle touches and 0 near misses. The official
grader FAILED with 0 points: landed_on_pad_before_checkpoint_4. Its preserved
historical entrance scoring lines were not crossed by the revised course route.
This is NOT a competition pass or a speed comparison against full run022.

SAFETY LIMITATION: minimum physical outline clearance was 0.0814 m, BELOW the
unchanged 0.12 m planning margin. The minimum occurred during initial climb,
next to Barrier 1, after southward takeoff drift. Thus 0 contacts does not
qualify the policy as safe. PF XY error vs truth was 0.1666 m at p95 and
0.1855 m maximum; the origin map alone does not solve estimation/tracking risk.
No vehicle footprint, safety limit, autopilot or grader threshold was reduced.
See sim_test_result.json for measured details and diagnostic qualifications.
Ten focused map/interface tests passed with no deselection.
