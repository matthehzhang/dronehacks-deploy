# New origin map (the stack's default course)

Source: `new_course_origin_map.zip` (team, 2026-11-07, scenario `topview_entrance`, baseline `f3825b9`).
It reconstructs the real entrance from the team's top view, photos and partial LiDAR. **Not a surveyed venue
map and not an official course export** (see `_map_provenance` in the course file).

## What changed in the course
- **Maze dividers re-laid.** M1 runs along the north wall from the west wall (no passage behind it). M2 is the
  angled panel north-east of the start. M3 is the angled panel leading to the tunnel. No central upright wall.
- **First goal is the tunnel.** There's no `maze_exit` goal (the entrance was re-laid). Tunnel, chicane window,
  landing pad and enclosure are unchanged from course v3.
- **The drone starts on the launch mark** (`sim_profile.fixture_offset_m` [0, 0]). That matches the real-drone
  bag 20261107-201744, whose localizer started at (−2.518, 0.91). The old training spawn was 0.2 m north.
- **Start pose is typed in, not measured** (−2.515, 0.914, 90°). The new OBJ has no `Guideline` group to read the
  heading from.

## How it's wired

| Piece | File |
|---|---|
| OBJ (+ .mtl) | `course/new_course_origin.obj` |
| Course file | `course/course_new_origin.json`: course v3's settings (refnav, navigation, leg policy, vehicle, `blocked_max_per_leg`) with the new `obj`, `map`, `start`, `sequence`, fixture and provenance |
| Voxel map | `maps/course_new_origin.npz`, built by `tools/build_map.py --course course/course_new_origin.json`. Identical to the zip's map (100,004 voxels), but marked as the published-OBJ build, so the planner uses **exact surface clearance**. On the 5 cm raster the 0.914 m window reads 0.412 m and is refused. |
| Default course | `coursenav/course.py` `DEFAULT_COURSE`. `load_course()`, `fly_course.py`, `loc_node`, `nav_node` and the tools all load it. Override with `COURSENAV_COURSE=course/course_v3.json`. |
| Tests | `tests/conftest.py` pins course v3 for the regression tests (they assert v3 geometry). `tests/test_new_origin_course.py` covers the new course. |

## Planner change: the launch mark is inside the planning buffer
The launch mark has **0.41 m of clearance** (exact; 0.400 m on the raster) against the **0.414 m requirement**
(7-inch reach 0.294 + 0.12 margin):

| Nearest surface | Distance from the launch mark |
|---|---|
| Barrier 1 (south) | 0.39 m |
| M2 maze divider | 0.45 m |
| Net side west | 0.51 m |

- The waypoint planner refused it ("start is blocked"). `planner.start_escape` now adds a first step
  (`via_start_escape`, at most `leg_policy.start_escape_m`, default 0.15 m) to the nearest plannable cell. The
  step never comes closer to anything than the launch point itself. Here it's 6 cm north.
- Path mode drops `via…` points. The navigator already steps out of the buffer itself (`escape_m` 0.6), and now
  **creeps** out at `escape_creep_mps` (0.1 m/s). The speed profile is 0 inside the buffer, and no executor can
  fly that (the depth-silence check refused it).

## Results

**Plan** (`python3 tools/plan_route.py`): window route, 14.65 m, estimated 81 s, required clearance everywhere
after the escape step, refnav's blocking rule replayed: no false blocks, depth-silence check: OK at every
stop. Gate B: no plan (only reachable through its own opening).

**Offline flight** (`tools/fly_offline.py --mode corrected --nav path --course-nav --px4 dynamic`, seeds 0-2):
3/3 complete, legs 5/5, no collision, minimum clearance 0.375-0.387 m, landed 0.14-0.17 m from the pad centre,
localizer error p95 0.071-0.081 m.

## Cautions
1. **Scoring:** the team's own sim run of this map reached the pad but scored **0 points**
   (`landed_on_pad_before_checkpoint_4`). The grader's entrance checkpoint lines are from the old layout, and the
   new route doesn't cross them. Find out the venue's real checkpoints before relying on this for score.
2. **Takeoff is tight.** Barrier 1 is 0.39 m south of the launch mark, which leaves 0.10 m beyond the prop reach
   before any drift. The team's sim had **0.081 m** minimum clearance during the climb, after southward takeoff
   drift (below the 0.12 m margin). Placement on the mark and a clean vertical climb matter. The escape step
   only starts after the climb.
3. **Approximate geometry.** Panel placements come from photos and partial LiDAR (the LiDAR registration was
   rejected). Expect the localizer's fit to be lower near the entrance than on the retained course parts. Check
   `/coursenav/prior_state` and `scan_fit` on the real course.
4. **It doesn't explain the real-drone bag.** From the launch mark, the forward view is the same in the old and
   new maps (each scores 100 % against the other). Bag 20261107-201744's 10-20 % fit is still unexplained by
   layout: placement, heading or depth, see `docs/LOCAL_MAP.md`.
5. Going back to course v3: `export COURSENAV_COURSE=course/course_v3.json`.
