#!/usr/bin/env bash
# Enable on-device depth on a skybeta drone and verify the VIO input survived. Run ON THE DRONE,
# on the ground, with the team's agreement: it restarts the stack (10 s), which on an FC with
# STOCK PX4 v1.16.2 wedges TELEM2 until the FC is power-cycled (skybeta CLAUDE.md bug #15).
#
#   team_ws/tools/enable_depth.sh          # apply + verify
#   team_ws/tools/enable_depth.sh revert   # restore the organizers' file
set -euo pipefail
CFG=~/skybeta/src/oakd_vslam_bringup/config/oakd_vslam.yaml
NEW=~/team_ws/config/oakd_vslam_depth.yaml
BAK=~/team_ws/config/oakd_vslam.yaml.organizers
if [[ "${1:-}" == "revert" ]]; then
  [[ -f "$BAK" ]] && cp "$BAK" "$CFG" && echo "restored $CFG"
  docker rm -f skybeta-stack >/dev/null && echo "stack restarting"; exit 0
fi
[[ -f "$BAK" ]] || cp "$CFG" "$BAK"
cp "$NEW" "$CFG"
echo "applied $NEW -> $CFG (backup: $BAK); restarting the stack"
docker rm -f skybeta-stack >/dev/null
sleep 40
docker exec skybeta-stack bash -lc 'source skybeta_env.sh >/dev/null 2>&1
  echo "--- topics"; ros2 topic list | grep -E "^/oak/(left/image_raw|stereo/)$|^/oak/stereo"
  for t in /oak/left/image_raw /oak/right/image_raw /oak/stereo/image_raw /visual_slam/tracking/odometry; do
    printf "%-36s " "$t"; timeout 12 ros2 topic hz -w 30 "$t" 2>&1 | grep -m1 average || echo "no data"
  done
  echo "--- depth header"; timeout 8 ros2 topic echo /oak/stereo/image_raw --once --field header.frame_id 2>/dev/null
  timeout 8 ros2 topic echo /oak/stereo/image_raw --once --field encoding 2>/dev/null
  timeout 8 ros2 topic echo /oak/stereo/camera_info --once --field k 2>/dev/null'
echo "PASS condition: left/right ~30 Hz (ros2 topic hz under-reads by ~10 %), odometry ~30 Hz, stereo ~15 Hz, frame oak_left_camera_optical_frame, encoding 16UC1."
