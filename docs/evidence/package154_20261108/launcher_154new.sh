#!/usr/bin/env bash
# Laptop: ./drone_tools/154new.sh [start|restart|watch|check|status]
# Jetson: ~/team_ws/tools/154new.sh [start|restart|watch|check|status]
# Installs this launcher when run from the laptop. Never arms the drone.
set -euo pipefail
ACTION=${1:-start}
case "$ACTION" in
  start|restart|watch|check|status) ;;
  -h|--help)
    echo "Usage: $0 [start|restart|watch|check|status]"
    echo 'start: start the installed package, or watch it if already running'
    echo 'restart: restart only the mission after a fresh disarmed/landed check'
    echo 'watch: read-only tmux view; Ctrl-b then d detaches'
    echo 'check: verify fresh disarmed/landed PX4 reports'
    echo 'status: show installed version, mission process and recorder status'
    exit 0 ;;
  *) echo "Unknown action: $ACTION. Use --help." >&2; exit 2 ;;
esac
[[ $# -le 1 ]] || { echo 'Expected at most one action.' >&2; exit 2; }

machine=$(hostname)
if [[ "$machine" != skybeta-14 ]]; then
  [[ "$machine" != skybeta-* ]] || { echo "Wrong drone: $machine" >&2; exit 1; }
  remote=(ssh -o BatchMode=yes -o ConnectTimeout=8 dronehacks)
  [[ "$("${remote[@]}" hostname)" == skybeta-14 ]] || {
    echo 'Cannot reach skybeta-14. Check its power and USB-C/Wi-Fi connection.' >&2
    exit 1
  }
  self=$(realpath -- "${BASH_SOURCE[0]}")
  "${remote[@]}" 'set -e; mkdir -p ~/team_ws/tools; t=$(mktemp ~/team_ws/tools/.154new.XXXXXX); trap '\''rm -f "$t"'\'' EXIT; cat > "$t"; chmod 755 "$t"; mv "$t" ~/team_ws/tools/154new.sh' < "$self"
  if [[ -t 0 && -t 1 && ( "$ACTION" == start || "$ACTION" == restart || "$ACTION" == watch ) ]]; then
    exec "${remote[@]}" -t "~/team_ws/tools/154new.sh $ACTION"
  fi
  exec "${remote[@]}" "~/team_ws/tools/154new.sh $ACTION"
fi

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SOURCE="$HOME/team_ws/src/kestrel_nav"
RECORDER="$HERE/group154_record.sh"
MISSION='^python3 /root/team_ws/src/kestrel_nav/fly_course.py '
STACK=skybeta-stack

verify_package() {
  grep -qx 'branch=package_154_new' "$SOURCE/INSTALLED_FROM.txt" || {
    echo 'The active package is not marked package_154_new; nothing started.' >&2; exit 1;
  }
  grep -qx 'COMPETITOR_PKG=kestrel_nav' /etc/skybeta/drone.env
  grep -qx 'COMPETITOR_LAUNCH=refnav_skybeta.launch.py' /etc/skybeta/drone.env
  docker exec "$STACK" tmux has-session -t skybeta
}
running() { docker exec "$STACK" pgrep -f "$MISSION" >/dev/null; }
watch_program() {
  [[ "${PACKAGE154_NO_WATCH:-0}" != 1 ]] || return 0
  echo 'Watching 154new. Historical logs are not a current readiness check.'
  if [[ -t 0 && -t 1 ]]; then
    echo 'Ctrl-b then d detaches; the mission keeps running.'
    exec docker exec -it "$STACK" tmux attach-session -r -t skybeta
  fi
  docker exec "$STACK" tmux capture-pane -p -J -t skybeta:competitor -S -40
}
check_ground() {
  docker exec -i "$STACK" bash -c 'source skybeta_env.sh >/dev/null; python3 -' <<'PY'
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from px4_msgs.msg import VehicleStatus, VehicleLandDetected

rclpy.init()
node = Node('package154_new_ground_check')
qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                 durability=DurabilityPolicy.VOLATILE)
seen = {}
def receive(key, msg):
    seen[key] = (msg, time.monotonic())
def topic(base, cls):
    version = getattr(cls, 'MESSAGE_VERSION', 0)
    return base + (f'_v{version}' if version else '')
node.create_subscription(VehicleStatus, topic('/fmu/out/vehicle_status', VehicleStatus),
                         lambda m: receive('status', m), qos)
node.create_subscription(VehicleLandDetected, topic('/fmu/out/vehicle_land_detected', VehicleLandDetected),
                         lambda m: receive('land', m), qos)
ok = False
deadline = time.monotonic() + 12
try:
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        now = time.monotonic()
        if all(k in seen and now - seen[k][1] < 1 for k in ('status', 'land')):
            status, land = seen['status'][0], seen['land'][0]
            ok = status.arming_state == VehicleStatus.ARMING_STATE_DISARMED and land.landed
            print(f'Fresh PX4 report: arming_state={status.arming_state}, landed={land.landed}', flush=True)
            break
finally:
    node.destroy_node()
    rclpy.shutdown()
if not ok:
    raise SystemExit('Nothing restarted: fresh disarmed AND landed reports are required. Check the PX4 connection.')
PY
}

case "$ACTION" in
  watch) watch_program; exit ;;
  status)
    cat "$SOURCE/INSTALLED_FROM.txt"
    if running; then echo 'Mission process running (not a flight-readiness check).'; else echo 'Mission not running.'; fi
    "$RECORDER" status
    exit ;;
  check) check_ground; exit ;;
esac

verify_package
[[ -x "$RECORDER" && -f "$HERE/group154_record.py" ]] || {
  echo 'Recorder helpers are missing; nothing restarted.' >&2; exit 1;
}
exec 9>"$HOME/team_ws/.154new-launch.lock"
flock -n 9 || { echo 'Another 154new start/restart is in progress.' >&2; exit 1; }
if [[ "$ACTION" == start ]] && running; then
  echo '154new is already running; attaching without restarting it.'
  "$RECORDER" start
  flock -u 9
  watch_program
  exit
fi

check_ground
echo 'Restarting only the mission. Keep the drone still on its launch mark.'
"$RECORDER" stop
docker exec "$STACK" tmux send-keys -t skybeta:competitor C-c
old_processes='ros2 launch kestrel_nav|python3 -m coursenav|python3 -m refnav|/kestrel_nav/fly_course.py|/kestrel_nav/host_depth'
for ((attempt=0; attempt<20; attempt++)); do
  docker exec "$STACK" pgrep -f "$old_processes" >/dev/null || break
  sleep 1
done
if docker exec "$STACK" pgrep -af "$old_processes"; then
  echo 'Old mission did not stop; refusing to start a second copy.' >&2; exit 1
fi
# Build and start through the same entrypoint used at boot. The agent, camera,
# SLAM, bridge, organiser monitoring and container itself stay running.
docker exec "$STACK" tmux respawn-pane -k -t skybeta:competitor run_competitor.sh
for ((attempt=0; attempt<90; attempt++)); do
  if running; then
    "$RECORDER" start || {
      echo 'Mission started but recording failed. Inspect recorder status before a run.' >&2; exit 1;
    }
    echo '154new started. Wait for live mission readiness; the pilot still arms on RC.'
    flock -u 9
    watch_program
    exit
  fi
  sleep 1
done
echo 'Mission did not start within 90 seconds. Inspect the competitor pane:' >&2
docker exec "$STACK" tmux capture-pane -p -J -t skybeta:competitor -S -60
exit 1
