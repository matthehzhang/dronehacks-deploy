#!/usr/bin/env bash
# Full group154 program on the drone. No arm command; the pilot starts the flight.
# Usage: group154.sh [start|watch|check]
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

check_ground() {
  docker exec -i skybeta-stack bash -c 'source skybeta_env.sh >/dev/null; python3 -' <<'PY'
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from px4_msgs.msg import VehicleStatus, VehicleLandDetected

rclpy.init()
node = Node('group154_ground_check')
qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                 durability=DurabilityPolicy.VOLATILE)
seen = {}
def receive(key, msg):
    seen[key] = (msg, time.monotonic())
def topic(base, cls):
    version = getattr(cls, 'MESSAGE_VERSION', 0)
    return base + (f'_v{version}' if version else '')
node.create_subscription(VehicleStatus, topic('/fmu/out/vehicle_status', VehicleStatus),
                         lambda msg: receive('status', msg), qos)
node.create_subscription(VehicleLandDetected, topic('/fmu/out/vehicle_land_detected', VehicleLandDetected),
                         lambda msg: receive('land', msg), qos)
deadline = time.monotonic() + 12
ok = False
try:
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        now = time.monotonic()
        if all(k in seen and now - seen[k][1] < 1 for k in ('status', 'land')):
            status, land = seen['status'][0], seen['land'][0]
            ok = status.arming_state == VehicleStatus.ARMING_STATE_DISARMED and land.landed
            print(f'Fresh PX4 state: arming_state={status.arming_state}, landed={land.landed}', flush=True)
            break
finally:
    node.destroy_node()
    rclpy.shutdown()
if not ok:
    raise SystemExit('Refusing restart: fresh disarmed AND landed telemetry is required.')
PY
}

watch_program() {
  echo 'Watching group154 read-only. Ctrl-b then d detaches; the mission keeps running.'
  echo 'Wait for current readiness before the pilot arms and switches to Position.'
  exec docker exec -it skybeta-stack tmux attach-session -r -t skybeta
}

case "${1:-start}" in
  check) check_ground ;;
  watch) watch_program ;;
  start)
    env_file=/etc/skybeta/drone.env
    grep -qx 'COMPETITOR_PKG=kestrel_nav' "$env_file"
    grep -qx 'COMPETITOR_LAUNCH=refnav_skybeta.launch.py' "$env_file"
    check_ground
    "$HERE/group154_record.sh" stop
    echo 'Restarting the full group154 stack. Keep the drone still on the launch mark.'
    sudo -n systemctl restart skybeta-stack.service
    for ((attempt=0; attempt<120; attempt++)); do
      if docker exec skybeta-stack tmux has-session -t skybeta 2>/dev/null; then
        "$HERE/group154_record.sh" start
        watch_program
      fi
      sleep 1
    done
    echo 'Stack did not start within 120 seconds; inspect: journalctl -u skybeta-stack.service' >&2
    exit 1
    ;;
  *) echo "Usage: $0 [start|watch|check]" >&2; exit 2 ;;
esac
