#!/usr/bin/env bash
# Automatic telemetry recording for group154, isolated from the flight container.
set -euo pipefail
NAME=group154-recorder
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WS=$(cd -- "$HERE/.." && pwd)
running() { [[ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null || true)" == true ]]; }
case "${1:-start}" in
  start)
    if running; then
      echo 'group154 rosbag is already recording.'
      exit 0
    fi
    docker inspect skybeta-stack >/dev/null
    image=$(docker inspect -f '{{.Config.Image}}' skybeta-stack)
    domain=$(sed -n 's/^ROS_DOMAIN_ID=//p' /etc/skybeta/drone.env)
    [[ "$domain" =~ ^[0-9]+$ ]]
    mkdir -p "$WS/flight_logs"
    out=$(mktemp -d "$WS/flight_logs/$(date +%Y%m%d-%H%M%S)_group154_XXXXXX")
    cp /etc/skybeta/drone.env "$out/"
    cp "$WS/src/kestrel_nav/course/course_v3.json" "$out/"
    sha256sum "$WS/src/kestrel_nav/refnav/core.py" "$WS/src/kestrel_nav/refnav/gate.py" \
      "$WS/src/kestrel_nav/maps/course_origin_5cm.npz" > "$out/source.sha256"
    docker run -d --rm --init --name "$NAME" --stop-signal SIGINT --stop-timeout 25 \
      --network host --ipc host --log-opt max-size=1m --log-opt max-file=1 \
      -e ROS_DOMAIN_ID="$domain" -e SKYBETA_DDS_SCOPE=local \
      -e FASTRTPS_DEFAULT_PROFILES_FILE=/etc/skybeta/fastdds.xml \
      -e FASTDDS_DEFAULT_PROFILES_FILE=/etc/skybeta/fastdds.xml \
      -v "$HOME/skybeta/host/dds/fastdds_localhost.xml:/etc/skybeta/fastdds.xml:ro" \
      -v "$WS:/root/team_ws:ro" -v "$out:/out" "$image" \
      bash -c 'source skybeta_env.sh >/dev/null; exec python3 -u /root/team_ws/tools/group154_record.py' >/dev/null
    for ((attempt=0; attempt<30; attempt++)); do
      if [[ -f "$out/recording.ready" ]] && running; then
        echo "Recording rosbag: $out/bag"
        echo "$out" > "$WS/flight_logs/group154_latest.txt"
        exit 0
      fi
      running || break
      sleep 1
    done
    echo "Recorder failed to become ready; inspect $out/rosbag.log" >&2
    docker stop "$NAME" >/dev/null 2>&1 || true
    exit 1
    ;;
  stop)
    if running; then docker stop -t 25 "$NAME" >/dev/null; fi
    echo 'Rosbag stopped cleanly; the drone program is unchanged.'
    ;;
  status)
    if running; then echo 'Recording'; else echo 'Not recording'; fi
    cat "$WS/flight_logs/group154_latest.txt" 2>/dev/null || true
    ;;
  *) echo "Usage: $0 [start|stop|status]" >&2; exit 2 ;;
esac
