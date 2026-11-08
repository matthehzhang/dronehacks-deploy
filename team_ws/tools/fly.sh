#!/usr/bin/env bash
# Fly a test on drone 14:   ~/team_ws/tools/fly.sh hop | tune | group119 | group154
# Points the stack at it, restarts the stack, then shows the mission's own prompts live
# ("NOT ready ..." -> "ready: arm (switch on Altitude), then switch to Position" -> "Offboard active" ...).
# Ctrl-C only stops watching; the mission keeps running. Drone on the ground, disarmed, still, before you run it.
set -uo pipefail
ENV=/etc/skybeta/drone.env; SRC=~/team_ws/src; POL=~/policies
case "${1:-}" in
  hop)      PKG=cage_test   LAUNCH=hop_test.launch.py ;;
  tune)     PKG=cage_test   LAUNCH=tune_test.launch.py ;;
  group119) PKG=kestrel_nav LAUNCH=group119.launch.py       POLICY=group119 ;;
  group154) PKG=kestrel_nav LAUNCH=refnav_skybeta.launch.py POLICY=group154 ;;
  *) sed -n 2,5p "$0"; exit 2 ;;
esac

# Never restart under an armed drone: that would kill the flight interface mid-flight. If the stack is up,
# a disarmed report is required (fails closed); FORCE=1 overrides when the state topic cannot be read.
if docker ps --format '{{.Names}}' | grep -qx skybeta-stack && [ "${FORCE:-0}" != 1 ]; then
  st=$(docker exec skybeta-stack bash -lc \
       'source skybeta_env.sh >/dev/null 2>&1; timeout 10 ros2 topic echo --no-daemon --spin-time 2 --once --full-length /skybeta/state std_msgs/msg/String' 2>/dev/null)
  if printf '%s' "$st" | grep -q '"armed": true'; then echo "ARMED: not restarting. Land and disarm first."; exit 1; fi
  printf '%s' "$st" | grep -q '"armed": false' \
    || { echo "could not read /skybeta/state. If the drone is disarmed: FORCE=1 $0 $1"; exit 1; }
fi

sudo -v || exit 1                      # ask for the password before changing anything

# group119/group154 live in ~/policies; put the chosen one in team_ws (old copy kept in ~/policies/_replaced).
if [ -n "${POLICY:-}" ] && ! diff -rq -x __pycache__ -x '*.pyc' "$POL/$POLICY/kestrel_nav" "$SRC/kestrel_nav" >/dev/null 2>&1; then
  B="$POL/_replaced/$(date +%Y%m%d-%H%M%S)"; mkdir -p "$B"
  [ -d "$SRC/kestrel_nav" ] && rsync -a "$SRC/kestrel_nav" "$B/"
  rsync -a --delete --exclude __pycache__ --exclude '*.pyc' "$POL/$POLICY/kestrel_nav/" "$SRC/kestrel_nav/" \
    || { echo "copying $POLICY failed; nothing restarted"; exit 1; }
  echo "== installed $POLICY (previous kestrel_nav saved in $B)"
fi

sudo sed -i.bak-fly -e "s|^COMPETITOR_PKG=.*|COMPETITOR_PKG=$PKG|" -e "s|^COMPETITOR_LAUNCH=.*|COMPETITOR_LAUNCH=$LAUNCH|" "$ENV"
grep -qx "COMPETITOR_PKG=$PKG" "$ENV" && grep -qx "COMPETITOR_LAUNCH=$LAUNCH" "$ENV" \
  || { echo "drone.env was not updated; nothing restarted"; exit 1; }
echo "== $1: restarting the stack (about a minute to ready). Ctrl-C stops watching, not the mission."
docker rm -f skybeta-stack >/dev/null 2>&1
sleep 5

# Live view of the competitor window: new lines only, without the interface's clamp warnings.
last=""
while :; do
  out=$(docker exec skybeta-stack tmux capture-pane -p -J -t skybeta:competitor -S -5000 2>/dev/null \
        | sed 's/ *$//' | grep -v '^$') || { sleep 2; continue; }
  new=$(printf '%s\n' "$out" | LAST="$last" awk '
          { line[NR] = $0; if ($0 == ENVIRON["LAST"]) at = NR }
          END { from = (ENVIRON["LAST"] == "" ? 1 : (at ? at + 1 : NR - 19)); if (from < 1) from = 1
                for (i = from; i <= NR; i++) print line[i] }')
  [ -n "$new" ] && printf '%s\n' "$new" | grep -v 'setpoint clamped'
  last=$(printf '%s\n' "$out" | tail -1)
  printf '%s\n' "$new" | grep -q 'team_ws build failed' && { echo "== BUILD FAILED: nothing will fly"; exit 1; }
  sleep 1
done
