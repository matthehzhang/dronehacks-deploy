#!/usr/bin/env bash
# refresh_patch.sh [deploy-branch]
#
# After a new skybeta fix lands in the deploy branch, capture it back into the skill: regenerate
# assets/skybeta.patch (deploy's vendored files vs the pristine workspace commit in VENDORED_FROM.txt)
# and copy the skybeta-only files into assets/new/. Then update SKILL.md's "Adaptations" table.
set -euo pipefail
ROOT="$HOME/projects/hackathon/dronehacks"; WS="$ROOT/workspace"; DEP="${DEPLOY_DIR:-$ROOT/deploy}"
PKG="$DEP/team_ws/src/kestrel_nav"; SK="$(cd "$(dirname "$0")/.." && pwd)"
[ -n "${1:-}" ] && git -C "$DEP" switch -q "$1"
COMMIT="$(awk '{print $1; exit}' "$PKG/VENDORED_FROM.txt" 2>/dev/null || true)"
COMMIT="${COMMIT:-749bca0cd50cc673ed4a0f431e9bd63b9a6fbef3}"      # group119, the first port
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/a" "$TMP/b"
git -C "$WS" archive "$COMMIT" refnav coursenav course maps fly_course.py flight.yaml onboard/config/common.yaml \
  onboard/vio/vio_health.py | tar -x -C "$TMP/a"
( cd "$TMP/a" && find . -type f -not -path '*/__pycache__/*' ) | while read -r f; do
  f="${f#./}"
  if ! cmp -s "$TMP/a/$f" "$PKG/$f"; then
    mkdir -p "$TMP/b/$(dirname "$f")"; cp "$PKG/$f" "$TMP/b/$f"
    mkdir -p "$TMP/a2/$(dirname "$f")"; cp "$TMP/a/$f" "$TMP/a2/$f"
    echo "adapted: $f"
  fi
done
( cd "$TMP" && diff -ruN a2 b | sed -e 's|^--- a2/|--- a/|' -e 's|^+++ b/|+++ b/|' ) > "$SK/assets/skybeta.patch" || true
cp "$PKG/refnav/skybeta.py" "$PKG/refnav/skybeta_health.py" "$SK/assets/new/refnav/"
L="$PKG/launch/refnav_skybeta.launch.py"; [ -f "$L" ] || L="$PKG/launch/group119.launch.py"
cp "$L" "$SK/assets/new/launch/refnav_skybeta.launch.py"
sed -i "s/group119\.launch\.py/refnav_skybeta.launch.py/g" "$SK/assets/new/launch/refnav_skybeta.launch.py"
T="$PKG/tests/test_refnav_skybeta.py"; [ -f "$T" ] || T="$PKG/tests/test_group119_skybeta.py"
cp "$T" "$SK/assets/new/tests/test_refnav_skybeta.py"
for f in coursenav/observed_free.py tests/test_observed_recovery.py tests/test_nav_recovery_protocol.py \
         tests/test_recovery_controller.py tests/test_priorstate.py tests/test_local_free_age.py tests/test_package_install.py \
         tests/test_no_route_recovery.py tests/test_course_clearing_masks.py tests/test_free_grid_reader.py \
         tests/test_prior_correction.py tests/test_prior_correction_node.py tests/test_local_ray_horizon.py \
         tests/test_recovery_creep.py; do
  mkdir -p "$SK/assets/new/$(dirname "$f")"
  cp "$PKG/$f" "$SK/assets/new/$f"
done
echo "patch: $(wc -l < "$SK/assets/skybeta.patch") lines vs $COMMIT. Now update SKILL.md 'Adaptations'."
