#!/usr/bin/env bash
# port.sh <workspace-ref> <deploy-branch> [base-branch]
#
# Vendor a FlyX-lab workspace branch (refnav + coursenav + course + maps + fly_course.py) UNCHANGED into the
# deploy repo's kestrel_nav package, apply the skybeta adaptations (assets/skybeta.patch + assets/new/*), wire
# setup.py, and run the offline checks. Does NOT commit or push: review, then commit/push yourself.
#
#   workspace-ref   branch or commit in ~/projects/hackathon/dronehacks/workspace (e.g. origin/group119)
#   deploy-branch   branch in ~/projects/hackathon/dronehacks/deploy to create or update
#   base-branch     deploy branch to start a NEW deploy-branch from (default: kestral_nav934am)
set -euo pipefail
REF="${1:?workspace ref}"; TARGET="${2:?deploy branch}"; BASE="${3:-kestral_nav934am}"
ROOT="$HOME/projects/hackathon/dronehacks"
WS="$ROOT/workspace"; DEP="${DEPLOY_DIR:-$ROOT/deploy}"; PKG="$DEP/team_ws/src/kestrel_nav"   # DEPLOY_DIR: a git worktree of deploy
SK="$(cd "$(dirname "$0")/.." && pwd)"
VENDORED=(refnav coursenav course maps fly_course.py flight.yaml onboard/__init__.py onboard/vio/__init__.py
          onboard/vio/vio_health.py onboard/config/common.yaml)

git -C "$WS" fetch -q origin || true
COMMIT="$(git -C "$WS" rev-parse --verify "$REF^{commit}")"
echo "[port] workspace $REF = $COMMIT"

if [ -n "$(git -C "$DEP" status --porcelain --untracked-files=no)" ]; then
  echo "[port] deploy has uncommitted tracked changes; commit or stash them first" >&2; exit 1
fi
if git -C "$DEP" rev-parse --verify -q "$TARGET" >/dev/null; then
  git -C "$DEP" switch -q "$TARGET"
else
  git -C "$DEP" switch -q -c "$TARGET" "$BASE"
fi
echo "[port] deploy branch $TARGET ($(git -C "$DEP" log --oneline -1))"

# 1. vendor UNCHANGED
( cd "$PKG" && rm -rf refnav coursenav course maps onboard fly_course.py flight.yaml )
git -C "$WS" archive "$COMMIT" "${VENDORED[@]}" | tar -x -C "$PKG"
echo "$COMMIT  $REF  (vendored by port-to-skybeta $(date -Iseconds))" > "$PKG/VENDORED_FROM.txt"

# 2. skybeta adaptations to vendored files
if ! ( cd "$PKG" && patch -p1 --forward --no-backup-if-mismatch -r - < "$SK/assets/skybeta.patch" ); then
  echo "[port] PATCH DID NOT APPLY CLEANLY: upstream changed one of the adapted files." >&2
  echo "[port] Re-apply the failed hunks by hand (SKILL.md, 'Adaptations'), then run scripts/refresh_patch.sh." >&2
  exit 2
fi

# 3. skybeta-only files
cp -a "$SK/assets/new/." "$PKG/"

# 4. setup.py: vendored packages, fly_course module, health entry point (idempotent)
python3 - "$PKG/setup.py" <<'EOF'
import re, sys
f = sys.argv[1]; s = open(f).read()
if "'refnav'" not in s:
    s = re.sub(r"(packages=\[[^\]]*)\]", r"\1, 'refnav', 'coursenav', 'onboard', 'onboard.vio']", s, count=1)
if "py_modules" not in s:
    s = s.replace("    data_files=[", "    py_modules=['fly_course'],\n    data_files=[", 1)
if "skybeta_health" not in s:
    s = s.replace("'console_scripts': [", "'console_scripts': [\n            'skybeta_health = refnav.skybeta_health:main',", 1)
open(f, "w").write(s)
EOF

# 5. offline checks (no ROS needed)
cd "$PKG"
echo "[port] dry run from the launch mark:"
timeout 600 python3 fly_course.py --dry-run --skip-preflight --event-start --start-from-mark --landed-z 0.10 2>&1 \
  | grep -E "start |legs,|navigation|planning from|Error|error" || true
rm -rf logs
echo "[port] adapter tests:"
PYTHONPATH=. timeout 600 python3 -m pytest -q -p no:cacheprovider tests/test_refnav_skybeta.py 2>&1 | tail -2
cat <<EOF

[port] done. Next (see SKILL.md):
  1. review:  git -C $DEP diff --stat
  2. full suite (laptop, <=4 procs):  cd $PKG && PYTHONPATH=. nice -n 10 python3 -m pytest -q tests
  3. commit + push $TARGET (only this branch)
  4. Gazebo:  cd $ROOT/sim && scripts/policy.sh $TARGET kestrel_nav refnav_skybeta.launch.py use_sim_time:=true
EOF
