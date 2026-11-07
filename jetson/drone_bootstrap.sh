#!/usr/bin/env bash
# skybeta-fleet: turn a freshly flashed Jetson (JetPack 6 / L4T R36) into fleet drone N.
#
# Runs ON THE JETSON, as the login user (not root; it asks for sudo once). Normally you
# do not run it yourself: provision/new_drone.sh copies it over and drives it. Every
# step is idempotent and records itself, so re-running resumes where it stopped.
#
#   drone_bootstrap.sh --drone-id 3 [--team NAME] [--camera "x y z roll pitch yaw"]
#                      [--auth-key-file PATH] [--ref BRANCH_OR_TAG] [--repo-dir DIR]
#                      [--registry REG] [--mirror HOST:PORT] [--insecure-registry HOST:PORT]
#                      [--registry-login USER --registry-token-file PATH] [--no-reboot] [--dry-run]
#
# Steps:
#   hostname    skybeta-NN (hostnamectl + /etc/hosts)
#   tailscale   install, then join as tag:drone with --ssh. The auth key is read
#               from --auth-key-file (tmpfs, written by new_drone.sh) and deleted
#               immediately after, success or failure.
#   repo        clone the PUBLIC skybeta repo at --ref into --repo-dir
#   identity    /etc/skybeta/drone.env from the repo template: DRONE_ID, ROS_DOMAIN_ID
#               (= DRONE_ID), TEAM, camera pose. An existing different file is kept.
#   provision   skybeta host/provision.sh in two halves around the registry step:
#               prereqs + UART patch + power first, then pull + doctor.
#               The UART patch needs one reboot: this script reboots and exits 10;
#               new_drone.sh waits for the Jetson and runs it again to finish.
#   registry    (between the halves, before any reboot) --insecure-registry adds a plain-
#               HTTP registry (e.g. registry:2 on the LAN) to /etc/docker/daemon.json;
#               --registry-login runs `docker login` with a token read from a tmpfs file
#               (new_drone.sh streams it there) and deletes the file. Use a READ-ONLY pull
#               token: it is stored in ~/.docker/config.json, which a team with root can read.
#   report      doctor + what to set on the flight controller for this drone
#
# Exit codes: 0 done; 10 rebooting -- run again afterwards; anything else: failed.
set -Eeuo pipefail

log()  { printf '[bootstrap] %s\n' "$*"; }
warn() { printf '[bootstrap] WARNING: %s\n' "$*" >&2; }
die()  { printf '[bootstrap] ERROR: %s\n' "$*" >&2; exit 1; }
usage() { sed -n '2,32p' "$0" | sed 's/^# \{0,1\}//'; }

DRONE_ID=""
TEAM="unassigned"
CAMERA=""
KEY_FILE=""
REF="${SKYBETA_REF:-fleet-ready}"     # switch to the release tag once fleet-ready is merged
REPO_URL="${SKYBETA_REPO_URL:-https://github.com/Artash-N/skybeta.git}"
REPO_DIR="${HOME}/skybeta"
REGISTRY="${REGISTRY:-}"
MIRROR="${MIRROR:-}"
INSECURE_REGISTRY=""
REG_USER=""
REG_TOKEN_FILE=""
REBOOT=1
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --drone-id)      DRONE_ID="${2:-}"; shift 2 ;;
    --team)          TEAM="${2:-}"; shift 2 ;;
    --camera)        CAMERA="${2:-}"; shift 2 ;;
    --auth-key-file) KEY_FILE="${2:-}"; shift 2 ;;
    --ref)           REF="${2:-}"; shift 2 ;;
    --repo-dir)      REPO_DIR="${2:-}"; shift 2 ;;
    --registry)      REGISTRY="${2:-}"; shift 2 ;;
    --mirror)        MIRROR="${2:-}"; shift 2 ;;
    --insecure-registry)   INSECURE_REGISTRY="${2:-}"; shift 2 ;;
    --registry-login)      REG_USER="${2:-}"; shift 2 ;;
    --registry-token-file) REG_TOKEN_FILE="${2:-}"; shift 2 ;;
    --no-reboot)     REBOOT=0; shift ;;
    --dry-run)       DRY_RUN=1; shift ;;
    -h|--help)       usage; exit 0 ;;
    *) usage >&2; die "unknown argument: $1" ;;
  esac
done

# The key files (Tailscale, registry token) must never outlive this script, whatever happens below.
cleanup_key() {
  [[ -n "${KEY_FILE}" && -e "${KEY_FILE}" ]] && rm -f "${KEY_FILE}"
  [[ -n "${REG_TOKEN_FILE}" && -e "${REG_TOKEN_FILE}" ]] && rm -f "${REG_TOKEN_FILE}"
  return 0
}
trap cleanup_key EXIT

[[ "${DRONE_ID}" =~ ^[0-9]+$ ]] && (( 10#${DRONE_ID} >= 1 && 10#${DRONE_ID} <= 99 )) \
  || die "--drone-id must be 1..99 (it becomes the DDS domain), got '${DRONE_ID}'"
DRONE_ID=$((10#${DRONE_ID}))
HOSTNAME_WANT="$(printf 'skybeta-%02d' "${DRONE_ID}")"
if [[ -n "${CAMERA}" ]]; then
  read -r -a CAM <<< "${CAMERA}"
  [[ ${#CAM[@]} -eq 6 ]] || die "--camera wants 6 numbers: \"x y z roll pitch yaw\""
fi
[[ "${TEAM}" =~ ^[A-Za-z0-9._-]+$ ]] || die "--team: letters, digits, . _ - only (it goes in drone.env)"

STATE_DIR="${HOME}/.skybeta"
STATE="${STATE_DIR}/bootstrap-state"
mkdir -p "${STATE_DIR}"; touch "${STATE}"
# Record of completed steps (each step is idempotent on its own; this is the audit trail).
mark()      { [[ "${DRY_RUN}" -eq 1 ]] || grep -qxF "$1" "${STATE}" || echo "$1" >> "${STATE}"; }
run()       { if [[ "${DRY_RUN}" -eq 1 ]]; then printf '    would run: %s\n' "$*"; else "$@"; fi; }

# --- preflight ----------------------------------------------------------------------
[[ -r /etc/nv_tegra_release && "$(head -1 /etc/nv_tegra_release)" == *R36* ]] \
  || die "not a JetPack 6 (L4T R36) Jetson: $(head -1 /etc/nv_tegra_release 2>/dev/null || echo 'no /etc/nv_tegra_release')"
[[ "$(id -u)" -ne 0 ]] || die "run as the drone's login user, not root (it uses sudo where needed)"
log "drone ${DRONE_ID} (${HOSTNAME_WANT}), team ${TEAM}, repo ${REPO_URL}@${REF} -> ${REPO_DIR}"
[[ "${DRY_RUN}" -eq 1 ]] && log "DRY RUN: nothing is changed, no sudo is used"

if [[ "${DRY_RUN}" -eq 0 ]]; then
  log "sudo is needed for the hostname, Tailscale, drone.env and provisioning:"
  sudo -v || die "sudo failed"
  # Keep the sudo timestamp fresh through long apt/docker steps.
  ( while sudo -n true 2>/dev/null; do sleep 50; done ) &
  SUDO_KEEPALIVE=$!
  trap 'cleanup_key; kill "${SUDO_KEEPALIVE}" 2>/dev/null || true' EXIT
fi

# --- hostname -------------------------------------------------------------------------
step_hostname() {
  if [[ "$(hostname)" == "${HOSTNAME_WANT}" ]]; then
    log "hostname: already ${HOSTNAME_WANT}"
  else
    log "hostname: $(hostname) -> ${HOSTNAME_WANT}"
    run sudo hostnamectl set-hostname "${HOSTNAME_WANT}"
    if grep -qE '^127\.0\.1\.1[[:space:]]' /etc/hosts; then
      run sudo sed -i -E "s/^127\.0\.1\.1[[:space:]].*/127.0.1.1\t${HOSTNAME_WANT}/" /etc/hosts
    else
      if [[ "${DRY_RUN}" -eq 1 ]]; then
        printf '    would append to /etc/hosts: 127.0.1.1\t%s\n' "${HOSTNAME_WANT}"
      else
        printf '127.0.1.1\t%s\n' "${HOSTNAME_WANT}" | sudo tee -a /etc/hosts >/dev/null
      fi
    fi
  fi
  mark hostname
}

# --- tailscale ------------------------------------------------------------------------
ts_field() {   # ts_field BackendState | tags | user
  tailscale status --json 2>/dev/null | python3 -c '
import json, sys
s = json.load(sys.stdin); me = s.get("Self", {}) or {}
f = sys.argv[1]
if f == "BackendState": print(s.get("BackendState", ""))
elif f == "tags":       print(",".join(me.get("Tags") or []))
elif f == "user":       print((s.get("User") or {}).get(str(me.get("UserID")), {}).get("LoginName", ""))
' "$1" 2>/dev/null || true
}

step_tailscale() {
  if ! command -v tailscale >/dev/null 2>&1; then
    log "tailscale: installing"
    if [[ "${DRY_RUN}" -eq 1 ]]; then
      printf '    would run: curl -fsSL https://tailscale.com/install.sh | sh\n'
    else
      curl -fsSL https://tailscale.com/install.sh | sh
    fi
  fi

  local state tags
  state="$(ts_field BackendState)"
  tags="$(ts_field tags)"
  if [[ "${state}" == "Running" && ",${tags}," == *",tag:drone,"* ]]; then
    log "tailscale: already on the tailnet as tag:drone"
    run sudo tailscale set --hostname="${HOSTNAME_WANT}" --ssh
  elif [[ "${state}" == "Running" ]]; then
    # Logged in as a person. Re-authenticating with the key would work, but it would
    # cut the SSH session this script is probably running in. Tag it from the console.
    warn "tailscale: logged in as $(ts_field user) rather than tag:drone."
    warn "  Tag it in the admin console (Machines -> ${HOSTNAME_WANT} -> Edit ACL tags -> tag:drone)."
    warn "  Leaving the login as it is so this SSH session survives."
    run sudo tailscale set --hostname="${HOSTNAME_WANT}" --ssh
  else
    [[ -n "${KEY_FILE}" && -r "${KEY_FILE}" ]] \
      || die "tailscale: not logged in and no readable --auth-key-file (new_drone.sh provides one)"
    log "tailscale: joining as tag:drone, hostname ${HOSTNAME_WANT}"
    # file: makes tailscale read the key itself -- it is never on a command line.
    run sudo tailscale up --auth-key="file:${KEY_FILE}" --advertise-tags=tag:drone \
      --hostname="${HOSTNAME_WANT}" --ssh
  fi
  cleanup_key
  if [[ "${DRY_RUN}" -eq 0 ]]; then
    log "tailscale: $(tailscale ip -4 2>/dev/null | head -1) as $(ts_field tags)"
  fi
  mark tailscale
}

# --- repo -----------------------------------------------------------------------------
step_repo() {
  if [[ -d "${REPO_DIR}/.git" ]]; then
    if [[ -n "$(git -C "${REPO_DIR}" status --porcelain)" ]]; then
      warn "repo: ${REPO_DIR} has local changes -- not touching it (commit or stash them, then re-run)"
    else
      log "repo: ${REPO_DIR} -> ${REF}"
      run git -C "${REPO_DIR}" fetch -q origin
      run git -C "${REPO_DIR}" checkout -q "${REF}"
      run git -C "${REPO_DIR}" pull -q --ff-only || true
    fi
  else
    log "repo: cloning ${REPO_URL}@${REF} -> ${REPO_DIR}"
    run git clone -q -b "${REF}" "${REPO_URL}" "${REPO_DIR}"
  fi
  if [[ "${DRY_RUN}" -eq 0 ]]; then
    [[ -f "${REPO_DIR}/host/drone.env.example" ]] \
      || die "repo: ${REPO_DIR}@${REF} has no host/drone.env.example -- that ref predates fleet support"
    log "repo: at $(git -C "${REPO_DIR}" log --oneline -1)"
  fi
  mark repo
}

# --- identity -------------------------------------------------------------------------
render_drone_env() {
  local tmpl="${REPO_DIR}/host/drone.env.example"
  local -a sed_args=(
    -e "s/^DRONE_ID=.*/DRONE_ID=${DRONE_ID}/"
    -e "s/^ROS_DOMAIN_ID=.*/ROS_DOMAIN_ID=${DRONE_ID}/"
    -e "s/^TEAM=.*/TEAM=${TEAM}/"
  )
  if [[ -n "${CAMERA}" ]]; then
    sed_args+=(
      -e "s/^CAMERA_X=.*/CAMERA_X=${CAM[0]}/"     -e "s/^CAMERA_Y=.*/CAMERA_Y=${CAM[1]}/"
      -e "s/^CAMERA_Z=.*/CAMERA_Z=${CAM[2]}/"     -e "s/^CAMERA_ROLL=.*/CAMERA_ROLL=${CAM[3]}/"
      -e "s/^CAMERA_PITCH=.*/CAMERA_PITCH=${CAM[4]}/" -e "s/^CAMERA_YAW=.*/CAMERA_YAW=${CAM[5]}/"
    )
  fi
  sed "${sed_args[@]}" "${tmpl}"
}

step_identity() {
  local target=/etc/skybeta/drone.env
  if [[ "${DRY_RUN}" -eq 1 && ! -f "${REPO_DIR}/host/drone.env.example" ]]; then
    # No template to render yet (the repo is not cloned in a dry run).
    if [[ -f "${target}" ]]; then
      printf '    %s already exists (DRONE_ID=%s); would keep it if it matches drone %s, else show the diff\n' \
        "${target}" "$(sed -n 's/^DRONE_ID=//p' "${target}")" "${DRONE_ID}"
    else
      printf '    would write %s from the repo template (DRONE_ID=%s ROS_DOMAIN_ID=%s TEAM=%s)\n' \
        "${target}" "${DRONE_ID}" "${DRONE_ID}" "${TEAM}"
    fi
    return 0
  fi
  local want; want="$(render_drone_env)"
  if [[ -f "${target}" ]]; then
    if [[ "$(cat "${target}")" == "${want}" ]]; then
      log "identity: ${target} already correct"
    else
      warn "identity: ${target} exists and differs -- keeping it. Difference (- current, + would write):"
      diff <(cat "${target}") <(printf '%s\n' "${want}") | sed 's/^/      /' >&2 || true
      warn "  To replace it: sudo rm ${target} and re-run."
    fi
  else
    log "identity: writing ${target} (drone ${DRONE_ID}, domain ${DRONE_ID}, team ${TEAM})"
    if [[ "${DRY_RUN}" -eq 1 ]]; then
      printf '    would write %s\n' "${target}"
    else
      sudo install -d -m 0755 /etc/skybeta
      printf '%s\n' "${want}" | sudo tee "${target}" >/dev/null
      sudo chmod 0644 "${target}"
    fi
  fi
  mark identity
}

# --- provision ------------------------------------------------------------------------
uart_patch_live() { [[ -e /proc/device-tree/bus@0/serial@3100000/iommus ]]; }

# --- registry access (before the pull, and before any reboot: the token is in tmpfs) ---
registry_host() { printf '%s' "${1%%/*}"; }
docker_logged_in() {   # is there an auths entry for this registry host in ~/.docker/config.json?
  python3 - "$1" <<'PY' 2>/dev/null
import json, os, sys
p = os.path.expanduser("~/.docker/config.json")
try:
    auths = json.load(open(p)).get("auths", {})
except Exception:
    sys.exit(1)
sys.exit(0 if any(k.split("://")[-1].rstrip("/") == sys.argv[1] for k in auths) else 1)
PY
}

step_registry() {
  if [[ -n "${INSECURE_REGISTRY}" ]]; then
    if [[ "${DRY_RUN}" -eq 1 ]]; then
      printf '    would add %s to insecure-registries in /etc/docker/daemon.json (and restart docker)\n' "${INSECURE_REGISTRY}"
    else
      local changed
      changed="$(sudo python3 - "${INSECURE_REGISTRY}" <<'PY'
import json, os, shutil, sys, time
p, want = "/etc/docker/daemon.json", sys.argv[1]
d = json.load(open(p)) if os.path.exists(p) and os.path.getsize(p) else {}
regs = d.setdefault("insecure-registries", [])
if want in regs:
    print("unchanged")
else:
    if os.path.exists(p):
        shutil.copy2(p, "%s.bak-%s" % (p, time.strftime("%Y%m%d-%H%M%S")))
    regs.append(want)
    tmp = p + ".skybeta-tmp"
    with open(tmp, "w") as f:
        json.dump(d, f, indent=2)
        f.write("\n")
    os.replace(tmp, p)
    print("changed")
PY
)" || die "registry: could not update /etc/docker/daemon.json"
      if [[ "${changed}" == changed ]]; then
        log "registry: ${INSECURE_REGISTRY} added to docker's insecure-registries; restarting docker"
        sudo systemctl restart docker || die "registry: docker did not restart"
      else
        log "registry: ${INSECURE_REGISTRY} already in insecure-registries"
      fi
    fi
  fi
  if [[ -n "${REG_USER}" ]]; then
    [[ -n "${REGISTRY}" ]] || die "--registry-login needs --registry"
    local host; host="$(registry_host "${REGISTRY}")"
    if [[ "${DRY_RUN}" -eq 1 ]]; then
      printf '    would run: docker login %s -u %s --password-stdin  (token from %s, then deleted)\n' \
        "${host}" "${REG_USER}" "${REG_TOKEN_FILE:-<none>}"
    elif [[ -n "${REG_TOKEN_FILE}" && -r "${REG_TOKEN_FILE}" ]]; then
      docker login "${host}" -u "${REG_USER}" --password-stdin < "${REG_TOKEN_FILE}" >/dev/null \
        || { rm -f "${REG_TOKEN_FILE}"; die "registry: docker login ${host} failed"; }
      rm -f "${REG_TOKEN_FILE}"
      log "registry: logged in to ${host} as ${REG_USER} (token file deleted; the credential now sits in ~/.docker/config.json -- use a read-only pull token)"
    elif docker_logged_in "${host}"; then
      log "registry: already logged in to ${host}"
    else
      die "registry: --registry-login ${REG_USER} but no token file (it is RAM-only and lost on reboot). Run new_drone.sh again"
    fi
  fi
  if [[ -n "${INSECURE_REGISTRY}${REG_USER}" ]]; then mark registry; fi
  return 0
}

step_provision() {
  local prov="${REPO_DIR}/host/provision.sh"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    if [[ -x "${prov}" ]]; then
      REGISTRY="${REGISTRY}" MIRROR="${MIRROR}" bash "${prov}" --dry-run | sed 's/^/    /'
    else
      printf '    would run: %s   (REGISTRY=%s)\n' "${prov}" "${REGISTRY:-<unset: image pull skipped>}"
    fi
    step_registry
    return 0
  fi
  log "provision (1/2): ${prov} --skip pull --skip doctor"
  bash "${prov}" --skip pull --skip doctor || die "provision.sh failed (see above); fix and re-run"
  step_registry

  if ! uart_patch_live; then
    # provision.sh staged the device-tree patch and stopped; it only goes live on boot.
    if [[ "${REBOOT}" -eq 1 ]]; then
      log "provision: the UART patch needs a reboot -- rebooting now. Run again afterwards (exit 10)."
      sudo systemctl reboot
      exit 10
    fi
    log "provision: reboot needed for the UART patch (--no-reboot given). Reboot, then re-run."
    exit 10
  fi
  [[ -n "${REGISTRY}" ]] || warn "provision: no --registry -- the image pull is skipped (new_drone.sh --image-from copies it from another drone)"
  log "provision (2/2): ${prov} --from pull"
  REGISTRY="${REGISTRY}" MIRROR="${MIRROR}" bash "${prov}" --from pull || die "provision.sh failed (see above); fix and re-run"
  mark provision
}

# --- report ---------------------------------------------------------------------------
step_report() {
  echo
  log "done: ${HOSTNAME_WANT} is drone ${DRONE_ID} (team ${TEAM})."
  [[ "${DRY_RUN}" -eq 1 ]] && return 0
  bash "${REPO_DIR}/host/doctor.sh" --no-uart || warn "doctor reported a FAIL (see above)"
  cat <<EOM

[bootstrap] Now on the flight controller (QGroundControl), then reboot the FC:
              load px4/params/02_telem2_uxrce_dds.params   (sets UXRCE_DDS_PTCFG=2)
              set  UXRCE_DDS_DOM_ID = ${DRONE_ID}
            and check on the Jetson:  cd ${REPO_DIR} && host/run.sh stack   then   check_topics.sh
EOM
}

step_hostname
step_tailscale
step_repo
step_identity
step_provision
step_report
exit 0
