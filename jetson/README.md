# Snapshot of skybeta-14 host setup (taken 2026-11-06)

Repo root = byte-for-byte copy of `/home/dronehacks/skybeta` on drone 14
(upstream `Artash-N/skybeta`, branch `fleet-ready`, commit `baf21fd`, VERSION 0.3.2,
clean working tree). Upstream remote kept as `upstream`.

This folder holds the things on the Jetson that live outside that checkout:

- `drone_bootstrap.sh`              `~/drone_bootstrap.sh` (fleet provisioning script, Oct 18 copy)
- `dot-skybeta/`                    `~/.skybeta/` (newer bootstrap script copy + step-state files)
- `team_ws/`                        `~/team_ws` (bind-mounted at /root/team_ws in the stack; empty `src/`)
- `NV_SET_TARGET_DOCKER_APT_REPO_MODAL.sh`  NVIDIA docker apt repo helper left in `~`
- `system-snapshot.txt`             `/etc/skybeta/drone.env`, `/etc/skybeta/status-agent.env`,
                                    `docker images`, `skybeta-stack.service`, bootstrap state, HEAD

Key facts: DRONE_ID=14, ROS_DOMAIN_ID=14, TEAM=unassigned, SKYBETA_DDS_SCOPE=local,
camera offset x=0.1053 z=0.0304 m, stack image `skybeta:0.3.2`, containers
`skybeta-stack` and `skybeta-ros-status` running, JetPack L4T R36.5.0 (kernel 5.15.185-tegra).
