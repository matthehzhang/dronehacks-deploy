# Resume after the user reports reconnection

Candidate: `50978085fedb6b3f56e1b23cb904167c15383eec`, branch `cram304`.
Local frozen archive source: `/tmp/blocked-recovery-5097808/kestrel_nav`.
Expected 159-file SHA256 manifest: `local_manifest.json` beside this file.

1. Use only `ssh dronehacks`. Read fresh `/skybeta/state`; require explicit disarmed and landed before replacing active files. Do not restart or command flight.
2. Recopy/verify exact candidate into `~/team_ws/src/kestrel_nav`; update and verify `~/policies/group154_blocked_recovery/kestrel_nav`. No nested workspace. Exclude only runtime logs, caches, build/install outputs and egg-info.
3. Restore the temporary verification helper if needed; its absence does not establish that a reboot occurred. Verify every manifest path and no extra runtime files. Signal COPY VERIFIED only after both checks pass.
4. Inside `skybeta-stack`, run `cd /root/team_ws && team_build.sh --packages-select kestrel_nav`. Source `skybeta_env.sh`, then `colcon list` must show only one `kestrel_nav`.
5. Run package `tools/check_localizer_startup.py --fit 0.35` and `--fit 0.42`. These use isolated ROS domain77 and localhost only. Repeat the prior evidence `newprior_integration/device/9b76afd/native_nav_constructor.py`, which generates a temporary dry-run route and constructs the actual navigator with all three retained preplan references. Do not launch the flight stack.
6. Resolve `ros2 launch kestrel_nav refnav_skybeta.launch.py --show-args` only. Preserve outputs and report native results separately from laptop/simulation tests. Update the saved marker only with results actually observed.

Current state is partial, not verified: active rsync succeeded after natural landing, but the saved copy and all post-copy verification/native checks are unfinished. The user confirmed disconnection. Reconnect attempts are paused until a new signal.
