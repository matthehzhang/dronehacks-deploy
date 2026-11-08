# Drone recordings pulled on 2026-11-08

The archive contains all 45 bag files copied from `skybeta-14` over USB-C,
plus their available metadata and accompanying run logs (389 files total).
Original directory paths are preserved under `dronehacks-bags-2026-11-08/`.
The source files on the drone were retained. No recordings were repaired or reindexed.
Some source recordings have no `metadata.yaml`; the archive preserves them as found.

Sources include `team_ws/runs`, `team_ws/flight_logs`, `team_ws/bags`,
`depth_bench/runs`, and the `matthew930am_ref` evidence recording.

The ZIP is stored in Git LFS. From the repository root:

```sh
git lfs pull --include="docs/evidence/drone_bags_20261108/*.zip"
unzip docs/evidence/drone_bags_20261108/dronehacks-bags-2026-11-08.zip
```

`manifest.json` lists every archived file, its size and SHA-256 checksum,
as well as the archive checksum. The USB copy was checksum-verified against
the drone, and the ZIP passed its integrity check before this commit.
