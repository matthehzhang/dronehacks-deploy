# Two most recent ROS bags from the 2026-11-08 pull

This archive now contains only the two newest bags by remote file modification time
within the original 43-bag USB-C pull:

- `20261107-235044_group154_OvyGUo/bag/bag_0.mcap` (14,360,576 bytes).
- `20261107-234611_group154_fmlluu/bag/bag_0.mcap` (46,792,532 bytes), with its `metadata.yaml`.

The first recording had no metadata file on the drone. Both are preserved unchanged;
no recording was repaired or reindexed. Their hashes match the verified USB pull.
The previous all-recordings archive is replaced in this branch's current tree;
older commits retain their historical contents. Source recordings on the drone remain.

The ZIP is stored in Git LFS:

```sh
git lfs pull --include="docs/evidence/drone_bags_20261108/*.zip"
unzip docs/evidence/drone_bags_20261108/dronehacks-bags-2026-11-08.zip
```

`manifest.json` records selection timestamps, file sizes and SHA-256 checksums.
The ZIP integrity check and extracted-file hash verification passed.
