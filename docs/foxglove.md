# Foxglove

View the drone's ROS 2 topics live in Foxglove.

## 1. Start the bridge (on the Jetson)

```bash
docker exec -it skybeta-stack bash -c '
  source skybeta_env.sh
  source /root/skybeta_ws/install/setup.bash
  ros2 launch foxglove_bridge foxglove_bridge_launch.xml port:=8766 address:=0.0.0.0'
```

Use port **8766**. Port 8765 is taken by `skybeta-status` and fails with `Bind Error`.

## 2. Connect (on your laptop)

1. Foxglove → **Open connection** → **Foxglove WebSocket** (not Rosbridge).
2. URL: `ws://<drone-ip>:8766`. Use the LAN IP, or the Tailscale IP if you're on Tailscale.
3. The desktop app works best. The web app may block `ws://` connections to local IPs.

## 3. Useful topics

| What | Panel | Topic |
|---|---|---|
| Cameras | Image | `/oak/left/image_raw/compressed` |
| IMU | Plot | `/oak/imu/data` |
| VSLAM | 3D | `/visual_slam/tracking/odometry`, `/tf` |
| PX4 | Raw Messages | `/fmu/out/vehicle_odometry`, `/fmu/out/battery_status` |

Over Wi-Fi, use the `/compressed` image topics. The raw images use too much bandwidth.
