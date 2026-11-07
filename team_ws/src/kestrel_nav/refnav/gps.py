"""Read native PX4 GPS before a GPS-only example. Never publishes any message."""
import math
import time


def usable_fix(msg):
    return (msg.timestamp > 0 and msg.fix_type >= 3 and msg.satellites_used >= 4
            and math.isfinite(msg.eph) and 0 <= msg.eph <= 10.)


def wait_for_gps(timeout_s=30.):
    import rclpy
    from rclpy.node import Node
    from px4_msgs.msg import SensorGps
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    if timeout_s < 25:
        raise ValueError("GPS absence checks require at least 25 seconds of discovery")
    rclpy.init(args=[])
    node = Node("gps_lesson_readiness")
    samples = []
    def on(msg):
        if usable_fix(msg):
            if not samples or msg.timestamp > samples[-1]:
                samples.append(msg.timestamp)
        else:
            samples.clear()
    qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                     history=HistoryPolicy.KEEP_LAST, depth=1)
    node.create_subscription(SensorGps, "/fmu/out/vehicle_gps_position", on, qos)
    deadline = time.monotonic() + timeout_s
    try:
        while time.monotonic() < deadline and len(samples) < 2:
            rclpy.spin_once(node, timeout_sec=.1)
        return len(samples) >= 2
    finally:
        node.destroy_node(); rclpy.try_shutdown()
