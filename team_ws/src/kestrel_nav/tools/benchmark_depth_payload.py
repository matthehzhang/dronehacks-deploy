#!/usr/bin/env python3
"""Measure ROS Image payload assignment, including conversion, on the current host.

Run after sourcing ROS. Validation-on emulates the costly generated setter path;
Jazzy normally disables it. These desktop timings are not Jetson measurements.
"""
import array
import json
import platform
import statistics
import time

import numpy as np
from rclpy.serialization import deserialize_message, serialize_message
from sensor_msgs.msg import Image


def main():
    depth = np.arange(512 * 320, dtype=np.uint16).reshape(320, 512)
    result = {'host': platform.node(), 'payload_bytes': depth.nbytes}
    for check_fields in (False, True):
        timing, messages = {}, []
        for name, make in (('bytes', lambda: depth.tobytes()),
                           ('uint8_array', lambda: array.array('B', depth.tobytes()))):
            try:
                msg = Image(height=320, width=512, encoding='16UC1', step=1024, check_fields=check_fields)
            except (AssertionError, TypeError):
                # Humble has always-on checks and no check_fields constructor option.
                if not check_fields:
                    break
                msg = Image(height=320, width=512, encoding='16UC1', step=1024)
            samples = []
            for _ in range(21):
                start = time.process_time_ns()
                msg.data = make()
                samples.append((time.process_time_ns() - start) / 1e6)
            timing[name + '_median_cpu_ms'] = statistics.median(samples)
            messages.append(msg)
        if messages:
            assert messages[0] == messages[1]
            # CDR padding bytes can differ; compare deserialized fields and pixel bytes.
            assert deserialize_message(serialize_message(messages[0]), Image) == messages[1]
            assert deserialize_message(serialize_message(messages[1]), Image) == messages[0]
            result['validation_' + str(check_fields).lower()] = timing
    result['messages_and_serialization_roundtrips_identical'] = True
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
