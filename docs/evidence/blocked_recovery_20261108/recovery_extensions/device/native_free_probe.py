import json
import os
import tempfile
import time
os.environ['ROS_DOMAIN_ID']='77'
os.environ['ROS_LOCALHOST_ONLY']='1'
import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid
from coursenav.loc_node import LocNode
from coursenav.localgrid import LocalGrid
from coursenav.observed_free import ObservedFree
from coursenav.scan import Scan

with tempfile.TemporaryDirectory(prefix='blocked-free-probe-') as logs:
    rclpy.init(args=['--ros-args','-p',f'log_dir:={logs}','-p','height_source:=local_position','-p','use_sim_time:=false'])
    node=LocNode()
    received=[]
    sub=node.create_subscription(OccupancyGrid,'/coursenav/local_occupancy',received.append,10)
    try:
        ray=Scan(np.array([0.]),np.array([2.]),np.array([.01]))
        results=[]
        for age in (0.,9.):
            node.local=LocalGrid(node.local.p)
            stamp=node.get_clock().now().nanoseconds*1e-9-age
            for _ in range(5):node.local.integrate(stamp,[0.,0.,0.],ray)
            received.clear()
            deadline=time.monotonic()+3.
            while not received and time.monotonic()<deadline:
                node._publish_local()
                rclpy.spin_once(node,timeout_sec=.1)
            assert received,'no native ROS local occupancy received'
            msg=received[-1]
            ij=node.local.ij([1.,0.])[0]
            prob=int(np.asarray(msg.data).reshape(msg.info.height,msg.info.width)[ij[1],ij[0]])
            expected_free=age==0.
            assert (0<=prob<=35) if expected_free else prob==-1,(age,prob)
            grid=ObservedFree.from_grid([msg.info.origin.position.x,msg.info.origin.position.y],msg.info.resolution,
                                       msg.info.width,msg.info.height,msg.data,
                                       msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9)
            assert grid.fresh(node.get_clock().now().nanoseconds*1e-9,2.)
            results.append(dict(observation_age_s=age,published_probability=prob))
        print(json.dumps(dict(result='PASS',domain=77,observed_free_import=True,native_ros_local_publish=results)))
    finally:
        node.destroy_node()
        rclpy.shutdown()
