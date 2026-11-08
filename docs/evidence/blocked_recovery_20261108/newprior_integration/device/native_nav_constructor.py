import json
import os
from pathlib import Path
import sys
import tempfile
import time
os.environ['ROS_DOMAIN_ID']='77'
os.environ['ROS_LOCALHOST_ONLY']='1'
sys.path.insert(0,'/root/team_ws/src/kestrel_nav')
import fly_course
import rclpy
from coursenav.nav_node import NavNode
with tempfile.TemporaryDirectory(prefix='blocked-native-nav-') as temp:
    os.environ['RUN_RECORD_DIR']=temp
    result=fly_course.main(['--dry-run','--skip-preflight','--no-own-localizer','--no-return',
                            '--start-from-mark','--landed-z','.10'])
    assert result==0,result
    routes=list(Path(temp).glob('run_*/route.json'))
    assert len(routes)==1,routes
    route=routes[0]
    rclpy.init(args=['--ros-args','-p',f'route_file:={route}','-p','declared_heading_deg:=0.0','-p','use_sim_time:=false'])
    node=None
    try:
        node=NavNode()
        deadline=time.monotonic()+1.2
        while time.monotonic()<deadline:rclpy.spin_once(node,timeout_sec=.1)
        expected=sum(leg['path'] for leg in node.nav_legs)
        assert len(node.cache)==expected,(len(node.cache),expected)
        print(json.dumps(dict(result='PASS',domain=77,native_nav_constructor=True,
                              total_legs=len(node.nav_legs),preplanned_path_legs=len(node.cache),
                              prior_state=node.prior_state,source='/root/team_ws/src/kestrel_nav')))
    finally:
        if node is not None:node.destroy_node()
        rclpy.shutdown()
