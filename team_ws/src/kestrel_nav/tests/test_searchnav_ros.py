"""ROS adapter reset regression; runs in the actual ROS/px4_msgs environment."""
from collections import deque
from types import SimpleNamespace, MethodType

import numpy as np
import pytest

pytest.importorskip('rclpy')
pytest.importorskip('px4_msgs')

from kestrel_nav.searchnav.node import SearchNode, quat_R, INPUT_LIMITS
from kestrel_nav.searchnav.grid import DepthGrid, GridParams, UNKNOWN
from kestrel_nav.searchnav.mission import SearchMission
from kestrel_nav.searchnav.floor import FloorTracker
from kestrel_nav.searchnav.inputs import DepthWatchdog, Px4ClockAnchor
from kestrel_nav.searchnav.target import TargetTracker, RGBBlueDetector
from kestrel_nav.searchnav.launch_volume import LaunchVolume, LaunchVolumeParams
from kestrel_nav.searchnav.mission import Observation


def test_ground_ekf_reset_clears_old_floor_map_history_target_and_qualification():
    grid=DepthGrid(GridParams(extent_m=4.))
    grid.reset((0.,0.),.8)
    grid.states.fill(0)
    node=SimpleNamespace(grid=grid,mission=SearchMission(grid),last_reset=(0,0,0),
        frame_resets=[],frame_epoch_stamp=-np.inf,pose_history=deque([(1.,np.zeros(3))]),
        att_history=deque([(1.,np.eye(3))]),att=np.eye(3),samples={'depth':(1.,1.)},
        errors={},floor_z=-.15,floor_source='calibrated_depth_plane',
        floor_tracker=FloorTracker(),depth_watchdog=DepthWatchdog(),last_depth=object(),
        latest_candidates=[{'old':'candidate'}],tracker=TargetTracker(),
        qualified_since=1.,ready=True,saw_disarmed=True,last_command={'old':'command'},
        landed=True,px4_accept=lambda source,m:2.)
    node.tracker.confirmed=((1.,1.),1.)
    node.launch_volume=LaunchVolume(LaunchVolumeParams(True,(-1.,-1.,-1.,1.,1.,1.)))
    assert node.launch_volume.grant((0.,0.,.8),1.,pilot_verified=True,saw_disarmed=True,
                                    allow_takeoff=True,pilot_started=True)
    node.reset_preflight_frame=MethodType(SearchNode.reset_preflight_frame,node)
    message=SimpleNamespace(x=10.,y=20.,z=-3.,vx=0.,vy=0.,vz=0.,heading=0.,
        xy_valid=True,z_valid=True,v_xy_valid=True,v_z_valid=True,
        xy_reset_counter=1,z_reset_counter=1,heading_reset_counter=1,
        dist_bottom=float('nan'),dist_bottom_valid=False)
    SearchNode.on_position(node,message)
    assert node.floor_z is None and node.floor_source is None
    assert node.grid.height is None and np.all(node.grid.states==UNKNOWN)
    assert node.tracker.confirmed is None and not node.latest_candidates
    assert len(node.pose_history)==1 and np.allclose(node.pose_history[0][1],(20.,10.,3.))
    assert not node.att_history and node.att is None
    assert not node.samples and node.last_depth is None
    assert not node.ready and not node.saw_disarmed and node.qualified_since is None
    assert node.frame_epoch_stamp==2. and node.mission.phase=='wait'
    assert node.last_reset==(1,1,1)
    assert not node.launch_volume.status(2.)['active']
    assert not node.depth_watchdog.usable(1.9,2.5)
    # Capture from before the frame epoch must not restore qualification even
    # if it would otherwise be within the normal .5s freshness window.
    node.now=lambda:2.1
    node.counts={}
    assert not SearchNode.accept(node,'depth',1.99)
    assert not node.samples


def test_fresh_independent_vio_health_survives_px4_epoch_without_reusing_spatial_data():
    node,now=sensor_node()
    node.frame_epoch_stamp=40.58332602
    now[0]=40.60416017
    node.mission.started=24.
    node.mission.phase='recover'
    # Actual bag regression: healthy VIO latency ~.1s means capture precedes
    # the fused-PX4 reset. The adapter uses this only for source health, so
    # retaining its timestamp cannot restore a map/target or old camera pose.
    assert SearchNode.accept(node,'vio',40.49166)
    assert node.samples['vio'][0]==40.49166
    assert not SearchNode.accept(node,'depth',40.49166)
    assert not SearchNode.accept(node,'attitude',40.49166)
    assert not node.fault
    # Original .5s capture deadline and monotonic receipt protection remain.
    assert not SearchNode.accept(node,'vio',40.0)
    assert node.fault


def image_message(t,depth=True):
    data=np.full((120,160),2.,np.float32) if depth else np.full((120,160),120,np.uint8)
    sec=int(t)
    return SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(sec=sec,nanosec=int(round((t-sec)*1e9))),frame_id='camera'),
                           encoding='32FC1' if depth else 'mono8',height=120,width=160,
                           step=640 if depth else 160,is_bigendian=0,data=data.tobytes())


def sensor_node():
    grid=DepthGrid(GridParams(extent_m=6.,border_px=4))
    grid.reset((0.,0.),1.)
    optical=np.array([[0.,0.,1.],[-1.,0.,0.],[0.,-1.,0.]])
    now=[1.]
    node=SimpleNamespace(now=lambda:now[0],grid=grid,mission=SearchMission(grid),
        frame_epoch_stamp=-np.inf,samples={},errors={},counts={},depth_hz=4.,detector_hz=3.,
        last_depth_run=-np.inf,last_mono_run=-np.inf,landed=False,floor_z=None,
        last_depth=None,depth_valid_fraction=0.,depth_watchdog=DepthWatchdog(),
        tracker=TargetTracker(),latest_candidates=[],fault='',capture_waiting=set(),
        calibrated_pose=lambda which,m:((100.,100.,79.5,59.5),optical,np.array([0.,0.,1.]),SimpleNamespace(d=[])))
    node.accept=MethodType(SearchNode.accept,node)
    node.reject=MethodType(SearchNode.reject,node)
    SearchNode.reset_openings(node)
    return node,now


def test_depth_freshness_tracks_every_usable_frame_while_mapping_is_throttled():
    node,now=sensor_node()
    SearchNode.on_depth(node,image_message(1.))
    assert node.samples['depth'][0]==1.
    assert node.counts['map_full_frames']==1
    now[0]=1.10
    SearchNode.on_depth(node,image_message(1.10))
    assert node.samples['depth'][0]==pytest.approx(1.10)
    assert node.depth_watchdog.capture==pytest.approx(1.10)
    assert node.counts['map_full_frames']==1 and node.counts['map_hit_frames']==1
    assert node.counts['depth_frames']==2


def test_mono_freshness_tracks_every_usable_frame_while_detection_is_throttled():
    node,now=sensor_node()
    SearchNode.on_mono(node,image_message(1.,False))
    now[0]=1.10
    SearchNode.on_mono(node,image_message(1.10,False))
    assert node.samples['mono'][0]==pytest.approx(1.10)
    assert node.counts['mono_frames']==2
    assert node.last_mono_run==1.  # inference itself remains rate capped.


def test_px4_before_first_ros_clock_cannot_poison_offset_or_freshness():
    node,now=sensor_node()
    node.last_source={}
    node.px4_offset=None
    node.px4_clock=Px4ClockAnchor()
    node.last_reset=None
    node.pose_history=deque()
    node.px4_accept=MethodType(SearchNode.px4_accept,node)
    message=SimpleNamespace(timestamp=2000000,x=0.,y=0.,z=0.,vx=0.,vy=0.,vz=0.,heading=0.,
        xy_valid=True,z_valid=True,v_xy_valid=True,v_z_valid=True,
        xy_reset_counter=0,z_reset_counter=0,heading_reset_counter=0,
        dist_bottom=float('nan'),dist_bottom_valid=False)
    now[0]=0.
    SearchNode.on_position(node,message)
    assert node.px4_offset is None and not node.last_source and not node.samples
    assert not node.pose_history
    now[0]=5.
    message.timestamp=0
    SearchNode.on_position(node,message)
    assert node.px4_offset is None and not node.samples
    for t in (5.,5.0625,5.125,5.1875,5.25):
        now[0]=t
        message.timestamp=int(t*1e6)
        SearchNode.on_position(node,message)
    assert node.px4_offset==0.
    assert node.samples['position']==(5.25,5.25)
    assert node.pose_history[-1][0]==5.25
    assert node.position_valid


def test_ground_boot_clock_epoch_reset_clears_old_packet_monotonic_counters():
    node,now=sensor_node()
    node.last_source={'position':100.,'attitude':100.}
    node.px4_offset=-90.
    node.px4_clock=Px4ClockAnchor()
    for ros in (10.,10.0625,10.125,10.1875,10.25):
        node.px4_clock.observe(ros,ros+90.)
    node.last_reset=(0,0,0)
    node.position=(0.,0.,0.)
    node.frame_resets=[]
    node.pose_history=deque()
    node.att_history=deque()
    node.att=None
    node.floor_source=None
    node.floor_tracker=FloorTracker()
    node.qualified_since=9.
    node.ready=True
    node.saw_disarmed=True
    node.last_command=None
    node.reset_preflight_frame=MethodType(SearchNode.reset_preflight_frame,node)
    now[0]=11.
    assert SearchNode.px4_accept(node,'position',SimpleNamespace(timestamp=1000000)) is None
    assert node.last_source=={} and node.px4_offset is None
    assert node.qualified_since is None and not node.ready
    assert node.grid.height is None and np.all(node.grid.states==UNKNOWN)


def test_flags_measured_one_hz_cadence_uses_existing_two_second_yaw_contract():
    guard=DepthWatchdog()
    assert guard.usable(30.,30.)
    node=SimpleNamespace(now=lambda:30.,source_limits=dict(INPUT_LIMITS),
        samples={k:(30.,30.) for k in INPUT_LIMITS},errors={},depth_watchdog=guard,
        link=True,frozen=False,mission=SearchMission(),position_valid=True,health=True,
        yaw_aligned=False,ev_yaw=True,ev_yaw_last=28.98,floor_z=-.15,allow_takeoff=True)
    node.samples['flags']=(28.98,28.98)
    okay,reason=SearchNode.readiness(node)
    assert okay,reason
    assert SearchNode.input_ages(node,30.)['flags']['max_age_s']==2.
    assert INPUT_LIMITS['depth']==INPUT_LIMITS['mono']==INPUT_LIMITS['vio']==.5
    assert INPUT_LIMITS['status']==INPUT_LIMITS['state']==1.
    node.samples['flags']=(27.99,27.99)
    okay,reason=SearchNode.readiness(node)
    assert not okay and reason=='flags: absent/stale input'


@pytest.mark.parametrize('z_changed', [False, True])
def test_small_airborne_reset_discards_geometry_and_requires_post_epoch_samples(z_changed):
    node, now = sensor_node()
    node.mission.started, node.mission.phase = 1., 'move'
    node.mission.hold_z, node.mission.hold_xy = 1., np.zeros(2)
    node.mission.route = [np.zeros(2), np.ones(2)]
    node.last_reset = (3, 0, 1)
    node.armed = node.offboard = True
    node.state = dict(armed=True, offboard_active=True)
    node.frozen = False
    node.floor_z, node.floor_source = -.15, 'calibrated_depth_plane'
    node.frame_resets = []
    node.pose_history = deque([(1., np.zeros(3))])
    node.att_history = deque([(1., np.eye(3))])
    node.att = np.eye(3)
    node.qualified_since, node.ready = .5, True
    node.samples = {k:(1.99, 1.99) for k in ('position','attitude','depth','mono','health')}
    node.grid.states.fill(0)
    node.tracker.confirmed = ((1., 1.), 1.9)
    node.latest_candidates = [{'old':True}]
    msg = SimpleNamespace(delta_xy=[-.0119, -.0091], delta_z=-.02,
                          delta_heading=1.5683)
    counters = (4, int(z_changed), 1)
    SearchNode.handle_airborne_reset(node, msg, counters, 2., (.01, 0., 1.02))
    assert node.mission.phase == 'recover' and not node.fault and not node.mission.route
    assert np.all(node.grid.states == UNKNOWN) and node.grid.height is not None
    assert node.floor_z == pytest.approx(-.13 if z_changed else -.15)
    assert node.mission.hold_z == pytest.approx(1.02 if z_changed else 1.)
    assert not node.pose_history and not node.att_history and node.att is None
    assert node.last_depth is None and node.tracker.confirmed is None and not node.latest_candidates
    assert not node.ready and node.qualified_since is None
    assert not SearchNode.recovery_spatial_ready(node)
    # Last-good timestamps remain subject to their original deadlines and do
    # not count as new spatial observations, even when receipt is recent.
    assert node.samples['depth'] == (1.99, 1.99)
    for source in ('position', 'attitude', 'depth', 'mono'):
        node.samples[source] = (2.01, 2.01)
    assert SearchNode.recovery_spatial_ready(node)


def test_large_airborne_reset_cannot_preserve_measured_floor_or_resume_route():
    node, now = sensor_node()
    node.mission.started, node.mission.phase = 1., 'move'
    node.mission.hold_z = 1.
    node.mission.route = [np.zeros(2), np.ones(2)]
    node.last_reset = (0, 0, 0)
    node.armed = node.offboard = True
    node.state = dict(armed=True, offboard_active=True)
    node.frozen = False
    node.floor_z, node.floor_source = -.15, 'calibrated_depth_plane'
    node.frame_resets = []
    node.pose_history, node.att_history = deque(), deque()
    msg = SimpleNamespace(delta_xy=[.3, 0.], delta_z=0., delta_heading=0.)
    SearchNode.handle_airborne_reset(node, msg, (1, 0, 0), 2., (.3, 0., 1.))
    assert node.fault and node.floor_z is None and node.grid.height is None
    assert not node.mission.route and np.all(node.grid.states == UNKNOWN)
    assert not node.frame_resets[-1]['recoverable']


def test_rgb_callback_keeps_color_calibration_separate_and_publishes_mask_and_proposals():
    node,now=sensor_node()
    SearchNode.reset_rgb(node)
    node.rgb_detector=RGBBlueDetector()
    node.floor_z=0.
    node.sync_tolerance=.12
    K=(100.,100.,79.5,79.5)
    R=np.diag([1.,-1.,-1.]);t=np.array([0.,0.,2.])
    used=[]
    def calibrated(which,m):
        used.append(which)
        return K,R,t,SimpleNamespace(d=[])
    node.calibrated_pose=calibrated
    node.last_depth=(1.,np.full((160,160),2.),K,R,t)
    masks=[];images=[]
    node.blue_mask_pub=SimpleNamespace(publish=masks.append)
    node.target_image_pub=SimpleNamespace(publish=images.append)
    image=np.full((160,160,3),(20,20,180),np.uint8);image[65:95,65:95]=(255,0,0)
    msg=image_message(1.)
    msg.width=msg.height=160;msg.step=480;msg.encoding='bgr8';msg.data=image.tobytes()
    SearchNode.on_rgb(node,msg)
    assert used==['rgb'] and len(node.rgb_candidates)==1
    assert node.rgb_depth_supported_at==1. and node.rgb_tracker.confirmed is None
    assert len(images)==len(masks)==1 and masks[0].encoding=='mono8'
    assert images[0].encoding=='bgr8' and node.counts['rgb_frames']==1
    assert not node.fault
    # A newer unmatched RGB image must withdraw support, not reuse old depth.
    now[0]=1.4;msg.header.stamp.nanosec=400000000
    SearchNode.on_rgb(node,msg)
    assert node.rgb_depth_supported_at==-np.inf and not node.rgb_candidates
    assert not node.fault


def test_stale_or_failed_rgb_cannot_steer_to_a_previously_confirmed_blue_target():
    node,now=sensor_node()
    SearchNode.reset_rgb(node)
    node.rgb_tracker.confirmed=((1.,2.),1.)
    node.rgb_depth_supported_at=1.
    assert SearchNode.selected_target(node,1.2)==(((1.,2.),1.),'rgb_hsv')
    assert SearchNode.selected_target(node,1.51)==(None,None)
    node.errors['rgb']='invalid calibration'
    assert SearchNode.selected_target(node,1.2)==(None,None)


def test_real_node_startup_diagnostics_publish_rgb_and_opening_status_without_streams():
    import json
    import rclpy
    owns_context=not rclpy.ok()
    if owns_context:rclpy.init(args=[])
    node=None
    try:
        node=SearchNode()
        emitted=[]
        node.status_pub=SimpleNamespace(publish=emitted.append)
        node.publish_diagnostics()
        data=json.loads(emitted[-1].data)
        assert data['phase']=='wait' and data['selected_target'] is None
        assert data['rgb_target']['color_observed'] is True
        assert data['opening']['clearance']['cached_voxels']==0
        assert data['opening']['slab_replay_remaining']==0
        assert data['launch_volume']['enabled'] is False
        assert data['launch_volume']['active'] is False
        assert data['map']['pilot_volume_only_cells']==0
    finally:
        if node is not None:node.destroy_node()
        if owns_context:rclpy.shutdown()


def test_launch_assertion_adapter_waits_for_pilot_then_never_regrants_after_reset():
    grid=DepthGrid(GridParams(extent_m=4.))
    grid.reset((0.,0.),1.)
    volume=LaunchVolume(LaunchVolumeParams(True,(-1.,-1.,-.2,1.,1.,1.8)))
    node=SimpleNamespace(grid=grid,mission=SearchMission(grid),launch_volume=volume,
                         saw_disarmed=True,allow_takeoff=True)
    for ready,armed,offboard,landed in ((False,True,True,True),(True,False,True,True),
                                       (True,True,False,True),(True,True,True,False)):
        SearchNode.update_launch_volume(node,Observation(10.,(0.,0.,0.),0.,ready=ready,
            armed=armed,offboard=offboard,landed=landed))
        assert not volume.grant_attempted
    observation=Observation(10.1,(0.,0.,0.),0.,ready=True,armed=True,offboard=True,landed=True)
    SearchNode.update_launch_volume(node,observation)
    assert volume.status(10.1)['active'] and volume.origin == (0.,0.,0.)
    first_id=volume.provenance_id
    SearchNode.update_launch_volume(node,observation)
    assert volume.status(10.1)['active'] and volume.provenance_id == first_id
    volume.reset()
    SearchNode.update_launch_volume(node,observation)
    assert not volume.status(10.1)['active'] and volume.provenance_id == first_id


@pytest.mark.parametrize('text,enabled', (('',False),('[-1,-1,-0.2,1,1,1.8]',True)))
def test_launch_json_cli_override_stays_string_for_empty_and_enabled_bounds(text,enabled):
    import rclpy
    owns_context=not rclpy.ok()
    if not owns_context:pytest.skip('test needs an isolated ROS argument context')
    rclpy.init(args=['--ros-args','-p',f'launch_volume_bounds_json:="{text}"',
                    '-p',f'pilot_verified_launch:={str(enabled).lower()}'])
    node=None
    try:
        node=SearchNode()
        assert node.launch_volume.p.enabled is enabled
        assert (node.launch_volume.p.relative_bounds is not None) is enabled
        assert not node.launch_volume.active and not node.launch_volume.grant_attempted
    finally:
        if node is not None:node.destroy_node()
        rclpy.shutdown()


def test_adapter_motion_refusal_cannot_promote_takeoff_into_scan():
    import rclpy
    from kestrel_nav.searchnav.mission import Command
    owns_context=not rclpy.ok()
    if owns_context:rclpy.init(args=[])
    node=None
    try:
        node=SearchNode()
        emitted=[]
        node.command_pub=SimpleNamespace(publish=emitted.append)
        node.twist_pub=SimpleNamespace(publish=emitted.append)
        node.now=lambda:2.
        node.readiness=lambda:(True,'test fresh inputs')
        node.position,node.velocity,node.yaw=(.01,0.,0.),(0.,0.,0.),0.
        node.armed=node.offboard=node.landed=node.saw_disarmed=True
        node.state={'armed':True,'offboard_active':True}
        node.qualified_since=-2.
        node.mission.started,node.mission.phase,node.mission.hold_z=1.,'takeoff',.8
        # Reproduce the old adapter trigger without sending anything to ROS:
        # an accidental lateral stationary correction has no route certificate.
        node.mission.tick=lambda observation:Command('takeoff','old hold regression',(.01,0.,.25,0.))
        node.tick()
        assert node.mission.phase=='takeoff' and node.mission.counts['scans']==0
        assert node.last_command['velocity_enu'][:2]==[0.,0.]
    finally:
        if node is not None:node.destroy_node()
        if owns_context:rclpy.shutdown()


def test_adaptive_corridor_adapter_keeps_full_route_certificate_and_new_hit_guard():
    import math
    node,now=sensor_node()
    now[0]=2.
    node.position,node.velocity=(0.,0.,1.),(0.,0.,0.)
    node.samples={'position':(2.,2.),'depth':(2.,2.)}
    K=(320./math.tan(math.radians(40.)),200./math.tan(math.radians(25.)),319.5,199.5)
    R=np.array([[0.,0.,1.],[-1.,0.,0.],[0.,-1.,0.]])
    t=np.array([.116,0.,1.029])
    node.grid.apply_observation(np.ones_like(node.grid.states,bool),np.zeros_like(node.grid.states,bool),2.)
    node.mission.phase='move';node.mission.hold_z=1.
    node.mission.route=[np.zeros(2),np.array([2.,0.])]
    node.mission.route_index=1
    node.last_depth=(2.,np.full((400,640),6.),K,R,t)
    assert SearchNode.corridor_error(node,(.4,0.,0.,0.)) is None
    assert node.corridor_status['plane_distance_m']>.60
    assert node.corridor_status['route_certified']
    # A farther plane must still catch an obstacle ahead of the stopping tube.
    node.last_depth=(2.,np.full((400,640),.45-.116),K,R,t)
    assert 'obstacle' in SearchNode.corridor_error(node,(.4,0.,0.,0.))
    node.last_depth=(2.,np.full((400,640),6.),K,R,t)
    node.grid.reset((0.,0.),1.)
    assert 'certificate' in SearchNode.corridor_error(node,(.4,0.,0.,0.))
    assert not node.corridor_status['route_certified']
