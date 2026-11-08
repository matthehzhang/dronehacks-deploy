"""Landing-target detection (colour + depth) and final-waypoint tracking; vertical probe."""
import math

import numpy as np
import pytest

from coursenav.target import (TargetTracker, depth_body_points, hsv_mask, largest_blob, level_to_course,
                              locate_target)

# optical (x right, y down, z forward) -> body FLU
R_BODY_CAM = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
T_BODY_CAM = np.array([.1, 0., .05])
K = (320., 320., 320., 200.)          # 640x400, ~90 deg wide


def render(pad_xy, pad_size=.62, height=.7, W=640, H=400, pad_z=0.):
    """Synthetic colour + depth of a flat blue square on a grey floor, level camera."""
    fx, fy, cx, cy = K
    v, u = np.mgrid[0:H, 0:W]
    ray_cam = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u, float)], -1)
    ray = ray_cam @ R_BODY_CAM.T
    o = T_BODY_CAM
    with np.errstate(divide='ignore', invalid='ignore'):
        s = np.where(ray[..., 2] < -1e-6, -(height + o[2]) / ray[..., 2], np.nan)
    p = o + s[..., None] * ray
    on_pad = (np.abs(p[..., 0] - pad_xy[0]) <= pad_size / 2) & (np.abs(p[..., 1] - pad_xy[1]) <= pad_size / 2)
    rgb = np.full((H, W, 3), 128, np.uint8)
    rgb[on_pad & np.isfinite(s)] = (60, 168, 240)                 # the sim pad's measured colour (RGB)
    depth = (s * ray_cam[..., 2]).astype(np.float32)               # optical z
    if pad_z:
        depth[on_pad] *= (height + o[2] - pad_z) / (height + o[2])
    return rgb, depth


def test_hsv_picks_the_pad_colour_not_grey_floor_or_red():
    img = np.array([[[60, 168, 240], [128, 128, 128], [240, 40, 40]]], np.uint8)
    assert hsv_mask(img)[0].tolist() == [True, False, False]


@pytest.mark.parametrize('pad', [(2.2, .3), (2.6, -.5), (1.95, 0.)])
def test_depth_projected_target_position(pad):
    rgb, depth = render(pad)
    blob, n = largest_blob(hsv_mask(rgb))
    assert blob is not None
    pts = depth_body_points(depth, K, R_BODY_CAM, T_BODY_CAM, stride=2)
    xy, info = locate_target(blob, K, R_BODY_CAM, T_BODY_CAM, pts, np.eye(3), .7)
    assert info['method'] == 'depth' and np.linalg.norm(xy - pad) < .06


def test_floor_ray_fallback_without_depth_and_off_floor_rejection():
    rgb, depth = render((2.3, .2))
    blob, _ = largest_blob(hsv_mask(rgb))
    xy, info = locate_target(blob, K, R_BODY_CAM, T_BODY_CAM, np.zeros((0, 3)), np.eye(3), .7)
    assert info['method'] == 'floor_ray' and np.linalg.norm(xy - (2.3, .2)) < .08
    rgb, depth = render((2.3, .2), pad_z=.5)                       # a blue thing 0.5 m above the floor
    pts = depth_body_points(depth, K, R_BODY_CAM, T_BODY_CAM, stride=2)
    xy, info = locate_target(largest_blob(hsv_mask(rgb))[0], K, R_BODY_CAM, T_BODY_CAM, pts, np.eye(3), .7)
    assert xy is None and info['reject'] == 'not_on_floor'


def test_level_to_course_pose():
    assert np.allclose(level_to_course((1., 0.), (2., 3., math.pi / 2)), (2., 4.))


def test_tracker_needs_agreement_follows_a_moved_target_and_bounds_the_shift():
    tr = TargetTracker((-2.04, -.79), max_shift_m=2.5)
    tr.observe(0., (-1.75, -.45))
    tr.observe(.25, (-1.76, -.44))
    assert tr.estimate is None                                     # two detections are not enough
    tr.observe(.5, (-1.74, -.46))
    assert np.linalg.norm(tr.estimate - (-1.75, -.45)) < .02
    for k in range(6):                                             # the target is moved again
        tr.observe(5. + k * .25, (-1.4, -.2))
    assert np.linalg.norm(tr.estimate - (-1.4, -.2)) < .02
    assert not tr.observe(9., (2.5, 2.5))                          # implausibly far from the declared pad
    inside = TargetTracker((0., 0.), inside=lambda xy: abs(xy[0]) < 1)
    assert not inside.observe(0., (1.5, 0.))


def test_navigator_moves_final_waypoint_and_replans_on_final_leg():
    from types import SimpleNamespace as NS
    from coursenav.nav_node import NavNode
    legs = [dict(stop=[0., 0.]), dict(stop=[-2.04, -.79])]
    tr = TargetTracker(legs[-1]['stop'])
    for k in range(3):
        tr.observe(k * .2, (-1.75, -.45))
    fake = NS(target=tr, nav_legs=legs, target_declared=np.array(legs[-1]['stop']), target_updates=0,
              depth_nav=NS(path='old'), get_logger=lambda: NS(info=lambda *a: None))
    NavNode._apply_target(fake, 1)
    assert np.allclose(legs[-1]['stop'], (-1.75, -.45)) and fake.depth_nav.path is None


def test_full_sweep_without_route_probes_a_new_height():
    from refnav.core import Mission, Params, Vehicle
    from refnav.vertical import VerticalCentring
    m = Mission(Params(track_mode='path', local_horizon=True))
    m.phase = 'go'; m.leg = 0; m.targets = [(4., 0.)]; m.waypoints_ne = [(4., 0.)]
    m.sp = [0., 0., -.7, 0.]; m.home = [0., 0., 0., 0.]
    m.t_started = m.t_phase = 0.; m.t_last = 1.
    m.health.bad_for = lambda now: 0.; m.offboard_seen = False
    m.vc = VerticalCentring(.7, .37, .24)
    t = 2.
    for _ in range(1500):
        t += .05
        m.depth_t = t
        m.step(t, Vehicle(pos=(0., 0., m.sp[2]), heading=m.sp[3], pos_ok=True, vel=(0., 0., 0.)))
        if m.vc.probes:
            break
    assert m.phase == 'search' and m.vc.probes == 1 and abs(m.vc.cruise - .95) < 1e-6
