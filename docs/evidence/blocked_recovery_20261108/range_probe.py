#!/usr/bin/env python3
"""Offline sensor-contract probe. No ROS, vehicle commands, or policy modifications.

Uses deployed decode_depth, DepthFrame, Mission.on_depth/_clearance/_blocks.
For host filtering it executes the actual HostDepth.process function extracted
from its AST, replacing only transport and stereo matching with supplied disparity.
This tests handling of returns, not the realism/accuracy of stereo matching.
"""
import argparse
import array
import ast
from dataclasses import fields
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace as NS

REPO = Path(__file__).resolve().parents[3]
PKG = REPO / 'team_ws/src/kestrel_nav'
sys.path.insert(0, str(PKG))
import numpy as np
from refnav import depth as D, core as C

H, W = 80, 100
K = (100., 100., 50., 40.)
R = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
T = np.zeros(3)


def host_filtered(z, maximum):
    source = PKG / 'kestrel_nav/host_depth.py'
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'HostDepth')
    function = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'process')
    ns = dict(np=np, time=time, array=array,
              cv2=NS(remap=lambda image, *args: image, INTER_LINEAR=1),
              img_to_gray=lambda message: np.zeros((H, W)),
              Image=lambda: NS(header=NS()))
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), ns)
    messages = []
    f, baseline = 100., .075
    node = NS(rect=dict(swap=False, m1=(None, None), m2=(None, None), f=f, B=baseline,
                        info=NS(header=NS())),
              sgbm=NS(compute=lambda *_: np.full((H, W), 16*f*baseline/z, np.float32)),
              min_m=.3, max_m=maximum, rect_frame='probe',
              pub_depth=NS(publish=messages.append), pub_info=NS(publish=lambda _: None),
              n_out=0, t_proc=0.)
    message = NS(header=NS(stamp=1))
    ns['process'](node, message, message)
    out = messages[-1]
    return D.decode_depth(out.encoding, H, W, out.step, out.data)


def analyze(name, image, profile):
    params = C.Params(route=[(5., 0.)], **profile)
    mission = C.Mission(params)
    mission.phase, mission.leg, mission.leg_heading = 'go', 0, 0.
    mission.leg_len, mission.s = 5., 1.
    vehicle = C.Vehicle(pos=(1., 0., -.8), heading=0.)
    frame = D.DepthFrame(image, K, R, T, np.eye(3), stride=params.depth_stride,
                         border_px=params.depth_border_px)
    clearance = None
    accepted = []
    for stamp in (1., 1.1, 1.2):
        accepted.append(mission.on_depth(stamp, image, K, R, T, (1., 0., 0., 0.), now=stamp))
        clearance = mission._clearance(vehicle)
    coverage, rays = frame.corridor_coverage(0., params.corridor_half_m, params.band_m,
                                            mission._stop_m(mission.s)+params.lead_max_m)
    blocked = None if clearance is None else mission._blocks(clearance, 4.)
    return dict(case=name, valid_fraction=frame.valid_frac, corridor_coverage=coverage,
                corridor_rays=rays, obstacle_point_count=len(frame.points()),
                accepted_frames=accepted, depth_rejection=mission.depth_reject,
                confirmed_clearance_m=('infinity' if clearance is not None and math.isinf(clearance) else clearance),
                stop_threshold_m=mission._stop_m(mission.s), blocked=blocked,
                classification='unknown' if clearance is None else 'obstacle_stop' if blocked else 'clear_for_stop_rule')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    doc = json.loads((PKG/'course/course_new_origin.json').read_text())
    names = {f.name for f in fields(C.Params)}
    profile = {k:v for k,v in doc['refnav_params'].items() if k in names}
    images = []
    for name, value in [('zero', 0.), ('NaN', math.nan), ('infinity', math.inf),
                        ('finite_far_8m', 8.), ('finite_false_near_0.6m', .6),
                        ('finite_false_near_0.9m', .9), ('finite_false_near_1.0m', 1.)]:
        raw = np.full((H,W), value, np.float32)
        images.append((name, D.decode_depth('32FC1', H, W, W*4, raw.tobytes())))
    for z, limit in [(8., 6.), (8., 12.), (13., 12.)]:
        images.append((f'host_depth_{z:g}m_limit_{limit:g}m', host_filtered(z,limit)))
    sentinel = np.full((H,W),65535,np.uint16)
    images.append(('external_uint16_65535_sentinel',D.decode_depth('16UC1',H,W,W*2,sentinel.tobytes())))
    data = dict(method='Synthetic planar depth / injected disparity; actual deployed decoding and core checks. Not a sensor accuracy test.',
                source_sha256={str(p.relative_to(PKG)):hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in [PKG/'refnav/depth.py',PKG/'refnav/core.py',PKG/'kestrel_nav/host_depth.py',PKG/'course/course_new_origin.json']},
                parameters=profile, cases=[analyze(name,img,profile) for name,img in images])
    text=json.dumps(data,indent=2,allow_nan=False)+'\n'
    if args.output:args.output.write_text(text)
    else:print(text,end='')

if __name__=='__main__':main()
