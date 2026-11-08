"""RGB blue identity requires calibrated floor/depth geometry and multiple views."""
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from kestrel_nav.searchnav.target import (RGBBlueDetector, HSVParams, decode_bgr,
                                         DetectorParams, TargetTracker)


def scene(color=(255,0,0)):
    bgr=np.full((160,160,3),(20,20,180),np.uint8)
    bgr[65:95,65:95]=color
    K=(100.,100.,79.5,79.5)
    R=np.diag([1.,-1.,-1.])
    t=np.array([0.,0.,2.])
    return bgr,K,R,t,np.full((160,160),2.)


def test_blue_floor_is_detected_but_red_gray_shadow_and_white_are_not():
    detector=RGBBlueDetector()
    for color in ((255,0,0),(220,40,20)):
        image,K,R,t,depth=scene(color)
        result=detector.detect(image,K,R,t,0.,depth,K,R,t)
        assert len(result)==1 and result[0].source=='rgb_hsv'
        assert result[0].reason.startswith('HSV blue') and np.linalg.norm(result[0].xy)<.05
        assert detector.last_mask[80,80]==255 and detector.last_mask[10,10]==0
    for color in ((0,0,255),(20,20,20),(100,100,100),(255,255,255)):
        image,K,R,t,depth=scene(color)
        assert not detector.detect(image,K,R,t,0.,depth,K,R,t)


def test_blue_raised_object_missing_depth_and_missing_floor_cannot_be_end_target():
    image,K,R,t,depth=scene()
    detector=RGBBlueDetector()
    for invalid in (np.full_like(depth,np.nan), np.full_like(depth,1.5)):
        assert not detector.detect(image,K,R,t,0.,invalid,K,R,t)
    assert not detector.detect(image,K,R,t,None,depth,K,R,t)
    assert not detector.detect(image,K,R,t,0.,None,None,None,None)
    assert detector.last_mask.any()  # A visible blue mask alone never promotes a target.


def test_separate_depth_camera_registration_rejects_inconsistent_floor_geometry():
    image,K,R,t,depth=scene()
    # A translated depth camera measures a different range to the same floor.
    td=np.array([.3,0.,3.])
    detector=RGBBlueDetector()
    assert not detector.detect(image,K,R,t,0.,depth,K,R,td)
    result=detector.detect(image,K,R,t,0.,np.full_like(depth,3.),K,R,td)
    assert len(result)==1


def test_rgb_tracker_still_requires_independent_views_and_rejects_ambiguous_blue_patches():
    image,K,R,t,depth=scene()
    candidate=RGBBlueDetector().detect(image,K,R,t,0.,depth,K,R,t)[0]
    tracker=TargetTracker(DetectorParams(qualified=True),source='rgb_hsv',color_observed=True)
    for i in range(6):
        assert tracker.update([candidate],(0.,0.),1.+i*.2) is None
    for i in range(6,12):
        tracker.update([candidate],((i-5)*.06,0.),1.+i*.2)
    assert tracker.confirmed is not None and tracker.status()['color_observed']
    from dataclasses import replace
    other=replace(candidate,xy=(1.,0.))
    for i in range(12,18):
        tracker.update([candidate,other],(i*.06,0.),1.+i*.2)
    assert tracker.confirmed is None


@pytest.mark.parametrize('encoding', ['bgr8','rgb8','bgra8','rgba8'])
def test_color_decoder_honors_channel_order_and_row_padding(encoding):
    channels=4 if 'a8' in encoding else 3
    pixel=[255,0,0] if encoding.startswith('bgr') else [0,0,255]
    if channels==4:pixel+=[255]
    row=bytes(pixel*2+[9,9])
    msg=SimpleNamespace(encoding=encoding,width=2,height=2,step=2*channels+2,data=row*2)
    decoded=decode_bgr(msg)
    assert decoded.shape==(2,2,3) and np.all(decoded==[255,0,0])


def test_mono_cannot_be_silently_reinterpreted_as_color_and_hsv_bounds_validate():
    with pytest.raises(ValueError):decode_bgr(SimpleNamespace(encoding='mono8'))
    with pytest.raises(ValueError):HSVParams(hue_low=140,hue_high=90)
