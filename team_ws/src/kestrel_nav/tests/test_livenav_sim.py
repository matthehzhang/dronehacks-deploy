"""Closed-loop course flight in the offline simulator (slow: ~45 s). Run with `-m slow` or directly."""
import pytest

from kestrel_nav.livenav.follower import FollowParams
from kestrel_nav.livenav.mission import LiveMission, MissionParams
from kestrel_nav.livenav.offline_sim import OfflineFlight, SimConfig


@pytest.mark.slow
def test_window_course_lands_offline():
    cfg = SimConfig(start_err_xy=(0.1, -0.1), start_err_yaw_deg=5.0)
    m = LiveMission(MissionParams(landed_z_m=cfg.landed_z), follow_params=FollowParams(v_cruise=0.4))
    res = OfflineFlight(m, cfg).run()
    assert res.result == "landed", res.result
    assert res.t_end < 120.0
    assert res.min_clearance_m >= 0.28
    assert res.land_err_m < 0.3
