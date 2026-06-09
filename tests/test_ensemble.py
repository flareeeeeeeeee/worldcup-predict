import numpy as np
from worldcup.ensemble import log_opinion_pool, match_wdl, blend_champion

CFG = {"ensemble": {"match": {"elo": 0.5, "dc": 0.5},
                    "champion": {"model": 0.5, "market": 0.5}}}


def test_pool_sums_to_one():
    out = log_opinion_pool([np.array([0.5, 0.3, 0.2]), np.array([0.4, 0.4, 0.2])], [0.5, 0.5])
    assert abs(out.sum() - 1.0) < 1e-9


def test_match_wdl_blends():
    out = match_wdl((0.6, 0.2, 0.2), (0.4, 0.3, 0.3), CFG)
    assert abs(out.sum() - 1.0) < 1e-9
    assert out[0] > out[2]


def test_blend_champion_union_of_teams():
    out = blend_champion({"A": 0.6, "B": 0.4}, {"A": 0.5, "B": 0.5}, CFG)
    assert abs(sum(out.values()) - 1.0) < 1e-9
