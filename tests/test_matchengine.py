import numpy as np
from worldcup.matchengine import sample_outcome, sample_scoreline, knockout_winner


def test_sample_outcome_deterministic_when_certain():
    rng = np.random.default_rng(0)
    assert sample_outcome(np.array([1.0, 0.0, 0.0]), rng) == 0
    assert sample_outcome(np.array([0.0, 0.0, 1.0]), rng) == 2


def test_scoreline_matches_outcome():
    rng = np.random.default_rng(1)
    m = np.full((4, 4), 1.0 / 16)
    h, a = sample_scoreline(m, outcome=0, rng=rng)  # home win
    assert h > a
    h, a = sample_scoreline(m, outcome=1, rng=rng)  # draw
    assert h == a


def test_knockout_no_draw():
    rng = np.random.default_rng(2)
    res = [knockout_winner(np.array([0.5, 0.4, 0.1]), rng) for _ in range(200)]
    assert set(res) <= {0, 1}
