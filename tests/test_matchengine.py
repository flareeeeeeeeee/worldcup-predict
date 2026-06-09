import numpy as np
from worldcup.matchengine import (sample_outcome, sample_scoreline, knockout_winner,
                                  most_likely_score, modal_score_given_outcome)


def test_most_likely_score_picks_argmax():
    m = np.zeros((3, 3))
    m[2, 1] = 0.9
    m[0, 0] = 0.1
    assert most_likely_score(m) == (2, 1)


def test_modal_score_given_outcome_stays_consistent():
    m = np.zeros((3, 3))
    m[1, 1] = 0.40  # draw is the single most likely score
    m[2, 0] = 0.35  # but home win is a plausible outcome
    m[0, 2] = 0.25
    assert modal_score_given_outcome(m, 0) == (2, 0)  # home win
    assert modal_score_given_outcome(m, 1) == (1, 1)  # draw
    assert modal_score_given_outcome(m, 2) == (0, 2)  # away win


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
