import numpy as np
from worldcup.backtest import log_loss, brier


def test_log_loss_perfect_is_zero():
    assert log_loss([np.array([1.0, 0.0, 0.0])], [0]) < 1e-9


def test_brier_perfect_is_zero():
    assert brier([np.array([0.0, 1.0, 0.0])], [1]) < 1e-9


def test_log_loss_penalizes_wrong():
    bad = log_loss([np.array([0.01, 0.01, 0.98])], [0])
    good = log_loss([np.array([0.98, 0.01, 0.01])], [0])
    assert bad > good
