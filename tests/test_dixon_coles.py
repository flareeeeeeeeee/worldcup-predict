import numpy as np
import pandas as pd
from worldcup.dixon_coles import fit, score_matrix, wdl_from_matrix


def _toy_results():
    rows = []
    base = pd.Timestamp("2026-01-01")
    # Strong scores lots vs Weak; repeat to give the fitter signal.
    for i in range(12):
        rows.append({"date": base, "home_team": "Strong", "away_team": "Weak",
                     "home_score": 3, "away_score": 0, "tournament": "Friendly", "neutral": True})
        rows.append({"date": base, "home_team": "Weak", "away_team": "Strong",
                     "home_score": 0, "away_score": 2, "tournament": "Friendly", "neutral": True})
    return pd.DataFrame(rows)


CFG = {"dixon_coles": {"half_life_days": 365.0, "max_goals": 10, "recent_window_days": 4000}}


def test_matrix_is_normalized():
    params = fit(_toy_results(), CFG, ref_date=pd.Timestamp("2026-02-01"))
    m = score_matrix(params, "Strong", "Weak", neutral=True, max_goals=10)
    assert abs(m.sum() - 1.0) < 1e-6


def test_stronger_team_more_likely_to_win():
    params = fit(_toy_results(), CFG, ref_date=pd.Timestamp("2026-02-01"))
    m = score_matrix(params, "Strong", "Weak", neutral=True, max_goals=10)
    p_home, p_draw, p_away = wdl_from_matrix(m)
    assert p_home > p_away
