import pandas as pd
from worldcup.elo import compute_ratings, match_probabilities

CFG = {
    "elo": {"initial_rating": 1500.0, "home_advantage": 65.0, "k_base": 40.0,
            "tournament_weights": {"Friendly": 1.0}, "default_tournament_weight": 1.0},
    "ensemble": {"draw_param": 0.28},
}


def test_winner_gains_rating():
    df = pd.DataFrame([{
        "date": pd.Timestamp("2025-01-01"), "home_team": "A", "away_team": "B",
        "home_score": 3, "away_score": 0, "tournament": "Friendly", "neutral": True,
    }])
    r = compute_ratings(df, CFG)
    assert r["A"] > 1500.0 and r["B"] < 1500.0
    assert round(r["A"] - 1500.0, 6) == round(1500.0 - r["B"], 6)  # zero-sum


def test_match_probabilities_sum_to_one_and_favor_stronger():
    p_home, p_draw, p_away = match_probabilities(1800, 1500, neutral=True, cfg=CFG)
    assert abs(p_home + p_draw + p_away - 1.0) < 1e-9
    assert p_home > p_away
