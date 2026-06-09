import pandas as pd


def _mov_multiplier(goal_diff: int) -> float:
    g = abs(goal_diff)
    if g <= 1:
        return 1.0
    if g == 2:
        return 1.5
    return (11.0 + g) / 8.0


def compute_ratings(results: pd.DataFrame, cfg: dict) -> dict:
    e = cfg["elo"]
    ratings: dict[str, float] = {}
    init = e["initial_rating"]
    weights = e["tournament_weights"]
    default_w = e["default_tournament_weight"]
    for _, m in results.sort_values("date").iterrows():
        h, a = m["home_team"], m["away_team"]
        ratings.setdefault(h, init)
        ratings.setdefault(a, init)
        ha = 0.0 if m["neutral"] else e["home_advantage"]
        diff = (ratings[h] + ha) - ratings[a]
        exp_home = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))
        gd = int(m["home_score"]) - int(m["away_score"])
        if gd > 0:
            score_home = 1.0
        elif gd < 0:
            score_home = 0.0
        else:
            score_home = 0.5
        k = e["k_base"] * weights.get(m.get("tournament", ""), default_w) * _mov_multiplier(gd)
        delta = k * (score_home - exp_home)
        ratings[h] += delta
        ratings[a] -= delta
    return ratings


def match_probabilities(r_home: float, r_away: float, neutral: bool, cfg: dict):
    ha = 0.0 if neutral else cfg["elo"]["home_advantage"]
    diff = (r_home + ha) - r_away
    e = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))
    d0 = cfg["ensemble"]["draw_param"]
    p_draw = d0 * (1.0 - abs(2.0 * e - 1.0))
    p_home = e * (1.0 - p_draw)
    p_away = (1.0 - e) * (1.0 - p_draw)
    return p_home, p_draw, p_away
