import numpy as np
import pandas as pd
from worldcup.simulate import run_monte_carlo


def _cfg():
    groups = {chr(65 + i): [f"t{i}{j}" for j in range(4)] for i in range(12)}
    return {
        "tournament": {"groups": groups},
        "simulation": {"n_sims": 50, "seed": 1},
        "elo": {"home_advantage": 0.0},
        "ensemble": {"match": {"elo": 0.5, "dc": 0.5}, "draw_param": 0.28},
        "dixon_coles": {"max_goals": 6},
    }


def test_champion_probs_sum_to_one():
    cfg = _cfg()
    teams = [f"t{i}{j}" for i in range(12) for j in range(4)]
    elo = {t: 1500.0 for t in teams}
    dc = {"attack": {t: 0.0 for t in teams}, "defence": {t: 0.0 for t in teams},
          "home_adv": 0.0, "rho": 0.0, "teams": teams}
    df = run_monte_carlo(cfg, elo, dc)
    assert abs(df["p_champion"].sum() - 1.0) < 1e-9
    assert (df["p_champion"] >= 0).all()
