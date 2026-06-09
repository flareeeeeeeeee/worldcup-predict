import copy
import numpy as np
import pandas as pd
from worldcup import elo as elo_mod
from worldcup import dixon_coles as dc_mod
from worldcup.ensemble import match_wdl


def _result_index(hg, ag):
    return 0 if hg > ag else (1 if hg == ag else 2)


def log_loss(probs, outcomes) -> float:
    eps = 1e-12
    return float(-np.mean([np.log(max(p[o], eps)) for p, o in zip(probs, outcomes)]))


def brier(probs, outcomes) -> float:
    total = 0.0
    for p, o in zip(probs, outcomes):
        target = np.zeros(3)
        target[o] = 1.0
        total += np.sum((p - target) ** 2)
    return float(total / len(outcomes))


def _holdout_split(results, cfg):
    split = results["date"].max() - pd.Timedelta(days=cfg["backtest"]["holdout_days"])
    train = results[results["date"] <= split]
    test = results[results["date"] > split]
    return train, test, split


def evaluate(results, cfg) -> dict:
    train, test, split = _holdout_split(results, cfg)
    elo_ratings = elo_mod.compute_ratings(train, cfg)
    dc_params = dc_mod.fit(train, cfg, ref_date=split)
    max_goals = cfg["dixon_coles"]["max_goals"]
    probs, outcomes = [], []
    for _, m in test.iterrows():
        h, a, neutral = m["home_team"], m["away_team"], bool(m["neutral"])
        e_wdl = elo_mod.match_probabilities(
            elo_ratings.get(h, 1500.0), elo_ratings.get(a, 1500.0), neutral, cfg)
        mat = dc_mod.score_matrix(dc_params, h, a, neutral, max_goals)
        d_wdl = dc_mod.wdl_from_matrix(mat)
        probs.append(match_wdl(e_wdl, d_wdl, cfg))
        outcomes.append(_result_index(m["home_score"], m["away_score"]))
    return {"log_loss": log_loss(probs, outcomes),
            "brier": brier(probs, outcomes), "n": len(outcomes)}


def calibrate_match_weights(results, cfg, grid=None):
    grid = grid if grid is not None else [i / 10 for i in range(1, 10)]
    best_w, best_score = None, float("inf")
    for w in grid:
        trial = copy.deepcopy(cfg)
        trial["ensemble"]["match"] = {"elo": w, "dc": 1.0 - w}
        score = evaluate(results, trial)["log_loss"]
        if score < best_score:
            best_score, best_w = score, {"elo": w, "dc": 1.0 - w}
    return best_w, best_score
