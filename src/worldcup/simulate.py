import numpy as np
import pandas as pd
from worldcup import elo as elo_mod
from worldcup import dixon_coles as dc_mod
from worldcup.ensemble import match_wdl
from worldcup.matchengine import sample_scoreline, knockout_winner
from worldcup.tournament import simulate_group, best_thirds, seed_bracket


def build_match_probs(teams, elo_ratings, dc_params, cfg):
    max_goals = cfg["dixon_coles"]["max_goals"]
    probs, matrices = {}, {}
    for h in teams:
        for a in teams:
            if h == a:
                continue
            e_wdl = elo_mod.match_probabilities(
                elo_ratings.get(h, 1500.0), elo_ratings.get(a, 1500.0), neutral=True, cfg=cfg)
            mat = dc_mod.score_matrix(dc_params, h, a, neutral=True, max_goals=max_goals)
            d_wdl = dc_mod.wdl_from_matrix(mat)
            probs[(h, a)] = match_wdl(e_wdl, d_wdl, cfg)
            matrices[(h, a)] = mat
    return probs, matrices


def _strength(elo_ratings, teams):
    return {t: elo_ratings.get(t, 1500.0) for t in teams}


def _simulate_once(groups, probs, matrices, strength, rng, track):
    qualifiers, third_stats, top2 = [], {}, []
    for gname, gteams in groups.items():
        def prob_fn(h, a):
            return probs[(h, a)]

        def score_fn(h, a, outcome, rng):
            return sample_scoreline(matrices[(h, a)], outcome, rng)

        ranking, table = simulate_group(gteams, prob_fn, score_fn, rng)
        qualifiers.extend(ranking[:2])
        top2.extend(ranking[:2])
        third = ranking[2]
        third_stats[third] = table[third]
    qualifiers.extend(best_thirds(third_stats, k=8))
    for t in top2:
        track["p_group_top2"][t] += 1
    pairs = seed_bracket(qualifiers, strength)
    for t in qualifiers:
        track["p_r32"][t] += 1
    # resolve knockout round by round, tracking how far each team goes.
    # 32 qualifiers -> winners are R16 (16), QF (8), SF (4), finalists (2), champion (1).
    round_labels = ["p_r16", "p_qf", "p_sf", "p_final", "p_champion"]

    def adv_fn(h, a, rng):
        return knockout_winner(probs[(h, a)], rng)

    current = pairs
    li = 0
    while True:
        winners = [h if adv_fn(h, a, rng) == 0 else a for h, a in current]
        if li < len(round_labels):
            for w in winners:
                track[round_labels[li]][w] += 1
        li += 1
        if len(winners) == 1:
            return winners[0]
        current = [(winners[i], winners[i + 1]) for i in range(0, len(winners), 2)]


def run_monte_carlo(cfg, elo_ratings, dc_params) -> pd.DataFrame:
    groups = cfg["tournament"]["groups"]
    teams = [t for g in groups.values() for t in g]
    probs, matrices = build_match_probs(teams, elo_ratings, dc_params, cfg)
    strength = _strength(elo_ratings, teams)
    cols = ["p_group_top2", "p_r32", "p_r16", "p_qf", "p_sf", "p_final", "p_champion"]
    track = {c: {t: 0 for t in teams} for c in cols}
    rng = np.random.default_rng(cfg["simulation"]["seed"])
    n = cfg["simulation"]["n_sims"]
    for _ in range(n):
        _simulate_once(groups, probs, matrices, strength, rng, track)
    df = pd.DataFrame({c: pd.Series(track[c]) for c in cols}) / n
    df.index.name = "team"
    return df.sort_values("p_champion", ascending=False)
