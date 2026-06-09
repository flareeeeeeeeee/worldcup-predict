import numpy as np
from worldcup.tournament import (round_robin_pairs, rank_group, simulate_group,
                                 best_thirds, seed_bracket, play_knockout)


def test_round_robin_six_pairs():
    assert len(round_robin_pairs(["A", "B", "C", "D"])) == 6


def test_rank_group_by_points_then_gd():
    table = {"A": {"pts": 9, "gd": 5, "gf": 7}, "B": {"pts": 6, "gd": 1, "gf": 4},
             "C": {"pts": 3, "gd": 0, "gf": 3}, "D": {"pts": 0, "gd": -6, "gf": 1}}
    assert rank_group(table, h2h={}) == ["A", "B", "C", "D"]


def test_simulate_group_deterministic_winner():
    rng = np.random.default_rng(0)
    teams = ["Strong", "B", "C", "D"]

    def prob_fn(h, a):
        return np.array([1.0, 0.0, 0.0]) if h == "Strong" else (
               np.array([0.0, 0.0, 1.0]) if a == "Strong" else np.array([0.0, 1.0, 0.0]))

    def score_fn(h, a, outcome, rng):
        return (2, 0) if outcome == 0 else (0, 0) if outcome == 1 else (0, 2)

    ranking, _ = simulate_group(teams, prob_fn, score_fn, rng)
    assert ranking[0] == "Strong"


def test_best_thirds_picks_top_k():
    stats = {"X": {"pts": 6, "gd": 3, "gf": 5}, "Y": {"pts": 3, "gd": 0, "gf": 2},
             "Z": {"pts": 1, "gd": -2, "gf": 1}}
    assert best_thirds(stats, k=2) == ["X", "Y"]


def test_play_knockout_returns_a_qualifier():
    rng = np.random.default_rng(1)
    teams = [f"t{i}" for i in range(32)]
    strength = {t: 32 - i for i, t in enumerate(teams)}
    pairs = seed_bracket(teams, strength)

    def adv_fn(h, a, rng):
        return 0 if strength[h] >= strength[a] else 1

    champ = play_knockout(pairs, adv_fn, rng)
    assert champ == "t0"  # strongest always wins under this adv_fn
