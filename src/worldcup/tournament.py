from itertools import combinations
import numpy as np
from worldcup.matchengine import sample_outcome


def round_robin_pairs(teams: list) -> list:
    return [(h, a) for h, a in combinations(teams, 2)]


def rank_group(table: dict, h2h: dict) -> list:
    def key(t):
        s = table[t]
        return (s["pts"], s["gd"], s["gf"], h2h.get(t, 0))
    return sorted(table.keys(), key=key, reverse=True)


def simulate_group(teams, prob_fn, score_fn, rng):
    table = {t: {"pts": 0, "gd": 0, "gf": 0} for t in teams}
    h2h_pts = {t: 0 for t in teams}
    results = {}
    for h, a in round_robin_pairs(teams):
        wdl = prob_fn(h, a)
        outcome = sample_outcome(wdl, rng)
        hg, ag = score_fn(h, a, outcome, rng)
        results[(h, a)] = (hg, ag)
        table[h]["gf"] += hg
        table[a]["gf"] += ag
        table[h]["gd"] += hg - ag
        table[a]["gd"] += ag - hg
        if hg > ag:
            table[h]["pts"] += 3
            h2h_pts[h] += 3
        elif hg < ag:
            table[a]["pts"] += 3
            h2h_pts[a] += 3
        else:
            table[h]["pts"] += 1
            table[a]["pts"] += 1
            h2h_pts[h] += 1
            h2h_pts[a] += 1
    ranking = rank_group(table, h2h_pts)
    return ranking, table


def best_thirds(third_stats: dict, k: int = 8) -> list:
    ordered = sorted(third_stats.keys(),
                     key=lambda t: (third_stats[t]["pts"], third_stats[t]["gd"], third_stats[t]["gf"]),
                     reverse=True)
    return ordered[:k]


def seed_bracket(qualifiers: list, strength: dict) -> list:
    seeded = sorted(qualifiers, key=lambda t: strength.get(t, 0.0), reverse=True)
    n = len(seeded)
    return [(seeded[i], seeded[n - 1 - i]) for i in range(n // 2)]


def play_knockout(pairs, adv_fn, rng) -> str:
    current = list(pairs)
    while len(current) >= 1:
        winners = []
        for h, a in current:
            winners.append(h if adv_fn(h, a, rng) == 0 else a)
        if len(winners) == 1:
            return winners[0]
        current = [(winners[i], winners[i + 1]) for i in range(0, len(winners), 2)]
    return current[0][0]
