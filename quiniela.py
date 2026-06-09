"""Genera una quiniela: marcador predicho para CADA partido de fase de grupos del
Mundial 2026, más los clasificados predichos por grupo. Deja columnas vacías de
resultado real para medir accuracy luego (ver score_quiniela.py).

Salida:
  output/quiniela.csv             -> 72 partidos de grupos con marcador predicho
  output/quiniela_qualifiers.csv  -> ganador y subcampeón predichos por grupo

Uso:  python quiniela.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import pandas as pd
from worldcup.config import load_config
from worldcup.ingest import load_results
from worldcup.teams import validate_groups
from worldcup import elo as elo_mod
from worldcup import dixon_coles as dc_mod
from worldcup.matchengine import modal_score_given_outcome
from worldcup.tournament import round_robin_pairs

_TOKENS = {0: "1", 1: "X", 2: "2"}


def main():
    cfg = load_config("config.yaml")
    validate_groups(cfg)
    results = load_results(cfg["data"]["results_csv"])
    groups = cfg["tournament"]["groups"]
    max_goals = cfg["dixon_coles"]["max_goals"]

    print("Ajustando modelos (Elo + Dixon-Coles)...")
    elo_ratings = elo_mod.compute_ratings(results, cfg)
    dc_params = dc_mod.fit(results, cfg, ref_date=results["date"].max())

    rows = []
    match_no = 0
    for gname, gteams in groups.items():
        for h, a in round_robin_pairs(gteams):
            match_no += 1
            mat = dc_mod.score_matrix(dc_params, h, a, neutral=True, max_goals=max_goals)
            ph, pdraw, pa = dc_mod.wdl_from_matrix(mat)
            outcome = int(max(range(3), key=[ph, pdraw, pa].__getitem__))
            hg, ag = modal_score_given_outcome(mat, outcome)
            rows.append({
                "match_no": match_no, "stage": "GROUP", "group": gname,
                "home": h, "away": a,
                "pred_home_goals": hg, "pred_away_goals": ag,
                "pred_result": _TOKENS[outcome],
                "p_home": round(ph, 3), "p_draw": round(pdraw, 3), "p_away": round(pa, 3),
                "actual_home_goals": "", "actual_away_goals": "",
                "result_hit": "", "score_hit": "",
            })
    quiniela = pd.DataFrame(rows)
    quiniela.to_csv("output/quiniela.csv", index=False, encoding="utf-8")

    # Clasificados predichos por grupo: usar p_group_top2 del Monte Carlo si existe,
    # si no, ordenar por Elo dentro del grupo.
    advance = {}
    try:
        probs = pd.read_csv("output/probabilities.csv", index_col="team")
        score_of = probs["p_group_top2"].to_dict()
    except (FileNotFoundError, KeyError):
        score_of = elo_ratings
    qual_rows = []
    for gname, gteams in groups.items():
        ranked = sorted(gteams, key=lambda t: score_of.get(t, 0.0), reverse=True)
        qual_rows.append({"group": gname,
                          "predicted_winner": ranked[0],
                          "predicted_runner_up": ranked[1]})
    pd.DataFrame(qual_rows).to_csv("output/quiniela_qualifiers.csv", index=False, encoding="utf-8")

    print(f"OK: {len(quiniela)} partidos -> output/quiniela.csv")
    print("OK: clasificados por grupo -> output/quiniela_qualifiers.csv")
    print("\nLlena actual_home_goals / actual_away_goals cuando jueguen y corre:")
    print("  python score_quiniela.py")


if __name__ == "__main__":
    main()
