"""Re-mezcla la probabilidad de campeón a distintos pesos de mercado SIN re-simular.

Uso:
  python reblend.py            # barrido de sensibilidad (0%, 25%, 45%, 70%, 100% mercado)
  python reblend.py 0.30       # mezcla puntual con 30% de peso de mercado

Lee output/probabilities.csv (columna p_champion = modelo puro Monte Carlo) y
data/raw/market_futures.csv. El peso del mercado "permanente" del pipeline vive en
config.yaml -> ensemble.champion.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import pandas as pd
from worldcup.config import load_config
from worldcup.ingest import load_market
from worldcup.market import devig
from worldcup.ensemble import blend_champion
from worldcup.teams import all_teams


def reblend(model_champ: dict, market: dict, market_weight: float) -> dict:
    cfg = {"ensemble": {"champion": {"model": 1.0 - market_weight, "market": market_weight}}}
    return blend_champion(model_champ, market, cfg)


def main():
    cfg = load_config("config.yaml")
    df = pd.read_csv("output/probabilities.csv", index_col="team")
    model_champ = df["p_champion"].to_dict()
    market = devig(load_market(cfg["data"]["market_csv"]))
    market = {t: market.get(t, 0.0) for t in all_teams(cfg)}

    if len(sys.argv) > 1:
        w = float(sys.argv[1])
        blended = reblend(model_champ, market, w)
        top = sorted(blended, key=blended.get, reverse=True)[:12]
        print(f"\nPeso mercado = {w:.0%}  |  Peso modelo = {1 - w:.0%}\n")
        print(f"{'Equipo':<22}{'P(campeón)':>12}")
        for t in top:
            print(f"{t:<22}{blended[t] * 100:>11.1f}%")
    else:
        weights = [0.0, 0.25, 0.45, 0.70, 1.0]
        cols = {w: reblend(model_champ, market, w) for w in weights}
        teams = sorted(model_champ, key=model_champ.get, reverse=True)[:12]
        header = "Equipo".ljust(22) + "".join(f"{int(w*100):>7}%" for w in weights)
        print("\nSensibilidad al peso de mercado (P de campeón):\n")
        print(header)
        print("-" * len(header))
        for t in teams:
            row = t[:21].ljust(22) + "".join(f"{cols[w][t]*100:>7.1f}" for w in weights)
            print(row)
        print("\n(0% = modelo puro · 100% = solo mercado · 45% = default del pipeline)")


if __name__ == "__main__":
    main()
