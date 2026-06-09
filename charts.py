"""Genera los gráficos (PNG) del experimento a partir de los CSV de output/.

Uso:  python charts.py
Salida: output/charts/*.png

Gráficos:
  1. sensitivity_lines.png  - P(campeón) vs peso de mercado (una línea por equipo top)
  2. champion_bars.png      - ranking de campeón: modelo puro vs mezclado con mercado
  3. round_funnel.png       - probabilidad de alcanzar cada ronda, por equipo top
  4. group_heatmap.png      - P(clasificar top-2) de los 48 equipos, grupo por grupo
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from worldcup.config import load_config

OUT = "output/charts"
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#cccccc",
    "axes.grid": True,
    "grid.color": "#e6e6e6",
    "grid.linewidth": 0.8,
    "font.size": 11,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})


def _pct(x, _pos=None):
    return f"{x * 100:.0f}%"


def chart_sensitivity(top_n=8):
    df = pd.read_csv("output/market_sensitivity.csv", index_col="team")
    weights = [0, 25, 45, 70, 100]
    cols = [f"market_{w}" for w in weights]
    teams = df.index[:top_n]
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(teams)))

    fig, ax = plt.subplots(figsize=(10, 6.5))
    for t, c in zip(teams, colors):
        y = df.loc[t, cols].to_numpy(dtype=float)
        ax.plot(weights, y, "-o", color=c, linewidth=2.2, markersize=5)
        ax.annotate(f"  {t}", (weights[-1], y[-1]), color=c, va="center",
                    fontsize=10, fontweight="bold")
    ax.axvline(45, color="#888888", linestyle="--", linewidth=1)
    ax.text(45, ax.get_ylim()[1], " default 45%", color="#888888", va="top", fontsize=9)
    ax.set_xlim(-3, 118)
    ax.set_xticks(weights)
    ax.yaxis.set_major_formatter(_pct)
    ax.set_xlabel("Peso del mercado en la mezcla  (0% = modelo puro · 100% = solo mercado)")
    ax.set_ylabel("Probabilidad de ser campeón")
    ax.set_title("¿Cómo se mueve la sensación? — sensibilidad al mercado")
    fig.savefig(f"{OUT}/sensitivity_lines.png")
    plt.close(fig)


def chart_champion_bars(top_n=16):
    df = pd.read_csv("output/probabilities.csv", index_col="team")
    df = df.sort_values("p_champion", ascending=False).head(top_n).iloc[::-1]
    y = np.arange(len(df))
    h = 0.4

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(y + h / 2, df["p_champion"], height=h, color="#2b6cb0", label="Modelo puro")
    ax.barh(y - h / 2, df["p_champion_blended"], height=h, color="#dd6b20",
            label="Mezclado con mercado")
    ax.set_yticks(y)
    ax.set_yticklabels(df.index)
    ax.xaxis.set_major_formatter(_pct)
    ax.set_xlabel("Probabilidad de ser campeón")
    ax.set_title(f"Candidatos al título 2026 — top {top_n}")
    ax.legend(loc="lower right", frameon=True)
    fig.savefig(f"{OUT}/champion_bars.png")
    plt.close(fig)


def chart_round_funnel(top_n=8):
    df = pd.read_csv("output/probabilities.csv", index_col="team")
    df = df.sort_values("p_champion", ascending=False).head(top_n)
    stages = ["p_r32", "p_r16", "p_qf", "p_sf", "p_final", "p_champion"]
    labels = ["Ronda 32", "Octavos", "Cuartos", "Semis", "Final", "Campeón"]
    x = np.arange(len(stages))
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(df)))

    fig, ax = plt.subplots(figsize=(10, 6.5))
    for (t, row), c in zip(df.iterrows(), colors):
        y = row[stages].to_numpy(dtype=float)
        ax.plot(x, y, "-o", color=c, linewidth=2.2, markersize=5, label=t)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.yaxis.set_major_formatter(_pct)
    ax.set_ylabel("Probabilidad de alcanzar la ronda")
    ax.set_title("Embudo del torneo — qué tan lejos llega cada favorito")
    ax.legend(loc="upper right", ncol=2, frameon=True, fontsize=9)
    fig.savefig(f"{OUT}/round_funnel.png")
    plt.close(fig)


def chart_group_heatmap():
    cfg = load_config("config.yaml")
    groups = cfg["tournament"]["groups"]
    probs = pd.read_csv("output/probabilities.csv", index_col="team")
    top2 = probs["p_group_top2"].to_dict()

    gnames = list(groups.keys())
    grid = np.array([[top2.get(t, 0.0) for t in groups[g]] for g in gnames])

    fig, ax = plt.subplots(figsize=(9, 9))
    im = ax.imshow(grid, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(4))
    ax.set_xticklabels([f"Bombo {i + 1}" for i in range(4)])
    ax.set_yticks(range(len(gnames)))
    ax.set_yticklabels([f"Grupo {g}" for g in gnames])
    for r, g in enumerate(gnames):
        for c, t in enumerate(groups[g]):
            val = grid[r, c]
            txt = f"{t}\n{val * 100:.0f}%"
            ax.text(c, r, txt, ha="center", va="center", fontsize=8,
                    color="white" if val > 0.55 else "#222222")
    ax.set_title("Probabilidad de clasificar (top-2) — grupo por grupo")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_major_formatter(_pct)
    fig.savefig(f"{OUT}/group_heatmap.png")
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    chart_sensitivity()
    chart_champion_bars()
    chart_round_funnel()
    chart_group_heatmap()
    print(f"Gráficos generados en {OUT}/:")
    for f in ["sensitivity_lines", "champion_bars", "round_funnel", "group_heatmap"]:
        print(f"  - {f}.png")


if __name__ == "__main__":
    main()
