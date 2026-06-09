def _pct(x):
    return f"{100.0 * x:.1f}%"


def champion_table(df, blended, top: int = 16) -> str:
    rows = ["| Team | Model P(champ) | Market-blended | P(final) | P(SF) |",
            "|---|---|---|---|---|"]
    for team in df.head(top).index:
        rows.append(f"| {team} | {_pct(df.loc[team, 'p_champion'])} | "
                    f"{_pct(blended.get(team, 0.0))} | {_pct(df.loc[team, 'p_final'])} | "
                    f"{_pct(df.loc[team, 'p_sf'])} |")
    return "\n".join(rows)


def ascii_bracket(path: dict) -> str:
    # path: {"R16": [...], "QF": [...], "SF": [...], "FINAL": "Team", "CHAMP": "Team"}
    lines = []
    for rnd in ["R16", "QF", "SF"]:
        if rnd in path:
            lines.append(f"{rnd}: " + "  |  ".join(path[rnd]))
    if "FINAL" in path:
        lines.append("FINAL: " + " vs ".join(path["FINAL"]) if isinstance(path["FINAL"], list)
                     else f"FINAL: {path['FINAL']}")
    if "CHAMP" in path:
        lines.append(f"CHAMPION: {path['CHAMP']}")
    return "```\n" + "\n".join(lines) + "\n```"


def pure_vs_market_delta(model: dict, blended: dict, top: int = 10) -> str:
    teams = sorted(set(model) | set(blended),
                   key=lambda t: abs(blended.get(t, 0.0) - model.get(t, 0.0)), reverse=True)
    rows = ["| Team | Model | Blended | Δ (market pull) |", "|---|---|---|---|"]
    for t in teams[:top]:
        d = blended.get(t, 0.0) - model.get(t, 0.0)
        rows.append(f"| {t} | {_pct(model.get(t, 0.0))} | {_pct(blended.get(t, 0.0))} | {d*100:+.1f}pp |")
    return "\n".join(rows)


def build_report(metrics, df, blended, modal_bracket_md, deltas_md) -> str:
    return f"""# World Cup 2026 — Predicción (Opus 4.8)

## Methodology
Ensemble de tres señales: Elo de selecciones (histórico completo), Dixon-Coles de goles
(ventana reciente con decaimiento temporal) combinados por log-opinion pool a nivel partido,
y cuotas de mercado de-vig mezcladas a nivel campeón. Torneo simulado {df.shape[0]} equipos
vía Monte Carlo con el formato real WC2026.

## Calibration (credencial estadística)
- Holdout log-loss: **{metrics['log_loss']:.4f}**
- Holdout Brier: **{metrics['brier']:.4f}**
- Partidos evaluados: {metrics['n']}

## Champion probabilities
{champion_table(df, blended)}

## Pure model vs market
{deltas_md}

## Modal bracket
{modal_bracket_md}
"""
