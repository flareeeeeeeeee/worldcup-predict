# World Cup 2026 — Predicción (Opus 4.8)

## Methodology
Ensemble de tres señales: Elo de selecciones (histórico completo), Dixon-Coles de goles
(ventana reciente con decaimiento temporal) combinados por log-opinion pool a nivel partido,
y cuotas de mercado de-vig mezcladas a nivel campeón. Torneo simulado 48 equipos
vía Monte Carlo con el formato real WC2026.

## Calibration (credencial estadística)
- Holdout log-loss: **0.8957**
- Holdout Brier: **0.5253**
- Partidos evaluados: 328

## Champion probabilities
| Team | Model P(champ) | Market-blended | P(final) | P(SF) |
|---|---|---|---|---|
| Argentina | 19.7% | 14.4% | 25.7% | 35.4% |
| Spain | 16.4% | 17.1% | 21.8% | 31.0% |
| France | 5.8% | 9.4% | 9.1% | 16.2% |
| Brazil | 5.5% | 7.0% | 9.3% | 25.0% |
| Portugal | 5.2% | 7.1% | 12.4% | 23.2% |
| England | 5.2% | 7.7% | 8.6% | 15.5% |
| Colombia | 4.5% | 3.4% | 7.6% | 21.7% |
| Japan | 4.4% | 2.7% | 11.5% | 20.8% |
| Norway | 4.4% | 3.5% | 11.4% | 20.3% |
| Ecuador | 4.2% | 2.4% | 7.6% | 21.1% |
| Netherlands | 4.1% | 4.4% | 11.3% | 20.0% |
| Germany | 3.9% | 5.0% | 7.1% | 21.4% |
| Morocco | 3.6% | 2.7% | 9.8% | 18.0% |
| Belgium | 2.6% | 2.5% | 7.8% | 15.7% |
| Switzerland | 2.3% | 1.9% | 7.7% | 15.7% |
| Turkey | 1.6% | 1.3% | 5.4% | 11.4% |

## Pure model vs market
| Team | Model | Blended | Δ (market pull) |
|---|---|---|---|
| Argentina | 19.7% | 14.4% | -5.2pp |
| France | 5.8% | 9.4% | +3.7pp |
| England | 5.2% | 7.7% | +2.5pp |
| Portugal | 5.2% | 7.1% | +1.9pp |
| Ecuador | 4.2% | 2.4% | -1.8pp |
| Japan | 4.4% | 2.7% | -1.7pp |
| Brazil | 5.5% | 7.0% | +1.5pp |
| Colombia | 4.5% | 3.4% | -1.1pp |
| Germany | 3.9% | 5.0% | +1.0pp |
| Morocco | 3.6% | 2.7% | -0.9pp |

## Modal bracket
```
FINAL: Argentina
CHAMPION: Argentina
```
