# World Cup 2026 — Predicción estadística (worldcup-predict)

**Fecha:** 2026-06-09
**Autor:** Juan Castillo + Claude (Opus 4.8)
**Estado:** Diseño aprobado — pendiente plan de implementación

## Contexto y objetivo

Experimento competitivo: generar la **mejor predicción estadística posible del Mundial 2026** (bracket completo + campeón) para competir contra otras IAs (Gemini y amigos). No se busca "verdad" sino una predicción **calibrada y defendible**. La ventaja diferencial frente a otras IAs es la **calibración medible** (backtest con log-loss/Brier) y el **anclaje al mercado**, en vez de razonamiento cualitativo.

### Decisiones tomadas (brainstorming)

| Decisión | Elección |
|---|---|
| Objetivo | Bracket completo + campeón |
| Datos primarios | `martj42/international_results` (CSV GitHub) + football-data.org para fixture/grupos oficiales |
| Modelo | Ensemble: Elo + Dixon-Coles + señal de mercado |
| Mercado | Incluido (cuotas de campeón de-vig) |
| Entregable | Reporte Markdown + tablas + bracket ASCII |
| Stack / ubicación | Python 3.11 en `D:\Development\worldcup-predict` |

## Arquitectura

Pipeline de 4 etapas; cada módulo es independiente y testeable.

```
worldcup-predict/
├── data/raw/              # CSVs descargados sin procesar
├── data/processed/        # datasets limpios para los modelos
├── src/
│   ├── ingest.py          # descarga datos → CSV (martj42 + football-data.org)
│   ├── elo.py             # ranking Elo de selecciones desde histórico
│   ├── dixon_coles.py     # modelo Poisson ataque/defensa con decaimiento temporal
│   ├── market.py          # cuotas → probabilidades implícitas de-vig
│   ├── ensemble.py        # log-opinion pool de las 3 señales
│   ├── simulate.py        # Monte Carlo del torneo (formato 48)
│   └── report.py          # genera reporte MD + bracket ASCII
├── tests/                 # backtest + tests unitarios
├── config.yaml            # pesos ensemble, params, N simulaciones
└── output/                # prediction.md, tablas, bracket
```

## Datos

### Fuente primaria — `martj42/international_results`
- CSV directo desde GitHub (raw), sin API key.
- `results.csv`: todos los internacionales (fecha, equipos, marcador, torneo, ciudad, país, neutral).
- Usos: entrenar Elo (histórico completo) y Dixon-Coles (ventana reciente con decaimiento, énfasis últimos 6 meses).

### football-data.org (API gratis del usuario)
- Solo para **fixture y composición oficial de grupos** del Mundial 2026 (competición `WC`).
- Rate-limit 10 req/min: cachear respuesta en `data/raw/`.
- Si la cobertura de WC2026 en el tier gratis falla, fallback: hardcodear el sorteo oficial en `config.yaml`.

### Mercado
- Capturar cuotas de campeón (futures) de un agregador público.
- De-vig: convertir cuotas a probabilidades implícitas y normalizar para quitar el margen de la casa.
- Guardar snapshot con fecha en `data/raw/market_futures.csv` (reproducibilidad).

## Modelos

### Elo (`elo.py`)
- Rating por selección, actualizado partido a partido sobre todo el histórico de `martj42`.
- Fórmula estilo eloratings.net: K ponderado por importancia del torneo, ventaja de local (salvo sede neutral), multiplicador por margen de victoria.
- Salida: rating actual por selección → P(victoria) por enfrentamiento.

### Dixon-Coles (`dixon_coles.py`)
- Estima fuerza de **ataque** y **defensa** por equipo desde goles de partidos internacionales.
- **Decaimiento temporal exponencial**: partidos recientes pesan más (énfasis últimos 6 meses).
- Corrección Dixon-Coles (parámetro τ) para dependencia en marcadores bajos (0-0, 1-0, 0-1, 1-1).
- Salida: tasas esperadas de goles por equipo → matriz de marcadores → P(victoria/empate/derrota) y marcador esperado.

### Mercado (`market.py`)
- Probabilidades implícitas de-vig como ancla de realidad (sabiduría colectiva).

## Ensemble y simulación

### Ensemble (`ensemble.py`) — dos niveles
El mercado gratuito solo da cuotas de **campeón** (no cuotas por cada partido), así que se mezcla en dos niveles distintos:

- **Nivel partido (Elo + Dixon-Coles):** log-opinion pool — media geométrica ponderada de los vectores P(victoria/empate/derrota) de Elo y DC, renormalizada. Arranque: Elo 0.55 / DC 0.45 (relativo). Esto alimenta la simulación Monte Carlo.
- **Nivel campeón (modelo + mercado):** la distribución de campeón que sale del Monte Carlo se mezcla con las probabilidades de-vig del mercado mediante log-opinion pool ponderado. Arranque: modelo 0.55 / mercado 0.45.
- Todos los pesos viven en `config.yaml` y se **calibran por backtest** (a nivel partido) y por concordancia con mercado (a nivel campeón).

### Simulación Monte Carlo (`simulate.py`)
- **Formato real WC2026**: 48 equipos → 12 grupos de 4 → clasifican 2 primeros de cada grupo + 8 mejores terceros → ronda de 32 → octavos → cuartos → semis → final.
- Fase de grupos: round-robin con puntos y **tiebreakers oficiales FIFA**; ranking de mejores terceros.
- Knockouts: si empate tras 90', simular prórroga y luego penales (resultado ponderado por fuerza relativa).
- N = 50.000 simulaciones. Frecuencias → probabilidades de avanzar cada ronda, llegar a final, ser campeón.

## Calibración (la credencial estadística)

- **Backtest**: reservar partidos internacionales recientes (hold-out), predecir con el ensemble y medir **log-loss** y **Brier score**.
- Tunear pesos del ensemble para minimizar el error en el hold-out.
- Reportar las métricas en el entregable: convierte "opinión" en "probabilidad calibrada" — el argumento para ganarle a otras IAs.

## Entregable

`output/prediction.md`:
1. Metodología + métricas de calibración del backtest.
2. Tabla ranking: P(campeón), P(final), P(semis) por equipo.
3. Probabilidades de clasificación por grupo.
4. **Bracket ASCII** del camino modal simulado.
5. Insights: sorpresas vs mercado, dark horses, sesgos detectados.
6. Delta **puro (Elo+DC) vs ensemble-con-mercado**: cuánto aporta el modelo propio sobre el mercado.

## Fuera de alcance (YAGNI)

- Dashboard HTML interactivo (se eligió reporte MD).
- xG / lineups / datos por jugador (no disponibles gratis con fiabilidad para selecciones).
- ML/gradient boosting (riesgo de overfitting con pocos partidos de selecciones).
- Actualización en vivo durante el torneo (predicción es pre-torneo).

## Riesgos

- **Cobertura de datos:** football-data.org tier gratis puede no exponer bien WC2026 → fallback a grupos hardcodeados.
- **Cuotas:** scraping de futures puede romperse → permitir entrada manual de cuotas en CSV.
- **Formato 48 nuevo:** los tiebreakers y el slotting de octavos deben seguir el reglamento oficial FIFA 2026 con cuidado.
