# worldcup-predict ⚽📊

Predicción probabilística **calibrada** del Mundial de Fútbol 2026, como entrada a una
competencia informal de predicción entre IAs (Claude Opus 4.8 vs. Gemini y otras).

Ensemble de **Elo** + **Dixon-Coles** + **señal de mercado**, resuelto con una simulación
**Monte Carlo de 50 000 torneos** sobre el formato real de 48 equipos.

> 📄 **El writeup completo del experimento está en [`PAPER.md`](PAPER.md).**

## Resultado (top 6)

| Equipo | Modelo P(campeón) | Mezclado c/ mercado |
|---|---|---|
| Argentina | 19.7 % | 14.4 % |
| España | 16.4 % | **17.1 %** |
| Francia | 5.8 % | 9.4 % |
| Brasil | 5.5 % | 7.0 % |
| Portugal | 5.2 % | 7.1 % |
| Inglaterra | 5.2 % | 7.7 % |

Calibración en *backtest* (n = 328): **log-loss 0.8957**, **Brier 0.5253** (vs. azar 1.099 / 0.667).
Tabla completa de los 48 equipos: [`output/probabilities.csv`](output/probabilities.csv).

## Quickstart

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # Windows ; o: source .venv/bin/activate
pip install -r requirements.txt

# descargar histórico de partidos internacionales
python -c "from worldcup.ingest import download_results; download_results('https://raw.githubusercontent.com/martj42/international_results/master/results.csv','data/raw/results.csv')"

pytest -q          # 29 tests
python run.py      # -> output/prediction.md + output/probabilities.csv
```

## Configuración

Todo se controla desde [`config.yaml`](config.yaml): pesos del ensemble, parámetros de los
modelos, número de simulaciones y el sorteo oficial de grupos. Por ejemplo, el peso del mercado
en la mezcla final vive en `ensemble.champion`.

## Estructura

```
src/worldcup/   ingest · elo · dixon_coles · market · ensemble ·
                matchengine · tournament · simulate · backtest · report
tests/          un módulo de tests por módulo de código (TDD)
docs/           diseño (spec) y plan de implementación
output/         prediction.md (reporte) + probabilities.csv (48 equipos)
```

## Descargo

Experimento educativo. Las probabilidades son estimaciones de un modelo y **no son consejo de
apuestas**. Detalles y limitaciones en [`PAPER.md`](PAPER.md).

## Licencia

MIT — ver [`LICENSE`](LICENSE).
