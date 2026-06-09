# Un ensemble calibrado para predecir la Copa Mundial de la FIFA 2026

**Autor:** Juan Castillo · **Asistente de modelado:** Claude Opus 4.8
**Fecha:** 9 de junio de 2026
**Repositorio:** `worldcup-predict`

---

## Resumen (Abstract)

Este trabajo documenta un experimento de predicción probabilística del Mundial de
Fútbol 2026 (48 equipos, sede Canadá–México–EE. UU.). El objetivo no es producir una
predicción "verdadera" —imposible por la naturaleza aleatoria del fútbol— sino construir
el sistema estadístico **mejor calibrado y más reproducible** posible con datos públicos
y gratuitos, como entrada a una competencia informal entre varias IAs (Claude, Gemini y
otras) que predicen el mismo torneo.

Combinamos tres señales independientes: (1) un ranking **Elo** de selecciones calculado
sobre el histórico completo de partidos internacionales (1872–2026), (2) un modelo de goles
**Dixon-Coles** con decaimiento temporal que pondera la forma reciente, y (3) la **señal de
mercado** (cuotas de campeón de casas de apuestas) tras quitarles el margen. Las dos primeras
se combinan a nivel partido mediante un *log-opinion pool*; el mercado se mezcla a nivel
campeón. El torneo se resuelve con una simulación **Monte Carlo de 50 000 iteraciones** sobre
el formato real de 2026.

En *backtest* sobre los 328 partidos internacionales de los últimos 180 días, el ensemble
logra **log-loss = 0.8957** y **Brier = 0.5253**, frente a los baselines de azar uniforme
de 1.0986 y 0.6667 respectivamente — una mejora consistente y medible. El sistema favorece a
**Argentina (19.7 %)** y **España (16.4 %)** como candidatas al título según el modelo puro;
al anclar al mercado, **España (17.1 %)** pasa al primer lugar.

---

## 1. Motivación

Las predicciones deportivas hechas por LLMs suelen ser **cualitativas**: el modelo "razona"
sobre plantillas y narrativas y emite un favorito. Eso es difícil de evaluar y fácil de sesgar
hacia los nombres más mediáticos. Este experimento adopta el enfoque opuesto: producir
**probabilidades** a partir de datos, y **medir su calibración** con métricas propias de
*forecasting* (log-loss, Brier). La hipótesis es que un pipeline estadístico transparente,
aun con datos gratuitos, vence en calibración a una predicción puramente narrativa.

## 2. Datos

| Fuente | Uso | Acceso |
|---|---|---|
| [`martj42/international_results`](https://github.com/martj42/international_results) | Histórico completo de partidos internacionales (fecha, marcador, torneo, sede) | CSV público, sin API key |
| Sorteo oficial FIFA 2026 | Composición de los 12 grupos (A–L) | Confirmado tras los repechajes de marzo 2026 |
| Cuotas de campeón (ESPN/agregadores) | Señal de mercado | Snapshot de inicio de junio 2026 |

- **Cobertura temporal de los datos:** hasta el **8 de junio de 2026** (49 378 partidos).
- **48 equipos** del sorteo, validados uno a uno contra las grafías del dataset.
- El snapshot de cuotas y el draw quedan versionados en el repo para reproducibilidad exacta.

> **Por qué `martj42` y no una API de selecciones:** las APIs gratuitas (p. ej. football-data.org)
> cubren bien ligas de clubes pero su cobertura de partidos de **selecciones** (amistosos,
> clasificatorias, Nations League) es irregular y con *rate-limits* duros. El dataset de martj42
> es el estándar abierto para modelado de selecciones.

## 3. Metodología

### 3.1 Elo de selecciones

Rating actualizado partido a partido sobre todo el histórico, al estilo de
*World Football Elo*. Para un partido con diferencia de rating `d` (incluyendo ventaja de
local salvo en sede neutral), la expectativa de puntos del local es:

$$ E_{\text{local}} = \frac{1}{1 + 10^{-d/400}} $$

La actualización aplica un factor `K` ponderado por importancia del torneo (Mundial 2.0,
clasificatorias 1.5, amistosos 1.0, …) y un multiplicador por margen de victoria `G`
(1 si gana por ≤1, 1.5 por 2, `(11+m)/8` por `m ≥ 3` goles). La probabilidad de empate se
modela como función de lo parejo del partido, con un parámetro `d₀` calibrable:

$$ p_{\text{empate}} = d_0\,\bigl(1 - |2E_{\text{local}} - 1|\bigr) $$

### 3.2 Modelo de goles Dixon-Coles

Estima una fuerza de **ataque** `αᵢ` y **defensa** `δᵢ` por equipo, más ventaja de local `γ`
y un parámetro `ρ` de corrección para marcadores bajos. Las tasas de goles esperadas son:

$$ \lambda_{\text{local}} = e^{\alpha_h - \delta_a + \gamma}, \qquad
   \mu_{\text{visit}}    = e^{\alpha_a - \delta_h} $$

El ajuste es por **máxima verosimilitud ponderada por recencia**: cada partido pesa
`exp(−ln2 · edad_días / vida_media)` con vida media de 365 días, de modo que la forma de los
últimos 6 meses domina. Se aplica la corrección Dixon-Coles `τ(·)` a los resultados 0-0, 1-0,
0-1 y 1-1 para capturar la dependencia en marcadores bajos. La matriz de marcadores resultante
da `P(victoria/empate/derrota)` y la distribución conjunta de goles.

### 3.3 Ensemble en dos niveles

Las fuentes gratuitas publican cuotas de **campeón**, no de cada partido; por eso la mezcla
ocurre en dos niveles distintos:

- **Nivel partido** — *log-opinion pool* (media geométrica ponderada, renormalizada) de los
  vectores `P(W/D/L)` de Elo y Dixon-Coles. Pesos calibrados por *backtest*.

$$ p \;\propto\; \prod_k p_k^{\,w_k}, \qquad \textstyle\sum_k w_k = 1 $$

- **Nivel campeón** — la distribución de campeón que sale del Monte Carlo se mezcla, con el
  mismo *log-opinion pool*, con las probabilidades de mercado **de-vig** (cuotas invertidas y
  normalizadas para quitar el margen de la casa).

### 3.4 Simulación Monte Carlo del torneo

Se simulan **50 000 torneos completos** (semilla fija = 42) con el **formato real 2026**:

1. **Fase de grupos:** 12 grupos de 4, todos contra todos. El ensemble decide el resultado
   (W/D/L) y la matriz Dixon-Coles —condicionada a ese resultado— aporta un marcador realista,
   necesario para los desempates por diferencia de goles.
2. **Clasificación:** primeros 2 de cada grupo (24) + los **8 mejores terceros**, con los
   desempates oficiales (puntos → diferencia → goles a favor → *head-to-head*).
3. **Eliminatorias:** 32 equipos → ronda de 32 → octavos → cuartos → semis → final. En empate
   se resuelve con probabilidad proporcional a la fuerza relativa (prórroga/penales).

Las frecuencias sobre las 50 000 corridas son las probabilidades de avanzar cada ronda.

## 4. Resultados

### 4.1 Calibración (backtest, últimos 180 días, n = 328)

| Métrica | Ensemble | Azar uniforme | Mejora |
|---|---|---|---|
| Log-loss | **0.8957** | 1.0986 | −18.5 % |
| Brier | **0.5253** | 0.6667 | −21.2 % |

Los pesos óptimos a nivel partido resultaron **Elo 0.5 / Dixon-Coles 0.5** (búsqueda en rejilla
minimizando log-loss en el *holdout*).

### 4.2 Probabilidades de campeón (top 12 de 48)

| # | Equipo | Modelo P(campeón) | Mezclado c/ mercado | P(final) | P(semis) |
|---|---|---|---|---|---|
| 1 | Argentina | 19.7 % | 14.4 % | 25.7 % | 35.4 % |
| 2 | España | 16.4 % | **17.1 %** | 21.8 % | 31.0 % |
| 3 | Francia | 5.8 % | 9.4 % | 9.1 % | 16.2 % |
| 4 | Brasil | 5.5 % | 7.0 % | 9.3 % | 25.0 % |
| 5 | Portugal | 5.2 % | 7.1 % | 12.4 % | 23.2 % |
| 6 | Inglaterra | 5.2 % | 7.7 % | 8.6 % | 15.5 % |
| 7 | Colombia | 4.5 % | 3.4 % | 7.6 % | 21.7 % |
| 8 | Japón | 4.4 % | 2.7 % | 11.5 % | 20.8 % |
| 9 | Noruega | 4.4 % | 3.5 % | 11.4 % | 20.3 % |
| 10 | Ecuador | 4.2 % | 2.4 % | 7.6 % | 21.1 % |
| 11 | Países Bajos | 4.1 % | 4.4 % | 11.3 % | 20.0 % |
| 12 | Alemania | 3.9 % | 5.0 % | 7.1 % | 21.4 % |

La tabla completa de los 48 equipos y todas las rondas está en
[`output/probabilities.csv`](output/probabilities.csv).

### 4.3 Lecturas

- **Discrepancia modelo–mercado:** el modelo pondera más la **forma reciente sudamericana**
  (Argentina, Colombia, Ecuador), mientras que el mercado infla a los grandes europeos
  (Francia, Inglaterra, Portugal). El mercado baja a Argentina −5.2 pp.
- **Dark horses del modelo:** Japón, Ecuador, Colombia y Noruega aparecen por encima de su
  cuota de mercado — el sistema premia su Elo y rendimiento recientes.
- **Decisión de submission:** el modelo puro apuesta por **Argentina** (campeón *contrarian*);
  la versión anclada al mercado prefiere **España** (mínimo riesgo).

## 5. Limitaciones y aproximaciones (honestas)

1. **Siembra de eliminatorias por fuerza Elo (1 vs 32), no por la tabla oficial de cruces FIFA.**
   Es una simplificación documentada: favorece levemente a equipos de Elo alto en grupos débiles
   (de ahí Japón/Ecuador algo inflados). Es el cambio de mayor impacto pendiente.
2. **Sin datos a nivel jugador** (lesiones, alineaciones, xG por partido): no hay fuente gratuita
   fiable para selecciones. El modelo es a nivel equipo.
3. **El mercado solo entra a nivel campeón** (las fuentes gratis no publican cuotas por partido).
4. **Snapshot estático:** la predicción es pre-torneo; no se actualiza con resultados en vivo.
5. **Cuotas de una sola fuente y un solo momento** — un promedio multi-casa sería más robusto.

## 6. Reproducibilidad

Todo el pipeline es determinista (semilla fija) y se ejecuta con datos públicos:

```bash
# 1. Entorno (Python 3.11+; probado en 3.13)
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # Windows PowerShell
# source .venv/bin/activate          # Linux/macOS
pip install -r requirements.txt

# 2. Descargar el histórico de partidos
python -c "from worldcup.ingest import download_results; download_results('https://raw.githubusercontent.com/martj42/international_results/master/results.csv','data/raw/results.csv')"

# 3. (El draw oficial ya viene en config.yaml; el snapshot de cuotas en data/raw/market_futures.csv)

# 4. Tests + pipeline completo
pytest -q
python run.py        # escribe output/prediction.md y output/probabilities.csv
```

Estructura del código (cada módulo, una responsabilidad): `ingest`, `elo`, `dixon_coles`,
`market`, `ensemble`, `matchengine`, `tournament`, `simulate`, `backtest`, `report`.
El diseño y el plan de implementación paso a paso están en `docs/superpowers/`.

## 7. Descargo de responsabilidad

Este es un **experimento educativo y recreativo**. Las probabilidades son estimaciones de un
modelo estadístico y **no constituyen consejo de apuestas**. El fútbol es intrínsecamente
incierto; ningún modelo "acierta" un torneo. El valor del trabajo está en la **metodología
transparente y la calibración medible**, no en el resultado puntual.

## 8. Reconocimientos y fuentes

- Datos de partidos: proyecto abierto [`martj42/international_results`](https://github.com/martj42/international_results).
- Sorteo y formato 2026: FIFA / cobertura pública (ESPN, Wikipedia).
- Marco metodológico: Elo (Arpad Elo; *World Football Elo*), Dixon & Coles (1997),
  *log-opinion pooling* (literatura de *forecast combination*).

## Cómo citar

```
Castillo, J. (2026). Un ensemble calibrado para predecir la Copa Mundial de la FIFA 2026.
Experimento worldcup-predict. https://github.com/<usuario>/worldcup-predict
```
