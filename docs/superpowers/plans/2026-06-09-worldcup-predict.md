# World Cup 2026 Prediction — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a calibrated statistical predictor of the 2026 World Cup (full bracket + champion) that ensembles a self-computed Elo rating, a Dixon-Coles goals model, and de-vigged market futures, then Monte-Carlo-simulates the real 48-team format.

**Architecture:** A 4-stage Python pipeline. (1) `ingest` downloads international results (martj42), the official WC group draw, and champion futures odds. (2) Two models score each match — Elo and Dixon-Coles — combined per-match by a log-opinion pool. (3) `simulate` runs 50k Monte Carlo tournaments: ensemble W/D/L decides the winner, the DC score matrix (conditioned on that outcome) supplies a realistic scoreline for group tiebreakers. (4) The Monte-Carlo champion distribution is blended with de-vigged market probabilities and rendered to a Markdown report with an ASCII bracket. A backtest on held-out recent matches measures log-loss/Brier and is the headline credential.

**Tech Stack:** Python 3.11, pandas, numpy, scipy (`optimize`, `stats`), PyYAML, requests, pytest.

---

## File Structure

```
worldcup-predict/
├── config.yaml                # all params, weights, group draw, tournament weights
├── requirements.txt
├── pytest.ini
├── data/raw/                  # downloaded CSVs (gitignored)
├── data/processed/            # cleaned datasets
├── src/worldcup/
│   ├── __init__.py
│   ├── config.py              # load_config()
│   ├── ingest.py              # download results / fixtures / market
│   ├── elo.py                 # Elo ratings + match W/D/L
│   ├── dixon_coles.py         # Poisson attack/defence fit + score matrix
│   ├── market.py              # odds → de-vigged probabilities
│   ├── ensemble.py            # log-opinion pool (match + champion levels)
│   ├── matchengine.py         # outcome + conditioned scoreline sampler
│   ├── tournament.py          # group sim, FIFA tiebreakers, best-thirds, knockout
│   ├── simulate.py            # Monte Carlo driver + aggregation
│   ├── backtest.py            # log-loss / Brier on holdout, weight calibration
│   └── report.py              # Markdown report + ASCII bracket
├── tests/                     # one test module per src module
└── run.py                     # end-to-end pipeline entrypoint
```

Each `src/worldcup/*.py` has one responsibility and a small, explicit interface (signatures fixed in the tasks below). Files that change together (a model and its probability mapping) live together.

---

## Task 0: Project scaffold

**Files:**
- Create: `requirements.txt`, `pytest.ini`, `.gitignore`, `src/worldcup/__init__.py`, `config.yaml`, `src/worldcup/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Initialize repo and structure**

```powershell
cd D:\Development\worldcup-predict
git init
nvm use estable  # not needed for python, skip if nvm errors
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
mkdir src\worldcup, tests, data\raw, data\processed, output -Force
```

- [ ] **Step 2: Write `requirements.txt`**

```
pandas==2.2.2
numpy==1.26.4
scipy==1.13.1
PyYAML==6.0.1
requests==2.32.3
pytest==8.2.2
```

- [ ] **Step 3: Write `.gitignore`**

```
.venv/
__pycache__/
data/raw/
output/
*.pyc
```

- [ ] **Step 4: Write `pytest.ini`**

```ini
[pytest]
pythonpath = src
testpaths = tests
```

- [ ] **Step 5: Install deps**

Run: `pip install -r requirements.txt`
Expected: all install without error.

- [ ] **Step 6: Write `config.yaml`** (group draw `tournament.groups` is filled in Task 5; leave the 12 keys with empty lists for now)

```yaml
data:
  results_url: "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
  football_data_token: "REPLACE_WITH_YOUR_TOKEN"
  results_csv: "data/raw/results.csv"
  market_csv: "data/raw/market_futures.csv"
elo:
  initial_rating: 1500.0
  home_advantage: 65.0
  k_base: 40.0
  tournament_weights:
    "FIFA World Cup": 2.0
    "FIFA World Cup qualification": 1.5
    "UEFA Nations League": 1.25
    "UEFA Euro": 1.75
    "Copa América": 1.75
    "African Cup of Nations": 1.5
    "AFC Asian Cup": 1.5
    "Friendly": 1.0
  default_tournament_weight: 1.0
dixon_coles:
  half_life_days: 365.0
  max_goals: 10
  recent_window_days: 1460
ensemble:
  match: { elo: 0.55, dc: 0.45 }
  champion: { model: 0.55, market: 0.45 }
  draw_param: 0.28
simulation:
  n_sims: 50000
  seed: 42
backtest:
  holdout_days: 180
tournament:
  groups:
    A: []
    B: []
    C: []
    D: []
    E: []
    F: []
    G: []
    H: []
    I: []
    J: []
    K: []
    L: []
```

- [ ] **Step 7: Write `src/worldcup/__init__.py`** (empty file)

- [ ] **Step 8: Write the failing test `tests/test_config.py`**

```python
from worldcup.config import load_config

def test_load_config_reads_yaml(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text("simulation:\n  n_sims: 10\n", encoding="utf-8")
    cfg = load_config(str(f))
    assert cfg["simulation"]["n_sims"] == 10
```

- [ ] **Step 9: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'worldcup.config'`

- [ ] **Step 10: Write `src/worldcup/config.py`**

```python
import yaml

def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)
```

- [ ] **Step 11: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 12: Commit**

```bash
git add -A
git commit -m "[CHORE] scaffold worldcup-predict project"
```

---

## Task 1: Ingest — download and normalize data

**Files:**
- Create: `src/worldcup/ingest.py`
- Test: `tests/test_ingest.py`

Interface:
- `download_results(url: str, dest: str) -> str` — GET CSV, write to `dest`, return `dest`.
- `load_results(path: str) -> pd.DataFrame` — read CSV, parse `date`, coerce score columns to int, drop rows with null scores.
- `load_market(path: str) -> dict[str, float]` — read a 2-column CSV `team,decimal_odds` into a dict.

- [ ] **Step 1: Write the failing test `tests/test_ingest.py`**

```python
import pandas as pd
from worldcup.ingest import load_results, load_market

def test_load_results_parses_and_filters(tmp_path):
    p = tmp_path / "results.csv"
    p.write_text(
        "date,home_team,away_team,home_score,away_score,tournament,city,country,neutral\n"
        "2026-03-01,Brazil,Argentina,2,1,Friendly,Rio,Brazil,FALSE\n"
        "2026-03-02,Spain,France,,,Friendly,Madrid,Spain,FALSE\n",
        encoding="utf-8",
    )
    df = load_results(str(p))
    assert len(df) == 1
    assert df.iloc[0]["home_score"] == 2
    assert pd.api.types.is_datetime64_any_dtype(df["date"])

def test_load_market_to_dict(tmp_path):
    p = tmp_path / "market.csv"
    p.write_text("team,decimal_odds\nBrazil,5.0\nFrance,6.5\n", encoding="utf-8")
    m = load_market(str(p))
    assert m["Brazil"] == 5.0 and m["France"] == 6.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ingest.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write `src/worldcup/ingest.py`**

```python
import requests
import pandas as pd

def download_results(url: str, dest: str) -> str:
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    with open(dest, "wb") as fh:
        fh.write(resp.content)
    return dest

def load_results(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df = df.dropna(subset=["date", "home_score", "away_score"]).copy()
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    if "neutral" in df.columns:
        df["neutral"] = df["neutral"].astype(str).str.upper().isin(["TRUE", "1"])
    else:
        df["neutral"] = False
    return df.sort_values("date").reset_index(drop=True)

def load_market(path: str) -> dict:
    df = pd.read_csv(path)
    return {str(r["team"]): float(r["decimal_odds"]) for _, r in df.iterrows()}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ingest.py -v`
Expected: PASS

- [ ] **Step 5: Manually fetch the real data once**

Run:
```powershell
python -c "from worldcup.ingest import download_results; download_results('https://raw.githubusercontent.com/martj42/international_results/master/results.csv','data/raw/results.csv')"
```
Expected: `data/raw/results.csv` exists, > 1MB. Verify: `python -c "from worldcup.ingest import load_results; print(load_results('data/raw/results.csv').tail(3))"` shows recent 2026 matches.

- [ ] **Step 6: Hand-build `data/raw/market_futures.csv`** by entering current champion futures (decimal odds) for the contenders from any public aggregator. Minimum: every team that appears in the group draw, two columns `team,decimal_odds`. Long-shots can share a nominal high odd (e.g. 1000).

- [ ] **Step 7: Commit**

```bash
git add src/worldcup/ingest.py tests/test_ingest.py
git commit -m "[FEAT] ingest: download/load international results and market odds"
```

---

## Task 2: Elo ratings and match W/D/L

**Files:**
- Create: `src/worldcup/elo.py`
- Test: `tests/test_elo.py`

Interface:
- `compute_ratings(results: pd.DataFrame, cfg: dict) -> dict[str, float]` — process matches chronologically; eloratings.net update with MoV multiplier and per-tournament K weight; home advantage applied unless `neutral`.
- `match_probabilities(r_home: float, r_away: float, neutral: bool, cfg: dict) -> tuple[float, float, float]` — returns `(p_home, p_draw, p_away)` summing to 1.

The draw model: with `e = 1/(1+10**(-diff/400))` (expected home points share) and `d0 = cfg["ensemble"]["draw_param"]`, set `p_draw = d0 * (1 - abs(2*e - 1))`, then `p_home = e*(1-p_draw)`, `p_away = (1-e)*(1-p_draw)`.

- [ ] **Step 1: Write the failing test `tests/test_elo.py`**

```python
import pandas as pd
from worldcup.elo import compute_ratings, match_probabilities

CFG = {
    "elo": {"initial_rating": 1500.0, "home_advantage": 65.0, "k_base": 40.0,
            "tournament_weights": {"Friendly": 1.0}, "default_tournament_weight": 1.0},
    "ensemble": {"draw_param": 0.28},
}

def test_winner_gains_rating():
    df = pd.DataFrame([{
        "date": pd.Timestamp("2025-01-01"), "home_team": "A", "away_team": "B",
        "home_score": 3, "away_score": 0, "tournament": "Friendly", "neutral": True,
    }])
    r = compute_ratings(df, CFG)
    assert r["A"] > 1500.0 and r["B"] < 1500.0
    assert round(r["A"] - 1500.0, 6) == round(1500.0 - r["B"], 6)  # zero-sum

def test_match_probabilities_sum_to_one_and_favor_stronger():
    p_home, p_draw, p_away = match_probabilities(1800, 1500, neutral=True, cfg=CFG)
    assert abs(p_home + p_draw + p_away - 1.0) < 1e-9
    assert p_home > p_away
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_elo.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write `src/worldcup/elo.py`**

```python
import pandas as pd

def _mov_multiplier(goal_diff: int) -> float:
    g = abs(goal_diff)
    if g <= 1:
        return 1.0
    if g == 2:
        return 1.5
    return (11.0 + g) / 8.0

def compute_ratings(results: pd.DataFrame, cfg: dict) -> dict:
    e = cfg["elo"]
    ratings: dict[str, float] = {}
    init = e["initial_rating"]
    weights = e["tournament_weights"]
    default_w = e["default_tournament_weight"]
    for _, m in results.sort_values("date").iterrows():
        h, a = m["home_team"], m["away_team"]
        ratings.setdefault(h, init)
        ratings.setdefault(a, init)
        ha = 0.0 if m["neutral"] else e["home_advantage"]
        diff = (ratings[h] + ha) - ratings[a]
        exp_home = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))
        gd = int(m["home_score"]) - int(m["away_score"])
        if gd > 0:
            score_home = 1.0
        elif gd < 0:
            score_home = 0.0
        else:
            score_home = 0.5
        k = e["k_base"] * weights.get(m.get("tournament", ""), default_w) * _mov_multiplier(gd)
        delta = k * (score_home - exp_home)
        ratings[h] += delta
        ratings[a] -= delta
    return ratings

def match_probabilities(r_home: float, r_away: float, neutral: bool, cfg: dict):
    ha = 0.0 if neutral else cfg["elo"]["home_advantage"]
    diff = (r_home + ha) - r_away
    e = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))
    d0 = cfg["ensemble"]["draw_param"]
    p_draw = d0 * (1.0 - abs(2.0 * e - 1.0))
    p_home = e * (1.0 - p_draw)
    p_away = (1.0 - e) * (1.0 - p_draw)
    return p_home, p_draw, p_away
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_elo.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/worldcup/elo.py tests/test_elo.py
git commit -m "[FEAT] elo: ratings from history + W/D/L match probabilities"
```

---

## Task 3: Dixon-Coles goals model

**Files:**
- Create: `src/worldcup/dixon_coles.py`
- Test: `tests/test_dixon_coles.py`

Interface:
- `fit(results: pd.DataFrame, cfg: dict, ref_date: pd.Timestamp) -> dict` — weighted MLE. Returns `{"attack": {team: a}, "defence": {team: d}, "home_adv": float, "rho": float, "teams": [..]}`. Weight each match by `exp(-ln(2) * age_days / half_life_days)`; only use matches within `recent_window_days` of `ref_date`.
- `score_matrix(params: dict, home: str, away: str, neutral: bool, max_goals: int) -> np.ndarray` — `(max_goals+1, max_goals+1)` joint probability matrix with the Dixon-Coles low-score `tau` correction.
- `wdl_from_matrix(matrix: np.ndarray) -> tuple[float, float, float]` — `(p_home, p_draw, p_away)`.

- [ ] **Step 1: Write the failing test `tests/test_dixon_coles.py`**

```python
import numpy as np
import pandas as pd
from worldcup.dixon_coles import fit, score_matrix, wdl_from_matrix

def _toy_results():
    rows = []
    base = pd.Timestamp("2026-01-01")
    # Strong scores lots vs Weak; repeat to give the fitter signal.
    for i in range(12):
        rows.append({"date": base, "home_team": "Strong", "away_team": "Weak",
                     "home_score": 3, "away_score": 0, "tournament": "Friendly", "neutral": True})
        rows.append({"date": base, "home_team": "Weak", "away_team": "Strong",
                     "home_score": 0, "away_score": 2, "tournament": "Friendly", "neutral": True})
    return pd.DataFrame(rows)

CFG = {"dixon_coles": {"half_life_days": 365.0, "max_goals": 10, "recent_window_days": 4000}}

def test_matrix_is_normalized():
    params = fit(_toy_results(), CFG, ref_date=pd.Timestamp("2026-02-01"))
    m = score_matrix(params, "Strong", "Weak", neutral=True, max_goals=10)
    assert abs(m.sum() - 1.0) < 1e-6

def test_stronger_team_more_likely_to_win():
    params = fit(_toy_results(), CFG, ref_date=pd.Timestamp("2026-02-01"))
    m = score_matrix(params, "Strong", "Weak", neutral=True, max_goals=10)
    p_home, p_draw, p_away = wdl_from_matrix(m)
    assert p_home > p_away
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dixon_coles.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write `src/worldcup/dixon_coles.py`**

```python
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

def _tau(h, a, lam, mu, rho):
    # Dixon-Coles low-score dependence correction.
    if h == 0 and a == 0:
        return 1.0 - lam * mu * rho
    if h == 0 and a == 1:
        return 1.0 + lam * rho
    if h == 1 and a == 0:
        return 1.0 + mu * rho
    if h == 1 and a == 1:
        return 1.0 - rho
    return 1.0

def fit(results: pd.DataFrame, cfg: dict, ref_date: pd.Timestamp) -> dict:
    dc = cfg["dixon_coles"]
    cutoff = ref_date - pd.Timedelta(days=dc["recent_window_days"])
    df = results[(results["date"] >= cutoff) & (results["date"] <= ref_date)].copy()
    teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    age = (ref_date - df["date"]).dt.days.to_numpy()
    w = np.exp(-np.log(2.0) * age / dc["half_life_days"])
    hi = df["home_team"].map(idx).to_numpy()
    ai = df["away_team"].map(idx).to_numpy()
    hg = df["home_score"].to_numpy()
    ag = df["away_score"].to_numpy()
    neutral = df["neutral"].to_numpy()

    # params: attack[n], defence[n], home_adv, rho
    x0 = np.concatenate([np.zeros(n), np.zeros(n), [0.25], [-0.05]])

    def nll(x):
        atk = x[:n]
        dfn = x[n:2 * n]
        home_adv = x[2 * n]
        rho = x[2 * n + 1]
        ha = np.where(neutral, 0.0, home_adv)
        lam = np.exp(atk[hi] - dfn[ai] + ha)
        mu = np.exp(atk[ai] - dfn[hi])
        ll = (poisson.logpmf(hg, lam) + poisson.logpmf(ag, mu))
        tau = np.array([_tau(int(h), int(a), l, m, rho)
                        for h, a, l, m in zip(hg, ag, lam, mu)])
        tau = np.clip(tau, 1e-9, None)
        ll = ll + np.log(tau)
        # identifiability: penalize mean(attack) drift toward 0
        pen = 1000.0 * (atk.mean() ** 2)
        return -np.sum(w * ll) + pen

    bounds = [(-3, 3)] * (2 * n) + [(-1, 1), (-0.2, 0.2)]
    res = minimize(nll, x0, method="L-BFGS-B", bounds=bounds)
    x = res.x
    return {
        "attack": {t: float(x[idx[t]]) for t in teams},
        "defence": {t: float(x[n + idx[t]]) for t in teams},
        "home_adv": float(x[2 * n]),
        "rho": float(x[2 * n + 1]),
        "teams": teams,
    }

def score_matrix(params: dict, home: str, away: str, neutral: bool, max_goals: int) -> np.ndarray:
    atk, dfn = params["attack"], params["defence"]
    ha = 0.0 if neutral else params["home_adv"]
    lam = np.exp(atk.get(home, 0.0) - dfn.get(away, 0.0) + ha)
    mu = np.exp(atk.get(away, 0.0) - dfn.get(home, 0.0))
    hg = poisson.pmf(np.arange(max_goals + 1), lam)
    ag = poisson.pmf(np.arange(max_goals + 1), mu)
    m = np.outer(hg, ag)
    rho = params["rho"]
    for (h, a) in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        m[h, a] *= _tau(h, a, lam, mu, rho)
    m = np.clip(m, 0.0, None)
    return m / m.sum()

def wdl_from_matrix(matrix: np.ndarray):
    p_home = float(np.tril(matrix, -1).sum())
    p_away = float(np.triu(matrix, 1).sum())
    p_draw = float(np.trace(matrix))
    return p_home, p_draw, p_away
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_dixon_coles.py -v`
Expected: PASS (fitter may take a few seconds on toy data).

- [ ] **Step 5: Commit**

```bash
git add src/worldcup/dixon_coles.py tests/test_dixon_coles.py
git commit -m "[FEAT] dixon-coles: time-weighted goals model + score matrix"
```

---

## Task 4: Market de-vig and ensemble pools

**Files:**
- Create: `src/worldcup/market.py`, `src/worldcup/ensemble.py`
- Test: `tests/test_market.py`, `tests/test_ensemble.py`

Interface:
- `market.devig(odds: dict[str, float]) -> dict[str, float]` — implied = 1/odds, normalized to sum 1.
- `ensemble.log_opinion_pool(vectors: list[np.ndarray], weights: list[float]) -> np.ndarray` — weighted geometric mean, renormalized.
- `ensemble.match_wdl(elo_wdl, dc_wdl, cfg) -> np.ndarray` — pool of the two W/D/L vectors with `cfg["ensemble"]["match"]` weights.
- `ensemble.blend_champion(model: dict[str,float], market: dict[str,float], cfg) -> dict[str,float]` — per-team log pool with `cfg["ensemble"]["champion"]` weights over the shared team set.

- [ ] **Step 1: Write the failing tests**

`tests/test_market.py`:
```python
from worldcup.market import devig

def test_devig_normalizes_and_orders():
    p = devig({"A": 2.0, "B": 4.0, "C": 4.0})
    assert abs(sum(p.values()) - 1.0) < 1e-9
    assert p["A"] > p["B"]
```

`tests/test_ensemble.py`:
```python
import numpy as np
from worldcup.ensemble import log_opinion_pool, match_wdl, blend_champion

CFG = {"ensemble": {"match": {"elo": 0.5, "dc": 0.5},
                    "champion": {"model": 0.5, "market": 0.5}}}

def test_pool_sums_to_one():
    out = log_opinion_pool([np.array([0.5, 0.3, 0.2]), np.array([0.4, 0.4, 0.2])], [0.5, 0.5])
    assert abs(out.sum() - 1.0) < 1e-9

def test_match_wdl_blends():
    out = match_wdl((0.6, 0.2, 0.2), (0.4, 0.3, 0.3), CFG)
    assert abs(out.sum() - 1.0) < 1e-9
    assert out[0] > out[2]

def test_blend_champion_union_of_teams():
    out = blend_champion({"A": 0.6, "B": 0.4}, {"A": 0.5, "B": 0.5}, CFG)
    assert abs(sum(out.values()) - 1.0) < 1e-9
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_market.py tests/test_ensemble.py -v`
Expected: FAIL — modules not found.

- [ ] **Step 3: Write `src/worldcup/market.py`**

```python
def devig(odds: dict) -> dict:
    implied = {t: 1.0 / o for t, o in odds.items()}
    total = sum(implied.values())
    return {t: v / total for t, v in implied.items()}
```

- [ ] **Step 4: Write `src/worldcup/ensemble.py`**

```python
import numpy as np

def log_opinion_pool(vectors, weights) -> np.ndarray:
    w = np.array(weights, dtype=float)
    w = w / w.sum()
    stacked = np.vstack([np.clip(np.asarray(v, dtype=float), 1e-12, None) for v in vectors])
    log_mix = (w[:, None] * np.log(stacked)).sum(axis=0)
    out = np.exp(log_mix)
    return out / out.sum()

def match_wdl(elo_wdl, dc_wdl, cfg) -> np.ndarray:
    m = cfg["ensemble"]["match"]
    return log_opinion_pool([np.array(elo_wdl), np.array(dc_wdl)], [m["elo"], m["dc"]])

def blend_champion(model: dict, market: dict, cfg) -> dict:
    c = cfg["ensemble"]["champion"]
    teams = sorted(set(model) | set(market))
    mv = np.array([model.get(t, 0.0) for t in teams])
    kv = np.array([market.get(t, 0.0) for t in teams])
    mixed = log_opinion_pool([mv, kv], [c["model"], c["market"]])
    return {t: float(p) for t, p in zip(teams, mixed)}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_market.py tests/test_ensemble.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/worldcup/market.py src/worldcup/ensemble.py tests/test_market.py tests/test_ensemble.py
git commit -m "[FEAT] market de-vig + log-opinion-pool ensemble"
```

---

## Task 5: Fill the official group draw + provider

**Files:**
- Modify: `config.yaml` (`tournament.groups`)
- Create: `src/worldcup/teams.py`
- Test: `tests/test_teams.py`

Interface:
- `teams.all_teams(cfg: dict) -> list[str]` — flatten the 12 groups.
- `teams.validate_groups(cfg: dict) -> None` — raise `ValueError` unless there are exactly 12 groups of 4 unique teams (48 total).

- [ ] **Step 1: Enter the official WC2026 draw into `config.yaml`** under `tournament.groups`, 4 team names per group (A–L). Use the exact spellings found in `data/raw/results.csv` (check with: `python -c "from worldcup.ingest import load_results; print(sorted(load_results('data/raw/results.csv')['home_team'].unique()))"`). If the official draw is unavailable, enter the 48 qualified teams in any 12×4 arrangement to unblock development and correct later.

- [ ] **Step 2: Write the failing test `tests/test_teams.py`**

```python
import pytest
from worldcup.teams import all_teams, validate_groups

def _cfg(groups):
    return {"tournament": {"groups": groups}}

def test_all_teams_flattens():
    cfg = _cfg({"A": ["x", "y"], "B": ["z"]})
    assert all_teams(cfg) == ["x", "y", "z"]

def test_validate_rejects_wrong_size():
    with pytest.raises(ValueError):
        validate_groups(_cfg({"A": ["x", "y", "z", "w"]}))  # only 1 group

def test_validate_accepts_48():
    groups = {chr(65 + i): [f"t{i}{j}" for j in range(4)] for i in range(12)}
    validate_groups(_cfg(groups))  # no raise
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_teams.py -v`
Expected: FAIL — module not found.

- [ ] **Step 4: Write `src/worldcup/teams.py`**

```python
def all_teams(cfg: dict) -> list:
    out = []
    for g in cfg["tournament"]["groups"].values():
        out.extend(g)
    return out

def validate_groups(cfg: dict) -> None:
    groups = cfg["tournament"]["groups"]
    if len(groups) != 12:
        raise ValueError(f"expected 12 groups, got {len(groups)}")
    teams = all_teams(cfg)
    for name, g in groups.items():
        if len(g) != 4:
            raise ValueError(f"group {name} must have 4 teams, got {len(g)}")
    if len(set(teams)) != 48:
        raise ValueError(f"expected 48 unique teams, got {len(set(teams))}")
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_teams.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add config.yaml src/worldcup/teams.py tests/test_teams.py
git commit -m "[FEAT] teams: official WC2026 group draw + validation"
```

---

## Task 6: Match engine — outcome + conditioned scoreline

**Files:**
- Create: `src/worldcup/matchengine.py`
- Test: `tests/test_matchengine.py`

Interface:
- `sample_outcome(wdl: np.ndarray, rng: np.random.Generator) -> int` — 0 home, 1 draw, 2 away.
- `sample_scoreline(matrix: np.ndarray, outcome: int, rng) -> tuple[int, int]` — restrict the DC score matrix to cells matching `outcome`, renormalize, sample one cell.
- `knockout_winner(wdl: np.ndarray, rng) -> int` — 0 home advances / 1 away advances; draw mass split proportionally (`p_home/(p_home+p_away)`).

- [ ] **Step 1: Write the failing test `tests/test_matchengine.py`**

```python
import numpy as np
from worldcup.matchengine import sample_outcome, sample_scoreline, knockout_winner

def test_sample_outcome_deterministic_when_certain():
    rng = np.random.default_rng(0)
    assert sample_outcome(np.array([1.0, 0.0, 0.0]), rng) == 0
    assert sample_outcome(np.array([0.0, 0.0, 1.0]), rng) == 2

def test_scoreline_matches_outcome():
    rng = np.random.default_rng(1)
    m = np.full((4, 4), 1.0 / 16)
    h, a = sample_scoreline(m, outcome=0, rng=rng)  # home win
    assert h > a
    h, a = sample_scoreline(m, outcome=1, rng=rng)  # draw
    assert h == a

def test_knockout_no_draw():
    rng = np.random.default_rng(2)
    res = [knockout_winner(np.array([0.5, 0.4, 0.1]), rng) for _ in range(200)]
    assert set(res) <= {0, 1}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_matchengine.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write `src/worldcup/matchengine.py`**

```python
import numpy as np

def sample_outcome(wdl: np.ndarray, rng: np.random.Generator) -> int:
    return int(rng.choice(3, p=wdl / wdl.sum()))

def sample_scoreline(matrix: np.ndarray, outcome: int, rng: np.random.Generator):
    n = matrix.shape[0]
    mask = np.zeros_like(matrix)
    for h in range(n):
        for a in range(n):
            if (outcome == 0 and h > a) or (outcome == 1 and h == a) or (outcome == 2 and h < a):
                mask[h, a] = matrix[h, a]
    total = mask.sum()
    if total <= 0:
        # fallback to a minimal scoreline consistent with the outcome
        return (1, 0) if outcome == 0 else (0, 0) if outcome == 1 else (0, 1)
    flat = (mask / total).ravel()
    cell = rng.choice(flat.size, p=flat)
    return int(cell // n), int(cell % n)

def knockout_winner(wdl: np.ndarray, rng: np.random.Generator) -> int:
    p_home, p_draw, p_away = wdl
    denom = p_home + p_away
    p_adv_home = 0.5 if denom <= 0 else p_home / denom
    return 0 if rng.random() < p_adv_home else 1
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_matchengine.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/worldcup/matchengine.py tests/test_matchengine.py
git commit -m "[FEAT] matchengine: outcome + conditioned scoreline + knockout resolver"
```

---

## Task 7: Tournament — group sim, FIFA tiebreakers, best thirds, knockout

**Files:**
- Create: `src/worldcup/tournament.py`
- Test: `tests/test_tournament.py`

Interface:
- `round_robin_pairs(teams: list[str]) -> list[tuple[str,str]]` — all 6 unordered pairs (first listed = home).
- `rank_group(table: dict, h2h: dict) -> list[str]` — order teams by points, goal diff, goals for, then head-to-head points among tied, then stable order.
- `simulate_group(teams, prob_fn, score_fn, rng) -> tuple[list[str], dict]` — returns ranking and each team's stat dict (`pts`,`gd`,`gf`).
- `best_thirds(third_stats: dict, k: int = 8) -> list[str]` — top k third-placed teams by pts, gd, gf.
- `seed_bracket(qualifiers: list[str], strength: dict) -> list[tuple[str,str]]` — seed 32 teams 1v32, 2v31… by descending `strength`.
- `play_knockout(pairs, adv_fn, strength) -> str` — recursively resolve to a single champion; returns champion and the modal path is captured by the caller.

`prob_fn(home, away) -> np.ndarray` (W/D/L) and `score_fn(home, away, outcome, rng) -> (hg, ag)` and `adv_fn(home, away, rng) -> 0|1` are injected so the simulator stays model-agnostic and testable.

- [ ] **Step 1: Write the failing test `tests/test_tournament.py`**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tournament.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write `src/worldcup/tournament.py`**

```python
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
        table[h]["gf"] += hg; table[a]["gf"] += ag
        table[h]["gd"] += hg - ag; table[a]["gd"] += ag - hg
        if hg > ag:
            table[h]["pts"] += 3; h2h_pts[h] += 3
        elif hg < ag:
            table[a]["pts"] += 3; h2h_pts[a] += 3
        else:
            table[h]["pts"] += 1; table[a]["pts"] += 1
            h2h_pts[h] += 1; h2h_pts[a] += 1
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_tournament.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/worldcup/tournament.py tests/test_tournament.py
git commit -m "[FEAT] tournament: groups, FIFA tiebreakers, best-thirds, knockout"
```

---

## Task 8: Monte Carlo driver + aggregation

**Files:**
- Create: `src/worldcup/simulate.py`
- Test: `tests/test_simulate.py`

Interface:
- `build_match_probs(teams, elo_ratings, dc_params, cfg) -> tuple[dict, dict]` — precompute the ensemble W/D/L and the DC score matrix for every ordered pair once (perf: avoids re-pooling 50k times). Returns `(probs, matrices)`.
- `run_monte_carlo(cfg, elo_ratings, dc_params) -> pd.DataFrame` — runs `n_sims` tournaments; returns per-team columns `p_group_top2, p_r32, p_r16, p_qf, p_sf, p_final, p_champion`.
- (The representative ASCII bracket is assembled in `run.py` from the champion ranking; enriching it to a full R16→final path is documented under "Notes / known approximations".)

- [ ] **Step 1: Write the failing test `tests/test_simulate.py`**

```python
import numpy as np
import pandas as pd
from worldcup.simulate import run_monte_carlo

def _cfg():
    groups = {chr(65 + i): [f"t{i}{j}" for j in range(4)] for i in range(12)}
    return {
        "tournament": {"groups": groups},
        "simulation": {"n_sims": 50, "seed": 1},
        "elo": {"home_advantage": 0.0},
        "ensemble": {"match": {"elo": 0.5, "dc": 0.5}, "draw_param": 0.28},
        "dixon_coles": {"max_goals": 6},
    }

def test_champion_probs_sum_to_one():
    cfg = _cfg()
    teams = [f"t{i}{j}" for i in range(12) for j in range(4)]
    elo = {t: 1500.0 for t in teams}
    dc = {"attack": {t: 0.0 for t in teams}, "defence": {t: 0.0 for t in teams},
          "home_adv": 0.0, "rho": 0.0, "teams": teams}
    df = run_monte_carlo(cfg, elo, dc)
    assert abs(df["p_champion"].sum() - 1.0) < 1e-9
    assert (df["p_champion"] >= 0).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_simulate.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write `src/worldcup/simulate.py`**

```python
import numpy as np
import pandas as pd
from worldcup import elo as elo_mod
from worldcup import dixon_coles as dc_mod
from worldcup.ensemble import match_wdl
from worldcup.matchengine import sample_outcome, sample_scoreline, knockout_winner
from worldcup.tournament import simulate_group, best_thirds, seed_bracket

def build_match_probs(teams, elo_ratings, dc_params, cfg) -> dict:
    max_goals = cfg["dixon_coles"]["max_goals"]
    probs, matrices = {}, {}
    for h in teams:
        for a in teams:
            if h == a:
                continue
            e_wdl = elo_mod.match_probabilities(
                elo_ratings.get(h, 1500.0), elo_ratings.get(a, 1500.0), neutral=True, cfg=cfg)
            mat = dc_mod.score_matrix(dc_params, h, a, neutral=True, max_goals=max_goals)
            d_wdl = dc_mod.wdl_from_matrix(mat)
            probs[(h, a)] = match_wdl(e_wdl, d_wdl, cfg)
            matrices[(h, a)] = mat
    return probs, matrices

def _strength(elo_ratings, teams):
    return {t: elo_ratings.get(t, 1500.0) for t in teams}

def _simulate_once(groups, probs, matrices, strength, rng, track):
    qualifiers, third_stats, top2 = [], {}, []
    for gname, gteams in groups.items():
        def prob_fn(h, a):
            return probs[(h, a)]
        def score_fn(h, a, outcome, rng):
            return sample_scoreline(matrices[(h, a)], outcome, rng)
        ranking, table = simulate_group(gteams, prob_fn, score_fn, rng)
        qualifiers.extend(ranking[:2])
        top2.extend(ranking[:2])
        third = ranking[2]
        third_stats[third] = table[third]
    qualifiers.extend(best_thirds(third_stats, k=8))
    for t in top2:
        track["p_group_top2"][t] += 1
    pairs = seed_bracket(qualifiers, strength)
    for t in qualifiers:
        track["p_r32"][t] += 1
    # resolve knockout round by round, tracking how far each team goes.
    # 32 qualifiers -> winners are R16 (16), QF (8), SF (4), finalists (2), champion (1).
    round_labels = ["p_r16", "p_qf", "p_sf", "p_final", "p_champion"]
    current = pairs
    def adv_fn(h, a, rng):
        return knockout_winner(probs[(h, a)], rng)
    li = 0
    while True:
        winners = [h if adv_fn(h, a, rng) == 0 else a for h, a in current]
        if li < len(round_labels):
            for w in winners:
                track[round_labels[li]][w] += 1
        li += 1
        if len(winners) == 1:
            return winners[0]
        current = [(winners[i], winners[i + 1]) for i in range(0, len(winners), 2)]

def run_monte_carlo(cfg, elo_ratings, dc_params) -> pd.DataFrame:
    groups = cfg["tournament"]["groups"]
    teams = [t for g in groups.values() for t in g]
    probs, matrices = build_match_probs(teams, elo_ratings, dc_params, cfg)
    strength = _strength(elo_ratings, teams)
    cols = ["p_group_top2", "p_r32", "p_r16", "p_qf", "p_sf", "p_final", "p_champion"]
    track = {c: {t: 0 for t in teams} for c in cols}
    rng = np.random.default_rng(cfg["simulation"]["seed"])
    n = cfg["simulation"]["n_sims"]
    for _ in range(n):
        _simulate_once(groups, probs, matrices, strength, rng, track)
    df = pd.DataFrame({c: pd.Series(track[c]) for c in cols}) / n
    df.index.name = "team"
    return df.sort_values("p_champion", ascending=False)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_simulate.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/worldcup/simulate.py tests/test_simulate.py
git commit -m "[FEAT] simulate: Monte Carlo driver with per-round aggregation"
```

---

## Task 9: Backtest — log-loss / Brier + weight calibration

**Files:**
- Create: `src/worldcup/backtest.py`
- Test: `tests/test_backtest.py`

Interface:
- `log_loss(probs: list[np.ndarray], outcomes: list[int]) -> float`
- `brier(probs: list[np.ndarray], outcomes: list[int]) -> float` — multiclass Brier.
- `evaluate(results, cfg) -> dict` — split holdout by `backtest.holdout_days`; train Elo on the pre-holdout slice, fit DC at the split date; for each holdout match build the ensemble W/D/L; return `{"log_loss":…, "brier":…, "n":…}`.
- `calibrate_match_weights(results, cfg, grid) -> tuple[dict, float]` — grid-search Elo/DC weight split minimizing holdout log-loss; return best `{"elo":w,"dc":1-w}` and its score.

- [ ] **Step 1: Write the failing test `tests/test_backtest.py`**

```python
import numpy as np
from worldcup.backtest import log_loss, brier

def test_log_loss_perfect_is_zero():
    assert log_loss([np.array([1.0, 0.0, 0.0])], [0]) < 1e-9

def test_brier_perfect_is_zero():
    assert brier([np.array([0.0, 1.0, 0.0])], [1]) < 1e-9

def test_log_loss_penalizes_wrong():
    bad = log_loss([np.array([0.01, 0.01, 0.98])], [0])
    good = log_loss([np.array([0.98, 0.01, 0.01])], [0])
    assert bad > good
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_backtest.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write `src/worldcup/backtest.py`**

```python
import copy
import numpy as np
import pandas as pd
from worldcup import elo as elo_mod
from worldcup import dixon_coles as dc_mod
from worldcup.ensemble import match_wdl

def _result_index(hg, ag):
    return 0 if hg > ag else (1 if hg == ag else 2)

def log_loss(probs, outcomes) -> float:
    eps = 1e-12
    return float(-np.mean([np.log(max(p[o], eps)) for p, o in zip(probs, outcomes)]))

def brier(probs, outcomes) -> float:
    total = 0.0
    for p, o in zip(probs, outcomes):
        target = np.zeros(3)
        target[o] = 1.0
        total += np.sum((p - target) ** 2)
    return float(total / len(outcomes))

def _holdout_split(results, cfg):
    split = results["date"].max() - pd.Timedelta(days=cfg["backtest"]["holdout_days"])
    train = results[results["date"] <= split]
    test = results[results["date"] > split]
    return train, test, split

def evaluate(results, cfg) -> dict:
    train, test, split = _holdout_split(results, cfg)
    elo_ratings = elo_mod.compute_ratings(train, cfg)
    dc_params = dc_mod.fit(train, cfg, ref_date=split)
    max_goals = cfg["dixon_coles"]["max_goals"]
    probs, outcomes = [], []
    for _, m in test.iterrows():
        h, a, neutral = m["home_team"], m["away_team"], bool(m["neutral"])
        e_wdl = elo_mod.match_probabilities(
            elo_ratings.get(h, 1500.0), elo_ratings.get(a, 1500.0), neutral, cfg)
        mat = dc_mod.score_matrix(dc_params, h, a, neutral, max_goals)
        d_wdl = dc_mod.wdl_from_matrix(mat)
        probs.append(match_wdl(e_wdl, d_wdl, cfg))
        outcomes.append(_result_index(m["home_score"], m["away_score"]))
    return {"log_loss": log_loss(probs, outcomes),
            "brier": brier(probs, outcomes), "n": len(outcomes)}

def calibrate_match_weights(results, cfg, grid=None):
    grid = grid if grid is not None else [i / 10 for i in range(1, 10)]
    best_w, best_score = None, float("inf")
    for w in grid:
        trial = copy.deepcopy(cfg)
        trial["ensemble"]["match"] = {"elo": w, "dc": 1.0 - w}
        score = evaluate(results, trial)["log_loss"]
        if score < best_score:
            best_score, best_w = score, {"elo": w, "dc": 1.0 - w}
    return best_w, best_score
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_backtest.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/worldcup/backtest.py tests/test_backtest.py
git commit -m "[FEAT] backtest: log-loss/Brier metrics + match-weight calibration"
```

---

## Task 10: Report — Markdown + ASCII bracket + market delta

**Files:**
- Create: `src/worldcup/report.py`
- Test: `tests/test_report.py`

Interface:
- `champion_table(df: pd.DataFrame, blended: dict, top: int = 16) -> str` — Markdown table: team, model P(champion), market-blended P(champion), P(final), P(SF).
- `ascii_bracket(path: dict) -> str` — render the modal knockout path (R16→final) as monospaced text.
- `pure_vs_market_delta(model: dict, blended: dict, top: int = 10) -> str` — table of the biggest shifts the market introduces.
- `build_report(metrics, df, blended, modal_path, deltas_md) -> str` — assemble the full document.

- [ ] **Step 1: Write the failing test `tests/test_report.py`**

```python
import pandas as pd
from worldcup.report import champion_table, pure_vs_market_delta, build_report

def test_champion_table_contains_team():
    df = pd.DataFrame({"p_champion": [0.2], "p_final": [0.3], "p_sf": [0.4]}, index=["Brazil"])
    df.index.name = "team"
    out = champion_table(df, {"Brazil": 0.18})
    assert "Brazil" in out and "%" in out

def test_delta_flags_shift():
    md = pure_vs_market_delta({"A": 0.30, "B": 0.10}, {"A": 0.20, "B": 0.20})
    assert "A" in md and "B" in md

def test_build_report_has_sections():
    df = pd.DataFrame({"p_champion": [0.2], "p_final": [0.3], "p_sf": [0.4]}, index=["Brazil"])
    df.index.name = "team"
    doc = build_report({"log_loss": 1.0, "brier": 0.6, "n": 100}, df,
                        {"Brazil": 0.2}, "FINAL: Brazil", "delta")
    assert "## Methodology" in doc and "## Champion probabilities" in doc and "Brier" in doc
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_report.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write `src/worldcup/report.py`**

```python
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
        lines.append("FINAL: " + " vs ".join(path["FINAL"]) if isinstance(path["FINAL"], list) else f"FINAL: {path['FINAL']}")
    if "CHAMP" in path:
        lines.append(f"🏆 CHAMPION: {path['CHAMP']}")
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_report.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/worldcup/report.py tests/test_report.py
git commit -m "[FEAT] report: markdown report, ASCII bracket, market-delta table"
```

---

## Task 11: End-to-end pipeline + full test run

**Files:**
- Create: `run.py`
- Test: full suite

Interface: `run.py` ties everything together and writes `output/prediction.md` and `output/probabilities.csv`.

- [ ] **Step 1: Write `run.py`**

```python
import pandas as pd
from worldcup.config import load_config
from worldcup.ingest import load_results, load_market
from worldcup import elo as elo_mod
from worldcup import dixon_coles as dc_mod
from worldcup.market import devig
from worldcup.ensemble import blend_champion
from worldcup.teams import validate_groups, all_teams
from worldcup.simulate import run_monte_carlo
from worldcup.backtest import evaluate, calibrate_match_weights
from worldcup.report import build_report, pure_vs_market_delta, ascii_bracket

def main():
    cfg = load_config("config.yaml")
    validate_groups(cfg)
    results = load_results(cfg["data"]["results_csv"])

    best_w, score = calibrate_match_weights(results, cfg)
    cfg["ensemble"]["match"] = best_w
    print(f"calibrated match weights={best_w} holdout log-loss={score:.4f}")
    metrics = evaluate(results, cfg)

    ref = results["date"].max()
    elo_ratings = elo_mod.compute_ratings(results, cfg)
    dc_params = dc_mod.fit(results, cfg, ref_date=ref)

    df = run_monte_carlo(cfg, elo_ratings, dc_params)
    model_champ = df["p_champion"].to_dict()
    market = devig(load_market(cfg["data"]["market_csv"]))
    market = {t: market.get(t, 0.0) for t in all_teams(cfg)}
    blended = blend_champion(model_champ, market, cfg)

    top = df.head(1).index[0]
    modal = ascii_bracket({"CHAMP": top, "FINAL": top})
    deltas = pure_vs_market_delta(model_champ, blended)
    doc = build_report(metrics, df, blended, modal, deltas)

    with open("output/prediction.md", "w", encoding="utf-8") as fh:
        fh.write(doc)
    df.assign(p_champion_blended=pd.Series(blended)).to_csv("output/probabilities.csv")
    print("wrote output/prediction.md and output/probabilities.csv")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the full unit suite**

Run: `pytest -v`
Expected: all tests PASS.

- [ ] **Step 3: Run the real pipeline**

Run: `python run.py`
Expected: prints calibrated weights + log-loss, writes `output/prediction.md` and `output/probabilities.csv`. Open `prediction.md` and sanity-check: champion probabilities sum near 100%, favorites are plausible, Brier < 0.66 (better than uniform).

- [ ] **Step 4: Commit**

```bash
git add run.py output/prediction.md output/probabilities.csv
git commit -m "[FEAT] end-to-end pipeline → prediction report"
```

---

## Notes / known approximations

- **Knockout seeding** uses strength-based 1v32 seeding rather than FIFA's fixed group-slot bracket table. This is a deliberate, documented simplification; if the official R32 slotting table is entered later, replace `seed_bracket` with a config-driven slotting function.
- **Market** enters only at champion level (free sources publish futures, not per-match odds). The match-level ensemble is Elo+DC, matching the spec.
- The `modal_bracket` in `run.py` Step 1 is minimal (champion only); enrich by capturing a single `_simulate_once` path through R16→final if a fuller ASCII bracket is desired.
```
