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
