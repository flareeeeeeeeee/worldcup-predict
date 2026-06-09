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
