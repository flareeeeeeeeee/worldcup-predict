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


def most_likely_score(matrix: np.ndarray):
    """Modal scoreline (home_goals, away_goals) of a Dixon-Coles score matrix."""
    n = matrix.shape[0]
    cell = int(np.argmax(matrix))
    return cell // n, cell % n


def modal_score_given_outcome(matrix: np.ndarray, outcome: int):
    """Most likely scoreline consistent with a given outcome (0 home, 1 draw, 2 away).

    Keeps the predicted score and the predicted 1X2 result aligned: e.g. if the most
    likely outcome is a home win, returns the most probable home-win scoreline.
    """
    n = matrix.shape[0]
    best, best_p = None, -1.0
    for h in range(n):
        for a in range(n):
            ok = ((outcome == 0 and h > a) or (outcome == 1 and h == a)
                  or (outcome == 2 and h < a))
            if ok and matrix[h, a] > best_p:
                best_p, best = matrix[h, a], (h, a)
    if best is None:
        return (1, 0) if outcome == 0 else (0, 0) if outcome == 1 else (0, 1)
    return best


def knockout_winner(wdl: np.ndarray, rng: np.random.Generator) -> int:
    p_home, p_draw, p_away = wdl
    denom = p_home + p_away
    p_adv_home = 0.5 if denom <= 0 else p_home / denom
    return 0 if rng.random() < p_adv_home else 1
