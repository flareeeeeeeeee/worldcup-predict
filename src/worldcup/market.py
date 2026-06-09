def devig(odds: dict) -> dict:
    implied = {t: 1.0 / o for t, o in odds.items()}
    total = sum(implied.values())
    return {t: v / total for t, v in implied.items()}
