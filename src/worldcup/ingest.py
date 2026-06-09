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
