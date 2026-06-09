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
