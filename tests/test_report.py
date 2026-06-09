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
