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
