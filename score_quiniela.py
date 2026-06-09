"""Mide el accuracy de la quiniela una vez jugados los partidos.

Llena las columnas actual_home_goals / actual_away_goals en output/quiniela.csv
(deja vacías las que aún no se juegan) y corre:  python score_quiniela.py

Reporta dos métricas sobre los partidos ya jugados:
  - Aciertos de resultado (1/X/2)
  - Aciertos de marcador exacto
Reescribe las columnas result_hit / score_hit en el CSV.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import pandas as pd


def _played(row):
    for col in ("actual_home_goals", "actual_away_goals"):
        v = row[col]
        if pd.isna(v) or str(v).strip() == "":
            return False
    return True


def _token(hg, ag):
    return "1" if hg > ag else ("X" if hg == ag else "2")


def main(path="output/quiniela.csv"):
    df = pd.read_csv(path)
    played = df[df.apply(_played, axis=1)].copy()
    if played.empty:
        print("Aún no hay resultados cargados. Llena actual_home_goals/actual_away_goals.")
        return

    res_col, sc_col = [], []
    res_hits, score_hits = [], []
    for _, row in df.iterrows():
        if not _played(row):
            res_col.append("")
            sc_col.append("")
            continue
        ah, aa = int(float(row["actual_home_goals"])), int(float(row["actual_away_goals"]))
        ph, pa = int(row["pred_home_goals"]), int(row["pred_away_goals"])
        rh = int(_token(ph, pa) == _token(ah, aa))
        sh = int(ph == ah and pa == aa)
        res_col.append(rh)
        sc_col.append(sh)
        res_hits.append(rh)
        score_hits.append(sh)

    df["result_hit"] = res_col
    df["score_hit"] = sc_col
    df.to_csv(path, index=False, encoding="utf-8")
    n = len(res_hits)
    print(f"Partidos jugados evaluados: {n}")
    print(f"Accuracy de resultado (1/X/2): {sum(res_hits)}/{n} = {100*sum(res_hits)/n:.1f}%")
    print(f"Accuracy de marcador exacto:   {sum(score_hits)}/{n} = {100*sum(score_hits)/n:.1f}%")
    print(f"\nCSV actualizado: {path}")


if __name__ == "__main__":
    main()
