#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
e7_subperiod.py — DLAP-TSE E7: 2020 boom-bust regime stability
================================================================
Slices the pooled OOS SDF portfolio return series (12 windows x 12 months,
in window order, common period 2008-07..2026-06) into subperiods:

  FULL     2013-07..2026-06  (144 months)
  BOOMBUST 2019-07..2022-06  (windows 6-8: COVID crash + 1399 boom + bust)
  BOOM     2020-07..2022-06  (windows 7-8)
  CALM     everything else

Reports annualized Sharpe per model per subperiod.
"""
import csv
import os
import math
from pathlib import Path

import numpy as np

_COUNTRY = os.environ.get("DLAP_COUNTRY", "").upper()
_RESDIR = {"TR": "results_tr", "PK": "results_pk"}.get(_COUNTRY, "results")
RES = Path(os.environ.get("DLAP_ROOT", str(Path.home() / "research/dlap-tse"))) / _RESDIR

WINDOW_STARTS = ["2013-07", "2014-07", "2015-07", "2016-07", "2017-07",
                 "2018-07", "2019-07", "2020-07", "2021-07", "2022-07",
                 "2023-07", "2024-07"]


def sharpe(r):
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    if len(r) < 6 or r.std() == 0:
        return math.nan
    return r.mean() / r.std() * math.sqrt(12)


def load_series(fname):
    with open(RES / fname, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return np.array([float(r["oos_return"]) for r in rows])


MODELS = {
    "Market": "e1_pooled_series.csv",
    "FF5": "e1_pooled_series.csv",
    "q-factor": "e1_pooled_series.csv",
    "PCA(5)": "e1_pooled_series.csv",
    "LASSO": "e1_pooled_series.csv",
    "E2": "e2_pooled_series.csv",
    "E3": "e3_pooled_series.csv",
    "E4B": "e4b_pooled_series.csv",
    "E5A": "e5a_pooled_series.csv",
}
# e1 file has a model column — filter per model
E1_MODELS = ["Market", "FF5", "q-factor", "PCA(5)", "LASSO"]


def load_all():
    out = {}
    e1 = {}
    with open(RES / "e1_pooled_series.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            e1.setdefault(r["model"], []).append(float(r["oos_return"]))
    for m in E1_MODELS:
        out[m] = np.array(e1[m])
    for m in ["E2", "E3", "E4B", "E5A"]:
        out[m] = load_series(f"{m.lower()}_pooled_series.csv")
    return out


def main():
    series = load_all()
    T = len(next(iter(series.values())))
    print(f"  pooled OOS series length: {T} months ({T // 12} windows)")
    if T == 144:
        slices = {
            "FULL": (0, 144),
            "BOOMBUST": (72, 108),   # windows 6-8 (IR 2020-22)
            "BOOM": (84, 108),       # windows 7-8
            "CALM": None,            # windows 0-5 + 9-11
        }
        calm_idx = list(range(0, 72)) + list(range(108, 144))
    else:
        half = T // 2
        slices = {"FULL": (0, T), "HALF1": (0, half), "HALF2": (half, T)}
        calm_idx = None

    print(f"{'model':<10}" + "".join(f"{k:>11}" for k in slices) + f"{'win6-8-sh':>12}")
    print("-" * 10 + "-" * 11 * len(slices) + "-" * 12)
    for m, s in series.items():
        row = f"{m:<10}"
        for k, sl in slices.items():
            seg = s[slice(*sl)] if sl is not None else s[calm_idx]
            row += f"{sharpe(seg):>11.3f}"
        per_win = [sharpe(s[i:i + 12]) for i in range(0, T, 12)]
        if len(per_win) >= 6:
            row += f"{float(np.mean(per_win[6:9])):>12.3f}"
        print(row)

    # per-window Sharpe matrix
    print("\nPer-window Sharpe (annualized, 12-month windows):")
    print(f"{'model':<10}" + "".join(f"w{i:<8}" for i in range(T // 12)))
    print("-" * 10 + "-" * 8 * (T // 12))
    for m, s in series.items():
        row = f"{m:<10}"
        for i in range(T // 12):
            row += f"{sharpe(s[i * 12:(i + 1) * 12]):>8.2f}"
        print(row)


if __name__ == "__main__":
    main()
