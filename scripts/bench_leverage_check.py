#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bench_leverage_check.py — leverage robustness for factor benchmarks
===================================================================
The factor benchmarks use unconstrained max-Sharpe weights w = Sigma^-1 mu.
Sharpe is scale-invariant, so the pooled Sharpe does not depend on the
(arbitrary) leverage scale — but the SDF portfolio (deep SDF, weights sum
to 1 by construction) and cumulative wealth do. This script:

  1. Documents the actual gross leverage (sum |w|) per model/window.
  2. Recomputes every benchmark with weights normalized to gross leverage 1
     (sum|w| = 1): Sharpe is unchanged by construction (sanity check), but
     monthly returns, min/max months, and cumulative wealth become the
     leverage-free objects that are comparable to the deep SDF portfolio.
  3. Saves results/bench_leverage.csv for the manuscript robustness table.

Outputs: results/bench_leverage.csv, results/bench_leverage_series.csv
"""
import csv
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from eval_core import (load_npz, load_rf, load_factors_ff5, load_factors_q,
                       shrunk_cov, max_sharpe_weights, sharpe_ann,
                       rolling_windows)

ROOT = Path(os.environ.get("DLAP_ROOT", str(Path.home() / "research/dlap-tse")))
# country-aware results dir (same pattern as e7/seed_sensitivity)
_C = os.environ.get("DLAP_COUNTRY", "").upper()
RES = ROOT / {"TR": "results_tr", "PK": "results_pk"}.get(_C, "results")
DATA = ROOT / "data"

N_PCA = 5


def gross_leverage(w):
    return float(np.abs(w).sum())


def normalize(w):
    s = np.abs(w).sum()
    return w / s if s > 0 else w


def wealth(r):
    return float(np.prod(1.0 + r))


def run():
    arr, dates, variables, tickers = load_npz()
    rf = load_rf()
    ff5 = load_factors_ff5()
    q = load_factors_q()
    common = [d for d in dates if (int(d[:4]), int(d[5:7])) in ff5 and
              (int(d[:4]), int(d[5:7])) in q]
    pos = {d: i for i, d in enumerate(dates)}
    idx = [pos[d] for d in common]
    T = len(common)

    R = arr[idx, :, 0].astype(float)
    R[R < -50.0] = np.nan
    rf_v = np.array([rf.get(d, np.nan) for d in common])
    R_exc = R - rf_v[:, None]

    F_ff5 = np.array([[ff5[(int(d[:4]), int(d[5:7]))][k] for k in
                       ["Mkt_RF", "SMB", "HML", "RMW", "CMA"]] for d in common])
    F_q = np.array([[q[(int(d[:4]), int(d[5:7]))][k] for k in
                      ["Mkt_RF", "ME", "IA", "ROE"]] for d in common])
    F_mkt = F_ff5[:, :1]

    windows = list(rolling_windows(list(range(T)), train=60, test=12))

    models = {"Market": F_mkt, "FF5": F_ff5, "q-factor": F_q}

    rows = []
    series = []
    # factor models
    for name, F_all in models.items():
        rp_unc, rp_norm, levs = [], [], []
        for tr_idx, te_idx in windows:
            F_tr, F_te = F_all[tr_idx], F_all[te_idx]
            w = max_sharpe_weights(F_tr)
            levs.append(gross_leverage(w))
            rp_unc.append(F_te @ w)
            rp_norm.append(F_te @ normalize(w))
        ru = np.concatenate(rp_unc)
        rn = np.concatenate(rp_norm)
        rows.append({
            "model": name,
            "sharpe_raw": sharpe_ann(ru),
            "sharpe_norm1": sharpe_ann(rn),
            "gross_lev_mean": float(np.mean(levs)),
            "gross_lev_max": float(np.max(levs)),
            "min_month_raw_pct": ru.min() * 100,
            "min_month_norm1_pct": rn.min() * 100,
            "wealth_raw": wealth(ru),
            "wealth_norm1": wealth(rn),
        })
        for v in rn:
            series.append((f"{name}-norm1", v))

    # PCA(5)
    rp_unc, rp_norm, levs = [], [], []
    for tr_idx, te_idx in windows:
        R_tr, R_te = R_exc[tr_idx], R_exc[te_idx]
        Rt = np.where(np.isfinite(R_tr), R_tr, 0.0)
        S = shrunk_cov(Rt)
        _, _, Vt = np.linalg.svd(S)
        V = Vt[:N_PCA].T
        F_tr = Rt @ V
        F_te = np.where(np.isfinite(R_te), R_te, 0.0) @ V
        w = max_sharpe_weights(F_tr)
        levs.append(gross_leverage(w))
        rp_unc.append(F_te @ w)
        rp_norm.append(F_te @ normalize(w))
    ru = np.concatenate(rp_unc)
    rn = np.concatenate(rp_norm)
    rows.append({
        "model": "PCA(5)",
        "sharpe_raw": sharpe_ann(ru),
        "sharpe_norm1": sharpe_ann(rn),
        "gross_lev_mean": float(np.mean(levs)),
        "gross_lev_max": float(np.max(levs)),
        "min_month_raw_pct": ru.min() * 100,
        "min_month_norm1_pct": rn.min() * 100,
        "wealth_raw": wealth(ru),
        "wealth_norm1": wealth(rn),
    })
    for v in rn:
        series.append(("PCA(5)-norm1", v))

    with open(RES / "bench_leverage.csv", "w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wcsv.writeheader()
        wcsv.writerows(rows)
    with open(RES / "bench_leverage_series.csv", "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["model", "oos_return"])
        wcsv.writerows(series)

    print(f"{'model':<10}{'Sharpe(raw)':>12}{'Sharpe(|w|=1)':>14}"
          f"{'lev mean':>9}{'lev max':>8}{'min raw%':>9}{'min n1%':>8}"
          f"{'wealth raw':>11}{'wealth n1':>10}")
    for r in rows:
        print(f"{r['model']:<10}{r['sharpe_raw']:>12.3f}{r['sharpe_norm1']:>14.3f}"
              f"{r['gross_lev_mean']:>9.2f}{r['gross_lev_max']:>8.2f}"
              f"{r['min_month_raw_pct']:>9.1f}{r['min_month_norm1_pct']:>8.1f}"
              f"{r['wealth_raw']:>11.2f}{r['wealth_norm1']:>10.2f}")


if __name__ == "__main__":
    run()
