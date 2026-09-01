#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ipca_benchmark.py — IPCA (Kelly, Pruitt & Su 2019) benchmark for DLAP-TSE
==========================================================================
Rolling-window: train 60m → test 12m.
OOS metric: cross-sectional predicted-vs-realized return correlation (EV proxy),
            and tangency-SDF-portfolio Sharpe from predicted factor loadings.

IPCA: E[R] = Γ x_i · f  (expected return = loadings × factors)
      Loadings: λ_i,t = Γ' x_i,t
      Portfolio: equal-weight on top vs bottom quintile of predicted returns.
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("DLAP_ROOT", str(Path.home() / "research/dlap-tse")))
COUNTRY = os.environ.get("DLAP_COUNTRY", "").upper()
if COUNTRY == "TR":
    DATA, RES = ROOT / "data_tr", ROOT / "results_tr"
elif COUNTRY == "PK":
    DATA, RES = ROOT / "data_pk", ROOT / "results_pk"
else:
    DATA, RES = ROOT / "data", ROOT / "results"


def load_all():
    d = np.load(DATA / "Char_all.npz", allow_pickle=True)
    X = d["data"].astype(np.float64)
    dates = list(d["date"])
    variables = list(d["variable"])
    tickers = list(d["ticker"])

    ret_dict = {}
    with open(DATA / "characteristics_panel.csv", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            ds = f"{int(row['year'])}-{int(row['month']):02d}"
            try:
                ret_dict[(ds, row["ticker"])] = float(row["ret_monthly"])
            except (ValueError, KeyError):
                pass

    rf_path = DATA / "risk_free_rate.csv"
    if not rf_path.exists():
        rf_path = ROOT.parent / "fama-five" / "data" / "risk_free_rate.csv"
    rf_dict = {}
    if rf_path.exists():
        with open(rf_path, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                try:
                    rf_dict[row["month"][:7]] = float(row["monthly_rate_pct"])
                except (ValueError, KeyError):
                    pass

    R = np.full((len(dates), len(tickers)), np.nan)
    for ti, ds in enumerate(dates):
        rf = rf_dict.get(ds, 0.0)
        for ni, tk in enumerate(tickers):
            raw = ret_dict.get((ds, tk))
            if raw is not None:
                R[ti, ni] = raw - rf

    return X, R, dates, tickers, variables


def rolling_windows(T, train=60, test=12):
    for s in range(0, T - train - test + 1, test):
        yield list(range(s, s + train)), list(range(s + train, s + train + test))


def sharpe_ann(r):
    r = np.asarray(r)
    r = r[np.isfinite(r)]
    if len(r) < 2 or r.std() == 0:
        return 0.0
    return r.mean() / r.std() * np.sqrt(12)


def run_ipca_window(X_train, R_train, X_test, R_test, n_factors, use_intercept=True):
    """Fit IPCA on train, evaluate OOS on test.
    
    OOS procedure:
    1. Fit IPCA on training data: y = Γ'x · f
    2. For each test month t:
       a. Loadings: λ_i = Γ' x_i,t  (for all stocks)
       b. Factor-mimicking portfolio weight: w_i = λ_i · f̂_T (last train factor)
       c. Portfolio return: r_p = Σ w_i r_i,t
    3. Pool portfolio returns → Sharpe.
    
    Also: predicted returns (cross-section) = λ_i · f̂_T, compared to realized
    cross-section for EV.
    """
    T_tr, N, L = X_train.shape
    T_te = X_test.shape[0]

    rows_X, rows_y = [], []
    for ti in range(T_tr):
        for ni in range(N):
            x = X_train[ti, ni]
            r = R_train[ti, ni]
            if not np.all(np.isfinite(x)) or not np.isfinite(r):
                continue
            rows_X.append([ni + 1, ti] + list(x))
            rows_y.append(r)

    if len(rows_X) < (n_factors + 1) * L + 50:
        return None

    cols = ["entity", "time"] + [f"c{i}" for i in range(L)]
    X_df = pd.DataFrame(rows_X, columns=cols).set_index(["entity", "time"])
    y_s = pd.Series(rows_y, index=X_df.index, name="ret")

    from ipca import InstrumentedPCA
    try:
        model = InstrumentedPCA(n_factors=n_factors, intercept=use_intercept,
                                max_iter=500, iter_tol=1e-3)
        model.fit(X=X_df, y=y_s)
    except Exception as e:
        return None

    Gamma = model.Gamma       # (L, K) or (L, K+1) with intercept
    Factors = model.Factors   # (K, T_tr) or (K+1, T_tr) with intercept
    f_last = Factors[:, -1]   # last estimated factor

    test_rp = []
    all_alpha_sq = []

    for ti in range(T_te):
        x_test = X_test[ti]  # (N, L)
        r_test = R_test[ti]  # (N,)

        # Loadings for test month
        lambdas = x_test @ Gamma  # (N, K) or (N, K+1)

        # Factor-mimicking weight: w_i = λ_i' f
        raw_w = lambdas @ f_last  # (N,)
        abs_w = np.abs(raw_w)
        w_sum = abs_w.sum()
        if w_sum < 1e-12 or not np.isfinite(w_sum):
            continue
        w_norm = raw_w / w_sum

        r_p = np.nansum(w_norm * r_test)
        if not np.isfinite(r_p):
            continue
        test_rp.append(r_p)

        # Pricing errors
        valid = np.isfinite(r_test)
        errs = (r_test[valid] - r_p) ** 2
        all_alpha_sq.extend(errs.tolist())

    if len(test_rp) < 3:
        return None
    return np.array(test_rp), np.array(all_alpha_sq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-factors", type=int, nargs="+", default=[3, 5, 8])
    args = ap.parse_args()

    X, R, dates, tickers, variables = load_all()
    T, N, L = X.shape
    print(f"== IPCA: {COUNTRY or 'IR'}, T={T}, N={N}, L={L}, "
          f"{dates[0]}..{dates[-1]} ==")

    windows = list(rolling_windows(T, train=60, test=12))
    print(f"  {len(windows)} windows")
    use_intercept = N >= 400

    results = {}
    for n_fac in args.n_factors:
        pooled_rp, all_alpha_sq, n_valid = [], [], 0

        for wi, (w_tr, w_te) in enumerate(windows):
            out = run_ipca_window(X[w_tr], R[w_tr], X[w_te], R[w_te],
                                   n_fac, use_intercept=use_intercept)
            if out is None:
                continue
            rp, alpha_sq = out
            pooled_rp.extend(rp.tolist())
            all_alpha_sq.extend(alpha_sq.tolist())
            n_valid += 1

        if not pooled_rp:
            print(f"  K={n_fac}: no valid windows")
            continue

        rp = np.array(pooled_rp)
        shp = sharpe_ann(rp)
        rms = np.sqrt(np.mean(all_alpha_sq)) * 100
        results[n_fac] = {"sharpe": shp, "rms_pct": rms, "n_win": n_valid}
        print(f"  K={n_fac}: Sharpe={shp:.4f}, RMS={rms:.2f}%, w={n_valid}")

    if not results:
        print("ERROR: all IPCA specs failed")
        sys.exit(1)

    best_k = max(results, key=lambda k: results[k]["sharpe"])
    b = results[best_k]
    print(f"\n== BEST: K={best_k}, Sharpe={b['sharpe']:.4f}, RMS={b['rms_pct']:.2f}% ==")

    out_path = RES / "ipca_results.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", f"IPCA({best_k})"])
        for k, v in sorted(results.items()):
            w.writerow([f"IPCA({k})_sharpe", f"{v['sharpe']:.4f}"])
            w.writerow([f"IPCA({k})_rms_pct", f"{v['rms_pct']:.3f}"])
            w.writerow([f"IPCA({k})_windows", v["n_win"]])
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
