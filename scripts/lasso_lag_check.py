#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LASSO benchmark with properly lagged characteristics (x_{t-1} -> r_t)."""
import csv
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, 'scripts')
from eval_core import load_npz, load_rf, load_factors_ff5, load_factors_q, rolling_windows
from run_e1 import eval_lasso

arr, dates, variables, tickers = load_npz()
rf = load_rf()
ff5 = load_factors_ff5()
q = load_factors_q()
common = [d for d in dates if (int(d[:4]), int(d[5:7])) in ff5 and (int(d[:4]), int(d[5:7])) in q]
pos = {d: i for i, d in enumerate(dates)}
idx = [pos[d] for d in common]
R = arr[idx, :, 0].astype(float)
R[R < -50.0] = np.nan
rf_v = np.array([rf.get(d, np.nan) for d in common])
R_exc = R - rf_v[:, None]
X = arr[idx, :, 1:].astype(float)
X[X < -50.0] = np.nan

# ── lag the characteristics by one month (x_{t-1} prices r_t) ──
X_lag = np.concatenate([np.full((1, X.shape[1], X.shape[2]), np.nan), X[:-1]], axis=0)

windows = list(rolling_windows(list(range(len(common))), train=60, test=12))

res = eval_lasso(R_exc, X_lag, common, windows, "LASSO-lagged")
print("LASSO-lagged:", {k: round(v, 4) if isinstance(v, float) else v
                        for k, v in res.items() if k != 'pooled_rp'})
rp = res['pooled_rp']
with open('results/lasso_lag_pooled_series.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['model', 'oos_return'])
    for v in rp:
        w.writerow(['LASSO-lagged', f'{v:.6f}'])
print("sharpe:", round(rp.mean() / rp.std() * np.sqrt(12), 4))
print("n months:", len(rp))
