#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_core.py — DLAP-TSE Phase 2 shared evaluation module
==========================================================
Implements the CPZ (2024) evaluation protocol for benchmark models:
  - Rolling windows: train 60 months -> test 12 months (step 12)
  - SDF portfolio return: r_p,t = w' F_t  (factor models) or
    r_p,t = sum_i omega_t(i) R^e_{t,i} / sum_i |omega_t(i)|  (deep SDF, CPZ)
  - Metrics (pooled over all OOS months):
      * OOS Sharpe (annualized, x sqrt(12))
      * Cross-sectional R^2 of mean excess returns (per test window, n-weighted avg)
      * Pricing errors: alpha_i = E[M R^e_i] -> RMS and max (monthly, in %)
      * HJ bound of test assets: sqrt(mu' Sigma^-1 mu) (train cov, shrunk)
  - Covariance shrinkage: Sigma' = (1-d) Sigma + d diag(Sigma), d = 0.2
"""
import csv
import os
import math
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("DLAP_ROOT", str(Path.home() / "research/dlap-tse")))
DATA = ROOT / "data"
FAMA = Path(os.environ.get("FAMA_ROOT", str(Path.home() / "research/fama-five/data")))

UNKNOWN = -99.99
SHRINK = 0.2


# ── loaders ──────────────────────────────────────────────────────────────
def load_npz():
    d = np.load(DATA / "Char_all.npz", allow_pickle=True)
    return (d["data"].astype(np.float64), list(d["date"]),
            list(d["variable"]), list(d["ticker"]))


def load_rf():
    """month 'YYYY-MM' -> monthly risk-free rate (decimal)"""
    out = {}
    with open(FAMA / "risk_free_rate.csv", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            out[row["month"][:7]] = float(row["monthly_rate_pct"])
    return out


def load_factors_ff5():
    """(y,m) -> dict of FF5 factor returns (excess, decimal).

    Winsorized variant (data/factors_winsorized/factors_2x3.csv): built from
    the same per-month 1%/99% winsorized stock returns as the deep-SDF panel,
    so benchmark inputs are symmetric with the SDF data prep (the raw-based
    fama-five factors contain capital-increase artifacts, e.g. RMW -95.6% in
    2009-09, which the deep SDF never saw).
    """
    out = {}
    with open(DATA / "factors_winsorized" / "factors_2x3.csv",
              encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            y, m = int(row["year"]), int(row["month"])
            out[(y, m)] = {k: float(row[k]) for k in
                           ["Mkt_RF", "SMB", "HML", "RMW", "CMA"]}
    return out


def load_factors_q():
    out = {}
    with open(DATA / "factors_winsorized" / "factors_q.csv",
              encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            y, m = int(row["year"]), int(row["month"])
            out[(y, m)] = {k: float(row[k]) for k in ["Mkt_RF", "ME", "IA", "ROE"]}
    return out


def lag_align(X, macro):
    """Proper t -> t+1 alignment: return of month t is priced by
    characteristics and macro state observed at month t-1 (canonical
    CPZ/GKX x_t -> r_{t+1} convention). Shifts X and macro DOWN one row;
    row 0 becomes NaN (masked in training)."""
    X_lag = np.concatenate(
        [np.full((1, X.shape[1], X.shape[2]), np.nan), X[:-1]], axis=0)
    macro_lag = np.concatenate(
        [np.full((1, macro.shape[1]), np.nan), macro[:-1]], axis=0)
    if len(macro_lag) > 1:
        macro_lag[0] = macro_lag[1]
    return X_lag, macro_lag


# ── helpers ──────────────────────────────────────────────────────────────
def shrunk_cov(X):
    """Cov of X (T x K), shrinkage toward diagonal."""
    T = X.shape[0]
    if T < 3:
        return np.eye(X.shape[1]) * 1e-6
    mu = X.mean(axis=0)
    Xc = X - mu
    S = Xc.T @ Xc / (T - 1)
    d = np.diag(S)
    S_shrunk = (1 - SHRINK) * S + SHRINK * np.diag(d)
    return S_shrunk


def max_sharpe_weights(F):
    """w = Sigma^-1 mu for excess factor returns F (T x K)."""
    mu = F.mean(axis=0)
    S = shrunk_cov(F)
    try:
        w = np.linalg.solve(S, mu)
    except np.linalg.LinAlgError:
        w = np.linalg.pinv(S) @ mu
    return w


def sharpe_ann(r):
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 3 or r.std() == 0:
        return math.nan
    return r.mean() / r.std() * math.sqrt(12)


def xs_r2(pred_mean, real_mean):
    """1 - SS_res/SS_tot over stocks (both arrays of same length, finite)."""
    p = np.asarray(pred_mean, float)
    r = np.asarray(real_mean, float)
    m = np.isfinite(p) & np.isfinite(r)
    if m.sum() < 5:
        return math.nan
    p, r = p[m], r[m]
    ss_tot = float(((r - r.mean()) ** 2).sum())
    if ss_tot == 0:
        return math.nan
    return 1.0 - float(((r - p) ** 2).sum()) / ss_tot


def ols_betas(y, F):
    """Time-series betas of y (T) on factor excess returns F (T x K), with const."""
    T = len(y)
    X = np.column_stack([np.ones(T), F])
    mask = np.isfinite(y) & np.isfinite(X).all(axis=1)
    if mask.sum() < 12:
        return None
    Xm, ym = X[mask], y[mask]
    try:
        beta = np.linalg.lstsq(Xm, ym, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None
    return beta  # [const, betas...]


# ── window generator ─────────────────────────────────────────────────────
def rolling_windows(months, train=60, test=12):
    """Yield (train_months, test_months) index lists."""
    T = len(months)
    for s in range(train, T - test + 1, test):
        yield months[s - train:s], months[s:s + test]


# ── metric aggregation ───────────────────────────────────────────────────
def aggregate(per_window):
    """per_window: list of dicts with sharpe, r2, rms_alpha, max_alpha,
    plus pooled raw series via 'pooled' dict. Returns summary dict."""
    sharps = [w["sharpe"] for w in per_window if w.get("sharpe") == w.get("sharpe")]
    r2s = [(w["r2"], w["n_stocks"]) for w in per_window if w.get("r2") == w.get("r2")]
    rms = [w["rms_alpha"] for w in per_window if w.get("rms_alpha") == w.get("rms_alpha")]
    mx = [w["max_alpha"] for w in per_window if w.get("max_alpha") == w.get("max_alpha")]
    n = len([w for w in per_window if w.get("n_oos")])
    return {
        "n_windows": n,
        "sharpe_mean": float(np.mean(sharps)) if sharps else math.nan,
        "sharpe_median": float(np.median(sharps)) if sharps else math.nan,
        "r2": float(np.average([r for r, _ in r2s], weights=[s for _, s in r2s])) if r2s else math.nan,
        "rms_alpha_pct": float(np.mean(rms)) * 100 if rms else math.nan,
        "max_alpha_pct": float(np.max(mx)) * 100 if mx else math.nan,
    }
