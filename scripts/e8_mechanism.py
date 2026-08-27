#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
e8_mechanism.py — DLAP-TSE v0.6: WHY does the liquidity filter work?
====================================================================
Mechanism analysis for the paper's sharpest portfolio-level finding (E8):
excluding the bottom 5% of stocks by train-window turnover raises the SDF
portfolio Sharpe (0.363 -> 0.819) while leaving pricing errors and EV almost
unchanged.

Hypothesis tested here (one coherent story):
  the deep SDF concentrates portfolio weight on the illiquid tail, whose
  extreme returns are largely microstructure noise (price-limit truncation,
  stale prices, retail order flow) rather than priced risk. In the weighted
  squared-pricing-error loss this noise is averaged away (the tail is only
  5% of the count-weighted loss), so pricing performance is untouched; in
  the SDF portfolio the same noise is concentrated and levered, so the
  portfolio collapses. Dropping the tail removes the noise from the
  portfolio without removing priced information.

Inputs (all pre-existing, no retraining):
  results/mechanism_dump/window_*.npz   (from train_e2.py --dump-mechanism,
                                         the E2 spec: sy 11 chars, lstm, seed 42)
  fama-five/data/tsetmc/{ticker}_prices.json  (daily TSETMC: priceChange,
                                         priceYesterday, qTotTran5J, zTotTran, qTotCap)
  fama-five/data/processed/monthly_returns.csv (n_days, monthly returns)

Outputs:
  results/e8_mechanism_stats.csv        (tail vs kept comparison table)
  results/e8_mechanism_by_window.csv    (per-window weight shares, for Figure)
  paper/figures/fig_e8_mechanism.pdf    (2-panel figure)
"""
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("DLAP_ROOT", str(Path.home() / "research/dlap-tse")))
# country-aware results dir (same pattern as e7/seed_sensitivity)
_C = os.environ.get("DLAP_COUNTRY", "").upper()
RES = ROOT / {"TR": "results_tr", "PK": "results_pk"}.get(_C, "results")
DUMP = RES / "mechanism_dump"
TSETMC = Path.home() / "research/fama-five/data/tsetmc"
MONTHLY = Path.home() / "research/fama-five/data/processed/monthly_returns.csv"
FIG_DIR = ROOT / "paper/figures"

# daily-stat window: the full OOS evaluation period of the paper
D0, D1 = 20130701, 20250630
# TSE daily price band is +-5%; "closed at limit" proxy with tolerance
LIM_LO, LIM_HI = 0.049, 0.055

AMIHUD_SCALE = 1e9  # report mean(|ret|/value) in 1e-9 units (rials are huge)


def load_dumps():
    """Return list of per-window dicts."""
    out = []
    for p in sorted(DUMP.glob("window_*.npz")):
        d = np.load(p, allow_pickle=True)
        out.append({
            "wi": int(p.stem.split("_")[1]),
            "tickers": list(d["tickers"]),
            "omega": d["omega_te"],           # (12, N)
            "alpha": d["alpha_te"],           # (N,)
            "R": d["R_te"],                   # (12, N)
            "months": list(d["months_te"]),
            "tr_turn": d["tr_turn"],          # (N,) mean train-window turnover
            "thr": float(d["thr"]),
        })
    return out


def load_monthly():
    """ticker -> dict(year-month -> (ret_monthly, n_days)) over test period."""
    out = {}
    with open(MONTHLY, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            y, m = int(r["year"]), int(r["month"])
            if not ((2013, 7) <= (y, m) <= (2025, 6)):
                continue
            out.setdefault(r["ticker"], {})[f"{y:04d}-{m:02d}"] = (
                float(r["ret_monthly"]), int(r["n_days"]))
    return out


def daily_stats(ticker):
    """Per-stock daily microstructure stats over [D0, D1]."""
    p = TSETMC / f"{ticker}_prices.json"
    if not p.exists():
        return None
    try:
        recs = json.load(open(p, encoding="utf-8"))["prices"]
    except Exception:
        return None
    rows = [r for r in recs if D0 <= int(r["dEven"]) <= D1 and r["pClosing"] > 0]
    if len(rows) < 60:
        return None
    pc = np.array([r["priceChange"] for r in rows], float)
    py = np.array([r["priceYesterday"] for r in rows], float)
    val = np.array([r.get("qTotCap", 0) or 0 for r in rows], float)
    trd = np.array([r.get("zTotTran", 0) or 0 for r in rows], float)
    ratio = np.abs(pc) / np.where(py > 0, py, np.nan)
    ok = np.isfinite(ratio) & (ratio > 0)
    # ex-day / capital-change filter: |dP|/P within the band + small epsilon
    clean = ok & (ratio <= LIM_HI + 0.005)
    if clean.sum() < 60:
        return None
    limit_hit = (ratio >= LIM_LO) & (ratio <= LIM_HI)
    amihud = np.nanmean(np.where(clean & (val > 0), ratio / np.where(val > 0, val, np.nan), np.nan))
    trades = trd[trd > 0]
    return {
        "limit_hit_freq": float(limit_hit[ok].mean()),
        "amihud": float(amihud * AMIHUD_SCALE),
        "trades_median": float(np.median(trades)) if len(trades) else np.nan,
    }


def monthly_stats(ticker, mrows):
    """Per-stock monthly stats over the test period."""
    d = mrows.get(ticker)
    if not d or len(d) < 24:
        return None
    months = sorted(d)
    rets = np.array([d[m][0] for m in months])
    ndays = np.array([d[m][1] for m in months])
    m = np.isfinite(rets) & (rets > -50)
    rets = rets[m]
    if len(rets) < 24:
        return None
    r = rets - rets.mean()
    ar1 = float(np.corrcoef(r[:-1], r[1:])[0, 1]) if len(r) > 3 and r.std() > 0 else np.nan
    return {
        "n_days_mean": float(ndays[m].mean()),
        "thin_frac": float((ndays[m] <= 3).mean()),
        "vol_monthly": float(rets.std()),
        "ar1": ar1,
        "max_ret": float(rets.max()),
    }


def main():
    wins = load_dumps()
    print(f"loaded {len(wins)} windows from {DUMP}")
    monthly = load_monthly()

    # ---- per-window group membership and SDF-weight shares -----------------
    wrows = []
    for w in wins:
        tt = w["tr_turn"]
        tail = tt <= w["thr"]              # E8 exclusion rule, exact
        om = w["omega"]                    # (12, N)
        absom = np.abs(om)
        wshare_abs = absom[:, tail].sum(axis=1) / absom.sum(axis=1)
        wshare_pos = (om[:, tail].clip(min=0).sum(axis=1)
                      / om.clip(min=0).sum(axis=1))
        a = w["alpha"]
        a2 = np.nan_to_num(a ** 2)
        a2share = a2[tail].sum() / a2.sum() if a2.sum() > 0 else np.nan
        rms_tail = float(np.sqrt(np.nanmean(a[tail] ** 2))) * 100
        rms_kept = float(np.sqrt(np.nanmean(a[~tail] ** 2))) * 100
        # EW portfolio returns (excess, pooled over the 12 OOS months)
        R = w["R"]
        ew_tail = np.nanmean(R[:, tail], axis=1)
        ew_kept = np.nanmean(R[:, ~tail], axis=1)
        wrows.append({
            "window": w["wi"],
            "month_start": w["months"][0],
            "n_tail": int(tail.sum()),
            "n_kept": int((~tail).sum()),
            "wshare_abs": float(wshare_abs.mean()),
            "wshare_pos": float(wshare_pos.mean()),
            "alpha2_share_pct": float(a2share * 100),
            "rms_alpha_tail_pct": rms_tail,
            "rms_alpha_kept_pct": rms_kept,
            "ew_tail_sharpe": float(np.nanmean(ew_tail) / (np.nanstd(ew_tail) + 1e-12)
                                    * np.sqrt(12)),
            "ew_kept_sharpe": float(np.nanmean(ew_kept) / (np.nanstd(ew_kept) + 1e-12)
                                    * np.sqrt(12)),
            "ew_tail_vol_pct": float(np.nanstd(ew_tail) * 100),
            "ew_kept_vol_pct": float(np.nanstd(ew_kept) * 100),
        })
    # pooled E2 Sharpe from the dump (reproducibility check vs 0.3628)
    rp_all = []
    for w in wins:
        om, R = w["omega"], w["R"]
        num = np.where(np.isfinite(R), om * R, 0.0).sum(axis=1)
        den = np.where(np.isfinite(R), np.abs(om), 0.0).sum(axis=1)
        rp = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
        rp_all.append(rp[np.isfinite(rp)])
    rp_all = np.concatenate(rp_all)
    sharpe_dump = float(np.nanmean(rp_all) / np.nanstd(rp_all) * np.sqrt(12))
    print(f"[check] E2 pooled Sharpe recomputed from dump: {sharpe_dump:.4f}"
          f"  (master_results: 0.3628)")

    # ---- stock-level daily + monthly stats ---------------------------------
    stat_rows = []          # per window x per stock, joined with membership
    covered = 0
    for w in wins:
        for i, tkr in enumerate(w["tickers"]):
            ds = daily_stats(tkr)
            ms = monthly_stats(tkr, monthly)
            if ds is None or ms is None:
                continue
            covered += 1
            stat_rows.append({
                "window": w["wi"],
                "ticker": tkr,
                "tail": bool(w["tr_turn"][i] <= w["thr"]),
                **ds, **ms,
            })
    print(f"stocks with daily+monthly stats: {covered} (window-stock pairs)")

    # ---- aggregate: mean over windows of group means ----------------------
    def gmean(rows, grp, key):
        vals = [r[key] for r in rows if r["tail"] == grp and np.isfinite(r[key])]
        return float(np.mean(vals)) if vals else np.nan

    groups = [("tail", True), ("kept", False)]
    metrics = [
        ("limit_hit_freq", "limit-hit freq (frac of days)", "%.4f"),
        ("amihud", f"Amihud (|ret|/value, x{AMIHUD_SCALE:g})", "%.4f"),
        ("trades_median", "median trades/day", "%.1f"),
        ("n_days_mean", "trading days/month", "%.2f"),
        ("thin_frac", "frac months <=3 trading days", "%.4f"),
        ("vol_monthly", "monthly ret vol", "%.4f"),
        ("ar1", "AR(1) monthly rets", "%.3f"),
        ("max_ret", "max monthly ret", "%.4f"),
    ]
    print("\n=== group comparison (mean over 12 windows) ===")
    rows_out = []
    for key, label, fmt in metrics:
        vals = {}
        for gname, gflag in groups:
            vals[gname] = gmean(stat_rows, gflag, key)
        rows_out.append({"metric": label, "tail": fmt % vals["tail"],
                         "kept": fmt % vals["kept"]})
        print(f"{label:<38} tail={fmt % vals['tail']}   kept={fmt % vals['kept']}")
    for key, label, fmt in [
            ("wshare_abs", "|omega| share on group", "%.4f"),
            ("wshare_pos", "positive-omega share on group", "%.4f"),
            ("alpha2_share_pct", "share of count-weighted alpha^2 loss", "%.2f"),
            ("rms_alpha_tail_pct", "RMS alpha of group (%)", "%.3f"),
            ("rms_alpha_kept_pct", "RMS alpha of group (%)", "%.3f"),
            ("ew_tail_sharpe", "EW portfolio Sharpe (ann.)", "%.3f"),
            ("ew_kept_sharpe", "EW portfolio Sharpe (ann.)", "%.3f"),
            ("ew_tail_vol_pct", "EW portfolio monthly vol (%)", "%.3f"),
            ("ew_kept_vol_pct", "EW portfolio monthly vol (%)", "%.3f")]:
        ws = [w[key] for w in wrows]
        v = float(np.nanmean(ws))
        if key in ("rms_alpha_kept_pct", "ew_kept_sharpe", "ew_kept_vol_pct"):
            rows_out.append({"metric": label, "tail": "", "kept": fmt % v})
            print(f"{label:<38} tail={'':>9}   kept={fmt % v}")
        else:
            rows_out.append({"metric": label, "tail": fmt % v, "kept": ""})
            print(f"{label:<38} tail={fmt % v}   kept={'':>9}")

    with open(RES / "e8_mechanism_stats.csv", "w", newline="", encoding="utf-8") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["metric", "tail", "kept"])
        for r in rows_out:
            wcsv.writerow([r["metric"], r["tail"], r["kept"]])
    with open(RES / "e8_mechanism_by_window.csv", "w", newline="", encoding="utf-8") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(wrows[0].keys()))
        wcsv.writeheader()
        for r in wrows:
            wcsv.writerow(r)
    print(f"\nsaved -> {RES/'e8_mechanism_stats.csv'} and "
          f"{RES/'e8_mechanism_by_window.csv'}")

    # ---- figure ------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # panel (a): per-window Sharpe of E2 vs E8 (seed 42) — the E8 gain is
        # concentrated in the windows where E2 collapses
        def _series(fname):
            with open(RES / fname, encoding="utf-8-sig", newline="") as f:
                return np.array([float(r["oos_return"]) for r in csv.DictReader(f)])
        e2s = _series("e2_pooled_series.csv")
        e8s = _series("e8_pooled_series.csv")
        def _wshp(x):
            return [float(np.nanmean(x[i*12:(i+1)*12]) / np.nanstd(x[i*12:(i+1)*12])
                          * np.sqrt(12)) for i in range(12)]
        s2, s8 = _wshp(e2s), _wshp(e8s)
        xs = np.arange(len(s2))
        fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
        axs[0].plot(xs, s2, color="0.25", linewidth=1.2, marker="o", markersize=3,
                    label="E2 (full cross-section)")
        axs[0].plot(xs, s8, color="0.65", linewidth=1.2, marker="s", markersize=3,
                    label="E8 (liquidity-filtered)")
        axs[0].axhline(0, color="0.8", linewidth=0.8)
        axs[0].set_ylabel("Sharpe ratio (12-month window)", fontsize=8)
        axs[0].set_xlabel("rolling window", fontsize=8)
        axs[0].set_xticks(xs[::2])
        axs[0].set_xticklabels([w["month_start"][:4] for w in wrows][::2],
                               fontsize=7)
        axs[0].tick_params(labelsize=7)
        axs[0].legend(frameon=False, fontsize=7, loc="lower left")
        tail_lh = [r["limit_hit_freq"] for r in stat_rows if r["tail"]]
        kept_lh = [r["limit_hit_freq"] for r in stat_rows if not r["tail"]]
        axs[1].hist(tail_lh, bins=20, alpha=0.65, color="0.35", label="excluded tail")
        axs[1].hist(kept_lh, bins=20, alpha=0.45, color="0.75", label="kept")
        axs[1].set_xlabel("days closed at price limit (frac)", fontsize=8)
        axs[1].set_ylabel("window-stock obs.", fontsize=8)
        axs[1].legend(frameon=False, fontsize=7)
        axs[1].tick_params(labelsize=7)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "fig_e8_mechanism.pdf")
        plt.close(fig)
        print(f"figure -> {FIG_DIR/'fig_e8_mechanism.pdf'}")
    except ImportError:
        print("matplotlib unavailable; skipping figure")


if __name__ == "__main__":
    sys.exit(main())
