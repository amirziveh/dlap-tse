#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q1 expansion artifact script — dlap-tse (round-04 revision).

Regenerates, from the raw data under data/ and results/:
  results/desc_stats.csv            (panel/macro/characteristic summary statistics)
  results/sharp_diff_bootstrap.csv  (moving-block bootstrap of Sharpe differences,
                                     block length 6, B = 10,000, seed 42)
  results/per_window_sharpe.csv     (annualized Sharpe by 12-month test window)
  paper/figures/fig_wealth.pdf      (cumulative wealth, log scale)
  paper/figures/fig_stocks.pdf      (stocks per month; shaded common period
                                     2008-07..2026-06; year-only x labels)
  paper/figures/fig_loadings.pdf    (mean SDF weights; st_rev relabeled
                                     "Short-term continuation" per round-04 M1)

How to run (from the project root, with the project venv):
  python scripts/q1_artifacts.py
  # or: DLAP_ROOT=/path/to/dlap-tse python scripts/q1_artifacts.py

No training is involved: all inputs are the existing CSVs/npz under data/ and
results/ produced by scripts/build_characteristics.py, scripts/run_e1.py,
scripts/train_e2.py, and the fama-five pipeline.
"""
import csv
import math
import os
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("DLAP_ROOT", str(Path.home() / "research/dlap-tse")))
DATA = ROOT / "data"
RES = ROOT / "results"
FIG = ROOT / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────
# WP1 — descriptive statistics
# ─────────────────────────────────────────────────────────────────────────
print("=" * 78)
print("WP1: descriptive statistics")
print("=" * 78)

# raw panel
rows = []
with open(DATA / "characteristics_panel.csv", encoding="utf-8-sig", newline="") as f:
    rd = csv.DictReader(f)
    fieldnames = rd.fieldnames
    for r in rd:
        rows.append(r)
n_sm = len(rows)
chars = fieldnames[4:]  # 20 characteristics after ticker,year,month,ret_monthly
months_full = sorted({(int(r["year"]), int(r["month"])) for r in rows})
n_months_full = len(months_full)
tickers = sorted({r["ticker"] for r in rows})
n_stocks = len(tickers)
print(f"panel: {n_sm} stock-months, {n_months_full} months, {n_stocks} stocks")
assert n_sm == 74147, n_sm
assert n_months_full == 303, n_months_full
assert n_stocks == 357, n_stocks

# npz return coverage (stocks with valid return per month)
arr = np.load(DATA / "Char_all.npz", allow_pickle=True)
d = arr["data"].astype(np.float64)          # (303, 357, 21)
dates = list(arr["date"])
ret_ok = d[:, :, 0] > -50
cnt_full = ret_ok.sum(axis=1)               # per month, full panel
common_idx = [i for i, dt in enumerate(dates) if "2008-07" <= dt <= "2026-06"]
cnt_common = cnt_full[common_idx]
print(f"npz shape {d.shape}; common months {len(common_idx)} ({dates[common_idx[0]]}..{dates[common_idx[-1]]})")

def mn(s):  # min/mean/max
    return (int(s.min()), round(float(s.mean()), 1), int(s.max()))

full_mm = mn(cnt_full)
common_mm = mn(cnt_common)
print(f"stocks/month full : min/mean/max = {full_mm}")
print(f"stocks/month common: min/mean/max = {common_mm}")
# round-04 m6: verify peak month and end-of-sample counts
peak_i = int(cnt_full.argmax())
print(f"peak count {int(cnt_full[peak_i])} at {dates[peak_i]}; "
      f"last two months: {dates[-2]}={int(cnt_full[-2])}, {dates[-1]}={int(cnt_full[-1])}")

# missing fraction of stock-months (npz sentinel)
frac_missing_ret = float((d[:, :, 0] < -50).mean())
print(f"fraction of (month,stock) cells with missing return (npz): {frac_missing_ret:.4f}")

# macro stats over common period from macro_panel.csv
macro_names = ["cbirate", "cpi", "usd_official", "brent", "gold_coin", "usd_market"]
mrows = {}
with open(DATA / "macro_panel.csv", encoding="utf-8-sig", newline="") as f:
    for r in csv.DictReader(f):
        mrows[r["month"]] = r
macro_stats = {}
for c in macro_names:
    vals = []
    for i in common_idx:
        v = mrows.get(dates[i], {}).get(c, "")
        try:
            vals.append(float(v))
        except (TypeError, ValueError):
            pass
    macro_stats[c] = (float(np.mean(vals)), float(np.std(vals)), len(vals))
    print(f"macro {c:>12}: mean={macro_stats[c][0]:.4f} std={macro_stats[c][1]:.4f} n={macro_stats[c][2]}")

# character stats on the RAW (pre-z-score) panel: mean, std, % missing
char_stats = {}
for c in chars:
    vals = []
    miss = 0
    for r in rows:
        v = r.get(c, "")
        try:
            vals.append(float(v))
        except (TypeError, ValueError):
            miss += 1
    v = np.array(vals)
    char_stats[c] = (float(np.nanmean(v)), float(np.nanstd(v)), 100.0 * miss / n_sm)
    print(f"char {c:>12}: mean={char_stats[c][0]:.4f} std={char_stats[c][1]:.4f} %miss={char_stats[c][2]:.2f}")

# ── write results/desc_stats.csv ─────────────────────────────────────────
with open(RES / "desc_stats.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["category", "item", "stat", "value"])
    w.writerow(["panel", "months_full", "n", n_months_full])
    w.writerow(["panel", "stocks", "n", n_stocks])
    w.writerow(["panel", "stock_months", "n", n_sm])
    w.writerow(["panel", "stocks_per_month_full", "min", full_mm[0]])
    w.writerow(["panel", "stocks_per_month_full", "mean", full_mm[1]])
    w.writerow(["panel", "stocks_per_month_full", "max", full_mm[2]])
    w.writerow(["panel", "stocks_per_month_common", "min", common_mm[0]])
    w.writerow(["panel", "stocks_per_month_common", "mean", common_mm[1]])
    w.writerow(["panel", "stocks_per_month_common", "max", common_mm[2]])
    w.writerow(["panel", "frac_missing_return_cells", "n", f"{frac_missing_ret:.4f}"])
    for c in macro_names:
        w.writerow(["macro", c, "mean", f"{macro_stats[c][0]:.6f}"])
        w.writerow(["macro", c, "std", f"{macro_stats[c][1]:.6f}"])
        w.writerow(["macro", c, "n_obs", macro_stats[c][2]])
    for c in chars:
        w.writerow(["char", c, "mean", f"{char_stats[c][0]:.6f}"])
        w.writerow(["char", c, "std", f"{char_stats[c][1]:.6f}"])
        w.writerow(["char", c, "pct_missing", f"{char_stats[c][2]:.4f}"])
print("saved results/desc_stats.csv")

# ─────────────────────────────────────────────────────────────────────────
# WP5 — per-window Sharpe (12 windows x 9 models)
# ─────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 78)
print("WP5: per-window Sharpe")
print("=" * 78)

def load_series(path, model=None):
    out = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        rd = csv.DictReader(f)
        for r in rd:
            if model is None or r.get("model", model) == model:
                out.append(float(r["oos_return"]))
    return np.array(out)

def sharpe_ann(r):
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    return float(r.mean() / r.std() * math.sqrt(12)) if len(r) >= 3 and r.std() > 0 else float("nan")

models_e1 = {}
with open(RES / "e1_pooled_series.csv", encoding="utf-8-sig", newline="") as f:
    for r in csv.DictReader(f):
        models_e1.setdefault(r["model"], []).append(float(r["oos_return"]))
for m in models_e1:
    models_e1[m] = np.array(models_e1[m])
    assert len(models_e1[m]) == 144, (m, len(models_e1[m]))

series = {"Market": models_e1["Market"], "FF5": models_e1["FF5"],
          "q-factor": models_e1["q-factor"], "PCA(5)": models_e1["PCA(5)"],
          "LASSO": models_e1["LASSO"]}
for m in ["E2", "E3", "E4B", "E5A"]:
    series[m] = load_series(RES / f"{m.lower()}_pooled_series.csv")
    assert len(series[m]) == 144, (m, len(series[m]))

win_labels = [f"{13 + i:02d}-{14 + i:02d}" for i in range(12)]  # 13-14 ... 24-25

per_win = {}
for m, s in series.items():
    sharps = [sharpe_ann(s[i * 12:(i + 1) * 12]) for i in range(12)]
    per_win[m] = sharps
    print(f"{m:>9}: " + " ".join(f"{x:6.2f}" for x in sharps) + f"  full={sharpe_ann(s):6.2f}")

# E2 vs FF5 / LASSO window counts
e2w = np.array(per_win["E2"])
ff5w = np.array(per_win["FF5"])
lassow = np.array(per_win["LASSO"])
print(f"E2 > FF5   in {int((e2w > ff5w).sum())} of 12 windows")
print(f"E2 > LASSO in {int((e2w > lassow).sum())} of 12 windows")

with open(RES / "per_window_sharpe.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["model"] + win_labels + ["full"])
    for m in ["Market", "FF5", "q-factor", "PCA(5)", "LASSO", "E2", "E3", "E4B", "E5A"]:
        w.writerow([m] + [f"{x:.4f}" for x in per_win[m]] + [f"{sharpe_ann(series[m]):.4f}"])
print("saved results/per_window_sharpe.csv")

# ─────────────────────────────────────────────────────────────────────────
# WP4 — moving-block bootstrap (block length 6, B=10000, seed 42)
# ─────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 78)
print("WP4: moving-block bootstrap")
print("=" * 78)
L, B, N = 6, 10000, 144
n_blocks = N // L  # 24
rng = np.random.RandomState(42)  # deterministic seed 42
starts = rng.randint(0, N - L + 1, size=(B, n_blocks))  # B x 24 start positions

def resample(s):
    idx = (starts[:, :, None] + np.arange(L)[None, None, :]).reshape(B, N)
    return s[idx]

e2 = series["E2"]
boot_out = []
for bench_name in ["FF5", "LASSO", "q-factor", "PCA(5)", "Market"]:
    b = series[bench_name]
    se2 = sharpe_ann(e2)
    sb = sharpe_ann(b)
    diff_pt = se2 - sb
    e2_b = resample(e2)
    b_b = resample(b)
    sh_e2 = e2_b.mean(axis=1) / e2_b.std(axis=1) * math.sqrt(12)
    sh_b = b_b.mean(axis=1) / b_b.std(axis=1) * math.sqrt(12)
    diffs = sh_e2 - sh_b
    ci_lo, ci_hi = np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)
    zero_excluded = ci_lo > 0.0
    print(f"E2 vs {bench_name:>7}: point diff={diff_pt:+.3f}  CI=[{ci_lo:+.3f}, {ci_hi:+.3f}]  zero_excluded={zero_excluded}")
    boot_out.append((f"E2 vs {bench_name}", f"{se2:.4f}", f"{sb:.4f}", f"{diff_pt:.4f}",
                     f"{ci_lo:.4f}", f"{ci_hi:.4f}", int(zero_excluded)))

with open(RES / "sharp_diff_bootstrap.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["pair", "sharpe_e2", "sharpe_bench", "diff", "ci_lo", "ci_hi", "zero_excluded"])
    for r_ in boot_out:
        w.writerow(r_)
print("saved results/sharp_diff_bootstrap.csv")

# ─────────────────────────────────────────────────────────────────────────
# WP3 — figures
# ─────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 78)
print("WP3: figures")
print("=" * 78)

# ---- fig_wealth.pdf ----
fig, ax = plt.subplots(figsize=(7.2, 4.2))
months_oos = [f"{2013 + (i + 6) // 12}-{(i + 6) % 12 + 1:02d}" for i in range(144)]
x = np.arange(144)
styles = [
    ("E2", "k-", 2.2),
    ("FF5", "b--", 1.4),
    ("LASSO", "g-.", 1.4),
    ("q-factor", "r:", 1.4),
    ("PCA(5)", "m--", 1.4),
    ("Market", "0.55", 1.2),
]
for name, fmt, lw in styles:
    w = np.cumprod(1.0 + series[name])
    ax.plot(x, w, fmt, lw=lw, label=name)
ax.set_yscale("log")
ax.set_ylabel("Cumulative wealth (log scale)")
ax.set_xlabel("Out-of-sample month (2013-07 to 2025-06)")
tick_pos = list(range(0, 144, 24))
ax.set_xticks(tick_pos)
ax.set_xticklabels([months_oos[t] for t in tick_pos], rotation=0, fontsize=8)
ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.grid(True, which="both", alpha=0.3, linewidth=0.5)
fig.tight_layout()
fig.savefig(FIG / "fig_wealth.pdf", bbox_inches="tight")
plt.close(fig)

# ---- fig_stocks.pdf (round-04 m6: legible year-only x labels) ----
fig, ax = plt.subplots(figsize=(7.2, 3.6))
x = np.arange(len(dates))
ax.plot(x, cnt_full, color="0.25", linewidth=1.0)
# shaded common period: exactly 2008-07..2026-06 (indices of the first/last
# common month in the dates array)
ax.axvspan(common_idx[0], common_idx[-1], color="0.85", alpha=0.6, zorder=0)
ax.set_ylabel("Stocks with valid return")
ax.set_xlabel("Month (2001-03 to 2026-07)")
tick_pos = list(range(0, len(dates), 48))  # every 4 years: 2001, 2005, ..., 2025
ax.set_xticks(tick_pos)
ax.set_xticklabels([dates[t][:4] for t in tick_pos], rotation=0, fontsize=9)
ax.grid(True, alpha=0.3, linewidth=0.5)
fig.tight_layout()
fig.savefig(FIG / "fig_stocks.pdf", bbox_inches="tight")
plt.close(fig)

# ---- fig_loadings.pdf (round-04 M1: st_rev relabeled) ----
ld = {}
with open(RES / "e6_loadings_all.csv", encoding="utf-8-sig", newline="") as f:
    for r in csv.DictReader(f):
        ld[r["char"]] = (float(r["mean_w"]), float(r["t_stat"]))
# round-04 M1: short-term reversal is one-month continuation on TSE
char_labels = {"st_rev": "Short-term continuation"}
order = sorted(ld.keys(), key=lambda c: ld[c][0])
fig, ax = plt.subplots(figsize=(7.2, 5.6))
pos = np.arange(len(order))
colors = []
for c in order:
    mw, t = ld[c]
    if mw >= 0:
        base = "#4C72B0" if abs(t) > 2 else "#A6BFE3"
    else:
        base = "#C44E52" if abs(t) > 2 else "#E8B4B6"
    colors.append(base)
ax.barh(pos, [ld[c][0] for c in order], color=colors, edgecolor="none", height=0.7)
ax.set_yticks(pos)
ax.set_yticklabels([char_labels.get(c, c) for c in order], fontsize=8)
ax.axvline(0, color="k", linewidth=0.8)
ax.set_xlabel("Mean SDF weight $\\bar{w}_j$ (144 out-of-sample months)")
fig.tight_layout()
fig.savefig(FIG / "fig_loadings.pdf", bbox_inches="tight")
plt.close(fig)

for fn in ["fig_wealth.pdf", "fig_stocks.pdf", "fig_loadings.pdf"]:
    print(f"{fn}: {os.path.getsize(FIG / fn)} bytes")

print("\nALL DONE")
