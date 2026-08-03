#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Persian-labeled figures for the Persian manuscript (paper_fa/figures_fa/).

Regenerates fig_wealth.pdf, fig_stocks.pdf, fig_loadings.pdf with Persian
axis labels (Bahij Nazanin + arabic_reshaper + python-bidi), using the SAME
data inputs as scripts/q1_artifacts.py (results/*.csv, data/Char_all.npz).

Run (from the project root, with the project venv):
  /home/ubuntu/venvs/dlap-tse/bin/python scripts/q1_artifacts_fa.py

Deterministic output (matplotlib CreationDate=None) -> byte-reproducible.
"""
import csv
import os
from pathlib import Path

import numpy as np

import arabic_reshaper
from bidi.algorithm import get_display

ROOT = Path(os.environ.get("DLAP_ROOT", str(Path.home() / "research/dlap-tse")))
DATA = ROOT / "data"
RES = ROOT / "results"
FIG = ROOT / "paper_fa" / "figures_fa"
FIG.mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt

# register Bahij Nazanin Persian (patched: Vazirmatn's true Persian digit
# outlines; Bahij's own digits are Arabic-styled) with matplotlib
font_manager.fontManager.addfont(str(Path.home() / ".fonts/bahij_nazanin_persian.ttf"))
font_manager.fontManager.addfont(str(Path.home() / ".fonts/bahij_nazanin_persian_bold.ttf"))
plt.rcParams["font.family"] = "Bahij Nazanin Persian"


def fa(s):
    """reshape + bidi for RTL Persian strings in matplotlib.
    Date tokens must be LRE-embedded (\u202a...\u202c) BEFORE get_display or
    python-bidi reverses digit-hyphen-digit runs; control marks are stripped
    from the output (matplotlib would render them as tofu)."""
    out = get_display(arabic_reshaper.reshape(s))
    for c in "\u202a\u202b\u202c\u202d\u2066\u2067\u2068\u2069":
        out = out.replace(c, "")
    return out


# ─────────────────────────────────────────────────────────────────────────
# data (identical to q1_artifacts.py)
# ─────────────────────────────────────────────────────────────────────────
def load_series(path, model=None):
    out = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        rd = csv.DictReader(f)
        for r in rd:
            if model is None or r.get("model", model) == model:
                out.append(float(r["oos_return"]))
    return np.array(out)


models_e1 = {}
with open(RES / "e1_pooled_series.csv", encoding="utf-8-sig", newline="") as f:
    for r in csv.DictReader(f):
        models_e1.setdefault(r["model"], []).append(float(r["oos_return"]))
for m in models_e1:
    models_e1[m] = np.array(models_e1[m])

series = {"Market": models_e1["Market"], "FF5": models_e1["FF5"],
          "q-factor": models_e1["q-factor"], "PCA(5)": models_e1["PCA(5)"],
          "LASSO": models_e1["LASSO"]}
for m in ["E2", "E3", "E4B", "E5A"]:
    series[m] = load_series(RES / f"{m.lower()}_pooled_series.csv")
    assert len(series[m]) == 144, m

arr = np.load(DATA / "Char_all.npz", allow_pickle=True)
d = arr["data"].astype(np.float64)
dates = list(arr["date"])
ret_ok = d[:, :, 0] > -50
cnt_full = ret_ok.sum(axis=1)
common_idx = [i for i, dt in enumerate(dates) if "2008-07" <= dt <= "2026-06"]

ld = {}
with open(RES / "e6_loadings_all.csv", encoding="utf-8-sig", newline="") as f:
    for r in csv.DictReader(f):
        ld[r["char"]] = (float(r["mean_w"]), float(r["t_stat"]))

# ─────────────────────────────────────────────────────────────────────────
# fig_wealth.pdf — Persian labels
# ─────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.2, 4.2))
months_oos = [f"{2013 + (i + 6) // 12}-{(i + 6) % 12 + 1:02d}" for i in range(144)]
x = np.arange(144)
# factor benchmarks at unit gross leverage (bench_leverage_series.csv) so that
# wealth is comparable with the SDF portfolio / LASSO (weights sum to one)
lev_series = {}
with open(RES / "bench_leverage_series.csv", encoding="utf-8-sig", newline="") as f:
    for r in csv.DictReader(f):
        lev_series.setdefault(r["model"], []).append(float(r["oos_return"]))
for m in lev_series:
    lev_series[m] = np.array(lev_series[m])
styles = [
    ("E2", "k-", 2.2),
    ("FF5", "b--", 1.4),
    ("LASSO", "g-.", 1.4),
    ("q-factor", "r:", 1.4),
    ("PCA(5)", "m--", 1.4),
    ("Market", "0.55", 1.2),
]
for name, fmt, lw in styles:
    s = lev_series.get(f"{name}-norm1", series.get(name))
    w = np.cumprod(1.0 + s)
    ax.plot(x, w, fmt, lw=lw, label=name)
ax.set_yscale("log")
ax.set_ylim(1e-2, 3e1)
ax.set_ylabel(fa("ثروت تجمعی (مقیاس لگاریتمی)"))
ax.set_xlabel(fa("ماه برون‌نمونه‌ای (از \u202a۲۰۱۳-۰۷\u202c تا \u202a۲۰۲۵-۰۶\u202c)"))
tick_pos = list(range(0, 144, 24))
ax.set_xticks(tick_pos)
ax.set_xticklabels([months_oos[t] for t in tick_pos], rotation=0, fontsize=8)
ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.grid(True, which="both", alpha=0.3, linewidth=0.5)
fig.tight_layout()
fig.savefig(FIG / "fig_wealth.pdf", bbox_inches="tight", metadata={"CreationDate": None})
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────
# fig_stocks.pdf — Persian labels
# ─────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.2, 3.6))
x = np.arange(len(dates))
ax.plot(x, cnt_full, color="0.25", linewidth=1.0)
ax.axvspan(common_idx[0], common_idx[-1], color="0.85", alpha=0.6, zorder=0)
ax.set_ylabel(fa("تعداد سهام دارای بازده معتبر"))
ax.set_xlabel(fa("ماه (از \u202a۲۰۰۱-۰۳\u202c تا \u202a۲۰۲۶-۰۷\u202c)"))
tick_pos = list(range(0, len(dates), 48))
ax.set_xticks(tick_pos)
ax.set_xticklabels([dates[t][:4] for t in tick_pos], rotation=0, fontsize=9)
ax.grid(True, alpha=0.3, linewidth=0.5)
fig.tight_layout()
fig.savefig(FIG / "fig_stocks.pdf", bbox_inches="tight", metadata={"CreationDate": None})
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────
# fig_loadings.pdf — Persian labels
# ─────────────────────────────────────────────────────────────────────────
char_labels = {
    "size": "اندازه",
    "st_rev": "تداوم یک‌ماهه",
    "turnover": "گردش معاملات",
    "vol": "نوسان‌پذیری",
    "bm": "ارزش دفتری به بازار",
    "mom": "مومنتوم",
    "roe": "بازده حقوق صاحبان سهام",
    "ag": "رشد دارایی‌ها",
    "ac": "اقلام تعهدی",
    "noa": "خالص دارایی‌های عملیاتی",
    "nsi": "انتشار خالص سهام",
    "gp": "سودآوری ناخالص",
    "cei": "انتشار ترکیبی سهام",
    "ita": "سرمایه‌گذاری به دارایی",
    "ig": "رشد سرمایه‌گذاری",
    "dist": "درماندگی مالی",
    "oscore": "اُ-اسکور",
    "investment": "سرمایه‌گذاری (I/A)",
    "cbop": "سودآوری عملیاتی نقدی",
    "dy": "بازده سود نقدی",
}
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
ax.set_yticklabels([fa(char_labels.get(c, c)) for c in order], fontsize=8)
ax.axvline(0, color="k", linewidth=0.8)
ax.set_xlabel(fa("میانگین وزن عامل تنزیل (۱۴۴ ماه برون‌نمونه‌ای)"))
fig.tight_layout()
fig.savefig(FIG / "fig_loadings.pdf", bbox_inches="tight", metadata={"CreationDate": None})
plt.close(fig)

for fn in ["fig_wealth.pdf", "fig_stocks.pdf", "fig_loadings.pdf"]:
    print(f"{fn}: {os.path.getsize(FIG / fn)} bytes")

print("\nALL DONE")
