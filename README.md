# DLAP-TSE — Deep Learning in Asset Pricing on the Tehran Stock Exchange

Replication of **Chen, Pelger & Zhu (2024)**, *"Deep Learning in Asset Pricing"*
(*Management Science* 70(2):714–750) on the Tehran Stock Exchange (TSE): a deep-learning
stochastic discount factor (SDF) `M = 1 − w(z)′x` whose characteristic weights are learned
by a neural network, evaluated out of sample against FF5, q-factor, PCA, LASSO, and the
market over 2013-07–2025-06 (12 rolling windows of 12 test months).

**Headline result:** the deep SDF (11 Stambaugh–Yuan signals, LSTM macro states) posts a
pooled out-of-sample Sharpe ratio of 0.853, ranked above every linear benchmark (FF5 0.840,
LASSO 0.825, q 0.728, PCA 0.607, Market 0.183), and explains 47% of pooled return variation
while the strict cross-sectional R² is negative for all linear models. The Sharpe gaps
versus FF5/LASSO/q are within bootstrap sampling noise; the edge concentrates in the
2019–2022 boom–bust episode. Loadings: turnover dominates (t=22.9), the value premium is
inverted, and the TSE shows **one-month continuation** (not the US reversal anomaly).

## Repository layout

```
results/      all result tables (master_results.csv = canonical 13-model table)
scripts/      data build, training, evaluation, and artifact-generation scripts
```

The manuscript itself is not yet in this repository: it is still being finalized
(author block pending) and will be added here upon submission/finalization.

## Requirements

- Python ≥ 3.10 with `numpy`, `scipy`, `torch` (CPU suffices), `matplotlib`
  (for `q1_artifacts.py` figures)
- The raw input data is built by the sibling project `fama-five`
  (`/home/ubuntu/research/fama-five` in the original environment) — see **Data** below.

## Data

The derived panels (`data/`) are **not** included in this repository (they are built from
TSE data sourced via TSETMC and Rahavard Novin under the original project's data terms).
To rebuild them:

1. Build the `fama-five` data pipeline (monthly returns, market caps, anomaly signals,
   FF5/q factors, risk-free rate — see the fama-five project).
2. Point the scripts at both roots and run, in order:

```bash
export DLAP_ROOT=/path/to/dlap-tse          # default: ~/research/dlap-tse
export FAMA_ROOT=/path/to/fama-five/data    # default: ~/research/fama-five/data

python scripts/build_characteristics.py     # 20-char monthly panel (formation-year aligned; bm = BE/ME)
python scripts/build_macro_panel.py         # 6-series macro panel
python scripts/build_winsorized_factors.py  # FF5 + HXZ q-factors built from the SAME winsorized
                                            # returns as the deep-SDF panel (benchmark symmetry)
python scripts/build_npz.py                 # Char_all.npz / Macro_all.npz (official CPZ layout)
```

Data conventions (critical — see the manuscript §3): formation-year alignment (month m≥7 of
year y → formation year y, else y−1); per-month z-scoring winsorized 1%/99% and clipped ±10;
returns winsorized 1%/99% per month (capital-increase artifacts up to +1,692%); missing
values stored as float32 −99.99 — always mask with `arr < -50`, never equality.
Alignment (critical): characteristics and macro states enter the pricing kernel with a
one-month lag relative to the returns they price (x_{t-1} → r_t, the CPZ/GKX convention);
`eval_core.lag_align` implements it and `train_e2.load_data` / `run_e1` (LASSO) apply it.
Book-to-market is BE/ME (book equity of the formation-year financial statements, in million
Rials ×1e6, over month-end market equity) — not a TE/TA leverage proxy.
Factor benchmarks are evaluated with max-Sharpe weights; for wealth-type objects the weights
are normalized to unit gross leverage (`scripts/bench_leverage_check.py` → results/bench_leverage.csv).

## Reproducing the paper's numbers

```bash
python scripts/run_e1.py          # E1: benchmark battery (~4 min)
python scripts/train_e2.py --charset sy            # E2 (11 signals, LSTM states)
python scripts/train_e2.py --charset all           # E3 (20 chars)
python scripts/train_e2.py --charset all --states const   # E4A
python scripts/train_e2.py --charset sy  --states const   # E4B
python scripts/train_e2.py --charset sy  --critic        # E5A
python scripts/train_e2.py --charset all --critic        # E5B
python scripts/train_e2.py --charset sy  --liq-filter    # E8
python scripts/train_e2.py --charset all --liq-filter    # E8B
python scripts/e6_loadings.py --charset all      # loadings (Table 5)
python scripts/e7_subperiod.py                   # subperiod Sharpes (Table 6)
python scripts/q1_artifacts.py                   # desc stats, bootstrap CIs, per-window
                                                 # table, and the three figures
```

All runs are deterministic (torch seed 42, LASSO CV seed 42, bootstrap seed 42) and
reproduce `results/*.csv` and the figures exactly.

**Verified 2026-08-03:** a fresh clone of this repository, fed the documented private
inputs, regenerates every results artifact byte-for-byte (all 30 CSVs and 3 figures) and
rebuilds the data panels identically. The one exception is `data/macro_panel.csv`, which
is fetched from live sources (World Bank, FRED, tgju.org): the sources revise the most
recent months' values over time (e.g., tgju revised the 2026-08 gold/USD values). The
paper's numbers correspond to the 2026-08-02 snapshot; no month inside the evaluation
period (through 2026-06) was affected in the verification run.

## Data & methodological notes

- The SDF loss is the mean squared pricing error `E[M R^e]²`, not return prediction.
- Deep SDFs train on 48 months with the preceding 12 held out for early stopping
  (60-month lookback); linear benchmarks estimate on the full 60 months.
- The FF5/q SDF portfolios use maximum-Sharpe weights with covariance shrinkage (δ=0.2);
  the FF5 portfolio is leveraged and has two sub-−100% months (2020-06/07) — the wealth
  figure and §5.2 disclose this.
- The official CPZ implementation (TensorFlow) and a PyTorch port used as architecture
  references live at `LouisChen1992/Deep_Learning_Asset_Pricing` and
  `Darenar/DeepLearningAssetPricing_torch`; they are not redistributed here.

## Citation

If you use this code or the results, please cite the paper (add the manuscript's citation
details and a DOI when available) and CPZ (2024), *Management Science* 70(2):714–750,
DOI 10.1287/mnsc.2023.4695.

## License

Code: MIT (see LICENSE). The manuscript text and figures are © the authors; the
`results/` tables are released under CC-BY-4.0. Data files are not redistributed (see Data).
