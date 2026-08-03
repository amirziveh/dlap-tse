# Phase 0 — Feasibility Spike: Findings

**Date:** 2026-08-03 · **Project:** dlap-tse (CPZ 2024 replication on TSE)
**Scope:** answer PLAN.md §10 open questions before committing to the build.

---

## Q1. Is the authors' public code accessible and runnable? → ✅ ACCESSIBLE (as reference)

**Official repository (TensorFlow):** `github.com/LouisChen1992/Deep_Learning_Asset_Pricing`
(Luyang "Louis" Chen's account; 41★, Feb 2021). Cloned to `code/official_cpz/`.

Pipeline (5 steps, per README):
1. `run.py --config=config/config.json` — trains the SDF network (GAN-style SDF + critic)
2. `model_GAN.ipynb` (first 8 cells) — generates SDF values
3. `create_RF_data.py` — builds R×F (return × factor) data
4. `run_RtnFcst_ensembles.py` — trains the residual-return prediction network (ensembles)
5. `model_GAN.ipynb` (remaining cells) — computes EV / XS-R² / weighted XS-R²

**Key hyperparameters extracted from official `config.json`:**

| Setting | Value | Our TSE adaptation |
|---|---|---|
| optimizer / lr | Adam / 0.001 | same |
| hidden layers | `[64, 64]`, 2 layers | same (small universe → keep small) |
| dropout (keep prob) | 0.95 | keep, guard overfitting |
| weighted loss | true | same |
| state net | LSTM, `num_units_rnn=[4]` → **4 latent states** | same (dim 4) |
| conditioning | `num_condition_moment=8`, macro_feature_dim=178 | we use ~8 raw macro series → same spirit |
| characteristics | `individual_feature_dim=46` | we use ~20 |
| windows | tSize 240 / valid 60 / test 300 | 60/12/12 rolling (TSE sample) |
| epochs | 1024, sub_epoch 4 | early stopping on validation pricing error |

**PyTorch port:** `github.com/Darenar/DeepLearningAssetPricing_torch` (2021, torch 1.9).
Cloned to `code/torch_port/` — same config, clean module split (`engine.py`, `data_loader.py`,
`portfolio_utils.py`, `models/`), plus `calculate_statistics.py` (EV, XS-R², weighted XS-R²).
This is the closest runnable template for our stack.

**Reference US dataset:** Google Drive folder `1TrYzMUA_xLID5-gXOy_as8sH2ahLwz-l`
(char/macro npz train/valid/test). Downloadable via `gdown` — useful only to mirror the exact
input format; we build our own TSE npz with the same layout.

**Runnability verdict:** TF 1.12 / Py 3.6 is not installable on this box; torch port needs
torch 1.9 (old but portable). **Decision: write our own PyTorch implementation (as planned),
using both repos as architecture/hyperparameter reference — NOT running their code on TSE data.**

---

## Q2. Can CBI data be collected programmatically? → ⚠️ CBI DIRECT IS DEAD — BUT BYPASSABLE (6/8 series free)

Tests (2026-08-03):
- `tsd.cbi.ir` (CBI time-series DB): **connection failure (000) from both this machine AND the Iran server (`ime`)** — effectively inaccessible
- `cbi.ir` main site: reachable (302 → `https://cbi.ir/`) — HTML table scraping possible but low value
- **No FRED API key in `.env` — not needed:** `fredgraph.csv` endpoint works keyless (verified, Brent DCOILBRENTEU 1987→present)
- **World Bank API works keyless** (watch the UTF-8 BOM — decode `utf-8-sig`)

Macro series coverage map (target ≈ 8 series):

| Series | Source | Status |
|---|---|---|
| CBI policy rate | **already local** (`fama-five/data/risk_free_rate.csv`, 2003–2026) | ✅ zero work |
| CPI (2010=100) | World Bank `FP.CPI.TOTL` | ✅ 1960–2025 |
| USD/IRR official | World Bank `PA.NUS.FCRF` | ✅ 1960–2023 (fill 2024+ from tgju market rate) |
| Brent oil | FRED `DCOILBRENTEU` csv | ✅ 1987–present, keyless |
| Gold coin (Emami) | tgju.org (`sekee`; scraping documented in iranian-data-pipelines skill) or TSETMC coin certs | ✅ reachable |
| USD/IRR market | tgju.org `price_dollar_rl` (verified live: 1,910,000 IRR) | ✅ |
| M2 | World Bank `FM.LBL.BMNY.GD.ZS` **ends 2016** | ⚠️ IMF IFS (free key) or CBI manual, or drop |
| Industrial production | no monthly public source | ⚠️ drop / replace (TEDPIX, exports) |
| Term spread | no bond market in Iran | ⚠️ replace (policy vs deposit spread) or drop |

**Verdict:** the plan's "3–5 days mostly manual" shrinks to ~0 manual days for 6/8 series.
M2 and IP need substitution — acceptable, since CPZ's own state extraction only needs a
*diverse* panel, and the fallback (single state = CBI rate) is already available.

---

## Q3. Cleanest 20-characteristic set → ✅ FULLY BUILDABLE, ALL INPUTS LOCAL

Inventory confirmed in `fama-five/data/`:

- `anomaly_signals.csv` (9,780 rows, ticker × year, **formation_year aligned**):
  nsi, cei, ac, noa, ag, ita, ig, dist, oscore, mom, gp, roe, bm_proxy, total_assets,
  net_income, total_equity, revenue, cogs
- `processed/cbop_panel.csv` — **cash-based operating profitability (cbop) + investment + accruals variants ALREADY COMPUTED**
- `processed/dps_panel.csv` — dividend yield (dps, payout ratio)
- `processed/ff5_accounting.csv` — op_at, investment, bm_proxy (FF5-style)
- `processed/market_cap_monthly.csv` — size = log(mcap)
- `processed/monthly_returns.csv` — short-term reversal (1-month lagged return)
- `processed/prices_tsetmc_adjusted.csv` — **volume column exists** → turnover = volume×price/mcap (or volume/shares_at_date)

**Proposed 20:** size, bm, mom, st_rev, roe, investment, asset_growth, accruals_bs, noa,
nsi, gross_profitability, cash_profitability(cbop), dividend_yield, turnover, cei, ita, ig,
dist, oscore, gp → exactly 20, zero new data collection. All cross-sectionally z-scored per
month; annual chars joined on formation_year to the following 12 months (existing SY logic).

---

## Q4. q-factor: construct or proxy? → ✅ CONSTRUCT (cheap)

q-factor needs MKT, size, I/A, ROE — **all inputs already local** (investment from cbop_panel,
roe from anomaly signals, mcap, returns). ~1 day of scripting. Keep the existing 2×2×2×2
factors as a robustness cross-check. PCA (from returns) and LASSO (from the same char panel)
are trivial as planned.

---

## Phase 0 verdict → GO

| Question | Verdict |
|---|---|
| Q1 authors' code | ✅ found, cloned, hyperparameters extracted (reference only) |
| Q2 macro data | ✅ 6/8 series programmatic & free; M2/IP substituted; CBI bypassed |
| Q3 20-char set | ✅ buildable from local files, zero collection |
| Q4 q-factor | ✅ construct from local data |

**Next: Phase 1 — Data assembly** (per PLAN.md §8):
1. `scripts/build_characteristics.py` — 20-char panel (formation-year aligned, monthly z-scores)
2. `scripts/build_macro_panel.py` — World Bank + FRED + tgju collectors → monthly macro panel 2005–2026
3. `scripts/build_qfactors.py` — q-factor 4-factor construction
4. Dataset format mirror of official npz layout (train/valid/test) for the future torch code

## Open items / risks
- Google Drive reference dataset download not verified (needs gdown; optional)
- tgju **history depth** unverified (current price works; multi-year history endpoint needs
  one scrape during Phase 1 — fallback: TSETMC coin certificates)
- 2024–2025 USD/IRR official gap → fill from tgju market rate (document in paper)
