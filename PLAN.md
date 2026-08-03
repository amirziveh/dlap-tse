# Replication Plan — Deep Learning in Asset Pricing (Chen, Pelger & Zhu 2024) on the Tehran Stock Exchange

**Project dir:** `/home/ubuntu/research/dlap-tse/`
**Plan date:** 2026-07-31
**Source paper:** Chen, L., Pelger, M. & Zhu, J. (2024). "Deep Learning in Asset Pricing." *Management Science* 70(2), 714–750 (online July 2023; first posted on arXiv as 1904.00745, 2019). DOI: [10.1287/mnsc.2023.4695](https://doi.org/10.1287/mnsc.2023.4695). 420+ citations.
**Method:** paper-replication-project skill workflow (four-dimension evaluation, experiment catalog, data audit, gap analysis)

---

## 1. Objective

Replicate and extend the CPZ (2023) deep-learning stochastic discount factor (SDF) estimation on the full TSE cross-section. Primary contributions of the resulting paper:

1. **First deep-learning SDF estimate for any frontier/emerging market** (verified: none exists for TSE or MENA).
2. TSE as a stress test of the method: sanctions-driven regime breaks, retail-dominated trading, 5% daily price bands, thin order books — conditions where linear factor models are known to fail.
3. Evidence on *which firm characteristics actually price TSE stocks* (SDF loadings), benchmarked against FF5, q-factor, PCA and LASSO.

**Success criteria (from the source paper):** the estimated SDF must beat FF5/q5/PCA/LASSO out-of-sample on (a) SDF-portfolio Sharpe ratio, (b) explained return variation (R²), (c) pricing errors (max Sharpe of test assets). We do NOT need to match US magnitudes — we need the *ranking* (DL > benchmarks) to hold on TSE.

---

## 2. Source paper — what we replicate

### 2.1 The model
CPZ estimate a **conditional SDF** of the form

```
M_{t+1} = 1 − Σ_j w_j(z_t) · x_{j,t}     (per stock, with w_j learned by a deep network)
```

- `x_{j,t}` = firm characteristics of stock i at time t (the paper uses ~40 from the Chen–Zimmermann library; we use a TSE-feasible subset)
- `z_t` = **macro-economic state variables**, *extracted by the network itself* from a panel of macro series (not hand-picked) — this gives time-varying factor loadings
- No-arbitrage pricing condition `E_t[M_{t+1} R_{i,t+1}] = 1` is the **criterion function** (the network is trained to make pricing errors small, not to maximize return prediction R²)

### 2.2 Key innovations
| Innovation | What it does |
|---|---|
| **No-arbitrage as loss** | Trains the SDF to *price* assets (E[MR]=1), not to predict returns — avoids the classic ML-overfit-to-predictability trap |
| **Adversarial test assets** | A second "critic" network searches for the portfolio the SDF prices worst; training alternates SDF↔critic, producing a robust SDF |
| **Macro-state extraction** | States are learned from a macro panel (default spread, term spread, dividend yield, inflation, growth, etc.) rather than assumed |

### 2.3 Evaluation protocol (replicate as-is)
- Rolling/expanding window: train on past window, evaluate next period OOS
- Metrics: OOS Sharpe of the SDF-implied portfolio; OOS explained variation (R²); pricing errors (max Sharpe ratio of test assets)
- Benchmarks: FF5 (2015), q-factor (Hou-Xue-Zhang 2015), PCA factors, LASSO-characteristic factors

---

## 3. Data audit & mapping

### 3.1 Verified available (local files in `research/fama-five/data/`)

| Input | Local source | Status |
|---|---|---|
| Monthly returns (adjusted) | `processed/monthly_returns.csv` — 2005–2026, 384 tickers | ✅ |
| Market cap monthly | `processed/market_cap_monthly.csv` | ✅ |
| **11 anomaly signals** (nsi, cei, ac, noa, ag, ita, ig, dist, oscore, mom, gp) + roe, bm_proxy, assets, equity, revenue, cogs | `mispricing/anomaly_signals.csv` (annual, all stocks) | ✅ core characteristics |
| FF3/FF5 factors (2×3, 2×2, 2×2×2×2) | `factors/factors_*.csv` | ✅ benchmark factors |
| Risk-free rate (CBI) | `risk_free_rate.csv` — 2003–2026 | ✅ |
| Accounting panels (BS/IS/CF) | `processed/ff5_accounting.csv`, `rahavard_*.csv` | ✅ for extra characteristics |
| TEDPIX index | TSETMC API `Index/GetIndexB2History` | ✅ fetchable |
| Dividend-adjusted prices | `processed/prices_adjusted.csv` | ✅ |

### 3.2 Needed — build or collect

| Input | Source | Effort |
|---|---|---|
| **Extra characteristics** (target ~20 total): size (log mcap), BM, momentum (12-1), short-term reversal, ROE, investment, accruals, NOA, NSI, asset growth, gross profitability, cash profitability, dividend yield, turnover | construct from existing panels (all inputs exist) | 1–2 days, scriptable |
| **Macro state panel** (~8 series): CBI policy rate ✅, CPI inflation, USD/IRR official, Brent oil, gold coin (TSE), M2 money supply, industrial production, term spread | CBI (cbi.ir), FRED (oil), TSETMC (gold) | 3–5 days mostly manual |
| Universe list with liquidity filters | `stock_universe.csv` | ✅ |

### 3.3 Benchmark factors
- FF5: **already computed** (2×3 and 2×2 variants) ✅
- q-factor (HXZ 2015): needs size, investment (have), ROE (have), expected growth — constructible; or use the 2×2×2×2 factors as proxy for benchmarking purposes (note in paper)
- PCA factors: from monthly returns — trivial to compute
- LASSO: needs characteristic panel — same data as SDF inputs

---

## 4. Gap analysis

| Gap | Severity | Workaround |
|---|---|---|
| Macro state panel must be collected | ⚠️ hard, not impossible | CBI website is manual; ~5 days; oil via FRED API; gold via TSETMC API |
| Delisting data absent | ⚠️ | Use survivorship-safe rebalancing; note limitation; robustness: include delisted-if-known subset |
| 5% daily price band distorts monthly variance | ⚠️ | Use adjusted returns (already have); report band-constrained trading as institutional feature (a *contribution*, not a bug) |
| Thin trading / low-liquidity tails | ⚠️ | Liquidity filters (drop bottom 5% by turnover/month); robustness check with/without |
| EPS values column empty | ❌ (not needed) | CPZ uses characteristics only — no EPS needed; SY signals already computed |
| Small cross-section (384 tickers, ~117k stock-month obs) | ⚠️ | *Favorable* for training speed; guard overfitting with early stopping + validation |
| Short sample vs US (2005–2026 vs 1958–2016) | ⚠️ | 21 years is enough for monthly evaluation; use rolling windows of 60 months |

---

## 5. Architecture & implementation design

### 5.1 Components (PyTorch, Colab GPU)
1. **SDF network** `M_net`: input = characteristics x_t (standardized) + macro states z_t (learned); output = SDF values per stock. Time-varying coefficients via conditioning on z_t.
2. **State extraction net** `Z_net`: input = macro panel (lagged) → low-dim state vector z_t (dim ≈ 4–8). Train jointly.
3. **Adversarial critic net** `A_net`: input = characteristics → portfolio weights ω(z_t) over stocks that maximize the SDF's pricing error. Alternating optimization.
4. **Loss:** squared pricing errors over all test assets:
   `L = mean( E[ M R^e ]² )` + regularization (weight decay, early stopping on validation pricing errors).

### 5.2 Training protocol
- Windows: train 60 months → validate 12 → test next 12 (rolling); OR expanding window (robustness)
- Standardization: cross-sectional z-score per month (characteristics)
- Optimizer: Adam; LR 1e-3 with schedule; batch = all stocks in month (small enough)
- Early stopping on validation pricing error
- Reference implementation: authors released public code (verify link in Phase 0 — Pelger's Stanford page / GitHub mirrors)

---

## 6. Experiment catalog

| # | Experiment | Output metric |
|---|---|---|
| E1 | Benchmark battery on TSE: FF5, q5, PCA, LASSO | OOS Sharpe, R², pricing errors |
| E2 | CPZ SDF, 11-characteristic set (existing SY signals) | same |
| E3 | CPZ SDF, 20-characteristic set (extended) | same |
| E4 | CPZ SDF with vs without macro states (z_t off) | does macro conditioning matter on TSE? |
| E5 | Adversarial critic on vs off | robustness of SDF |
| E6 | SDF loadings → rank which characteristics price TSE stocks | loading table + economic interpretation |
| E7 | Subperiod: 2020 boom-bust (1399-1400 SH) | regime stability |
| E8 | Liquidity filter: exclude bottom 5% turnover | robustness |

---

## 7. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Implementation complexity (adversarial loop) | Stepwise build: E2 without critic first → add critic; reuse authors' code as scaffold |
| Overfitting in small universe | Early stopping, validation pricing error, fewer characteristics in E2 vs E3, dropout |
| Macro data collection delays | Start collection in Phase 0 immediately; fall back to z_t = single state (CBI rate) for v1 |
| Look-ahead bias in accounting characteristics | Use announcement-date-matched values (CODAL dates available); SY signals already formation-year aligned |
| Reviewer "why TSE?" | Built-in: frontier market + sanctions + price bands = extreme institutional stress test |

---

## 8. Timeline (part-time, ~10 weeks)

| Phase | Weeks | Deliverable |
|---|---|---|
| **0. Feasibility spike** | 1 | Verify authors' code reachable; confirm characteristic set buildable; sample macro series |
| **1. Data assembly** | 2 | 20-characteristic panel; macro state panel; benchmark factors (q5 construction) |
| **2. Benchmarks** | 2 | E1 complete — benchmarks on TSE |
| **3. DLAP implementation** | 3 | E2→E3→E4→E5 (incremental, validated) |
| **4. Evaluation & robustness** | 1.5 | E6–E8, full results tables |
| **5. Write-up** | 1 | Paper draft (English) + Persian abstract |

---

## 9. Deliverables
- `dlap-tse/` project: scripts, data build notebooks, results tables
- Paper: "Deep learning asset pricing on the Tehran Stock Exchange" — target journals: *Emerging Markets Review*, *Journal of Empirical Finance*, *International Review of Financial Analysis* (IRFA), *Journal of Behavioral and Experimental Finance*
- Replication-complete markers: benchmarks < DL-SDF on all three OOS metrics

---

## 10. Phase-0 open questions (answer before committing)
1. Is the authors' public code still accessible and runnable (versions)?
2. Can CBI data (CPI, M2, FX) be collected programmatically or is it fully manual?
3. What is the cleanest 20-characteristic set given TSE accounting quirks (e.g., Persian fiscal year 21-Mar)? — reuse SY formation-year logic
4. Should q5 be constructed or proxied by the 2×2×2×2 factor set?

---

*This plan follows the paper-replication-project skill workflow: evaluation → data audit → gap analysis → experiment catalog → execution plan.*
