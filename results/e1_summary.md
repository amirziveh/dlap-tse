# E1 — Benchmark Battery Results (2026-08-03)

**Protocol (CPZ 2024-style):** rolling windows train 60m / test 12m, step 12;
common period 2008-07..2026-06 (214 months, 12 OOS windows, 144 OOS months).
Factor SDFs: max-Sharpe weights on factor returns (shrunk cov, δ=0.2).
LASSO: linear char SDF, λ by 3-fold CV. Test assets: 357 individual stocks.
Returns winsorized 1%/99% per month; z-chars winsorized + clipped ±10.

## Results (pooled OOS)

| Model | OOS Sharpe | Sharpe (mean win) | XS-R² | RMS α (%) | max |α| (%) |
|---|---|---|---|---|---|---|
| Market (CAPM) | 0.183 | 0.142 | −2.14 | 5.99 | 32.4 |
| FF5 | **0.840** | 0.846 | −3.38 | 7.56 | 56.1 |
| q-factor | 0.728 | 0.542 | −3.05 | 5.87 | 40.3 |
| PCA(5) | 0.607 | 0.509 | −3.09 | 5.21 | 31.5 |
| LASSO | 0.825 | 0.575 | −2.11 | 7.21 | 56.8 |

- LASSO: λ ≈ 3e-4, 18/20 characteristics nonzero (mom shrunk to zero in CV)
- HJ bound (avg max-Sharpe of test assets, train cov): **2.914 monthly** (~10.1 annualized)

## Reading

1. **Ranking (Sharpe):** FF5 ≈ LASSO > q > PCA > Market. The linear benchmarks are
   weak but FF5 leads on TSE — it is the benchmark to beat for the DL-SDF.
2. **XS-R² negative for all benchmarks (−2.1 to −3.4):** no linear factor model
   explains the TSE cross-section out of sample. Consistent with the local
   literature (Davallou & Badri 2015: CAPM/FF3/Carhart all rejected on TSE).
   This is the gap the deep SDF must fill — and the DL-SDF's success criterion
   is *beating this ranking*, not US-scale magnitudes.
3. **Pricing errors 5–7.6% RMS monthly** (individual-stock test assets; large by
   US standards but TSE monthly vol is ~10-15% per stock).
4. **HJ bound 2.91 monthly** — the test assets contain a lot of exploitable
   cross-sectional structure (theoretical max Sharpe ≫ what any linear model
   achieves). Headroom for a nonlinear SDF is large.

## Files
- `results/e1_benchmarks.csv` — one row per model (canonical table)
- `results/e1_pooled_series.csv` — pooled OOS SDF portfolio returns per model
- `scripts/run_e1.py` — runner · `scripts/eval_core.py` — shared evaluation module

## Notes for the manuscript
- Breakpoints/construction: all-stock (no NYSE analog); value-weighted 2×3 q-sorts
- Winsorization of returns is documented data prep (TSE adjustment artifacts:
  185 returns > +100%/month pre-winsorization, 2.41% clipped)
- Shrinkage δ=0.2 for max-Sharpe weights; ridge on HJ-bound cov
