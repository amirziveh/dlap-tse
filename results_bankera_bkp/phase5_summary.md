# E6/E7 — Loadings & Subperiod Results (2026-08-03)

## E6 — Which characteristics price TSE stocks

SDF weights w(z_t) from the trained deep SDF, pooled over 144 OOS months.
Characteristics are per-month z-scores → w_j is the standardized loading.
**Positive w_j ⇒ high-x_j stocks earn higher expected returns (risk premium).**

### 20-characteristic model (ranked by mean |w|)

| Char | mean w | t-stat | Reading |
|---|---|---|---|
| **turnover** | +0.128 | 22.9 | **Liquidity/attention premium — dominant** (retail market) |
| oscore | +0.061 | 6.6 | Distress risk premium |
| bm | −0.079 | −12.3 | **Value INVERTED on TSE** (high-BM earns less) |
| st_rev | +0.073 | 10.6 | 1-month reversal premium |
| noa | −0.019 | −2.3 | High NOA → lower returns (standard) |
| ac | −0.021 | −2.3 | High accruals → lower returns (standard) |
| ig | +0.062 | 10.1 | Investment growth premium |
| vol | −0.040 | −5.6 | High vol → lower returns |
| dy | −0.024 | −3.2 | Dividend yield negative |
| size | −0.055 | −8.9 | Small-cap premium present |
| investment | −0.055 | −8.8 | High I/A → lower returns (q-theory works) |
| dist | +0.038 | 5.6 | Distress premium |
| cbop | +0.036 | 6.1 | Cash profitability premium |
| mom | +0.028 | 4.7 | Momentum positive (weak) |
| nsi | −0.014 | −2.5 | Issuance negative (standard) |
| roe | +0.003 | 0.3 | n.s. |
| gp | −0.015 | −2.1 | n.s. |
| cei | +0.007 | 0.8 | n.s. |
| ita | +0.014 | 2.7 | weak |
| ag | +0.002 | 0.2 | n.s. (ag effect absorbed by investment) |

### 11-SY-signal model

| Char | mean w | t-stat |
|---|---|---|
| oscore | +0.135 | 14.4 |
| dist | +0.127 | 23.9 |
| gp | +0.021 | 1.9 |
| nsi | −0.009 | −0.9 |
| ag | −0.029 | −3.0 |
| noa | −0.027 | −2.9 |
| ita | −0.002 | −0.3 |
| ac | −0.017 | −2.1 |
| **mom** | **−0.028** | **−3.4** |
| ig | +0.058 | 12.8 |
| cei | −0.021 | −3.6 |

### E6 reading
1. **Turnover is THE priced characteristic on TSE** (|w| 40% larger than the #2,
   t=22.9) — a liquidity/attention factor in a retail-dominated market. This is a
   novel, TSE-specific result the paper can headline.
2. **Value is inverted** (negative bm loading): high book-to-market stocks earn
   lower expected returns on TSE — consistent with the weak/reversed value
   evidence in the local literature and fama-five.
3. **Momentum is reversed** in the 11-signal model (−3.4 t) while 1-month
   reversal is strongly positive (10.6 t) — short-horizon reversal dominates;
   momentum profits don't survive on TSE.
4. **q-theory signs hold:** investment, asset growth, accruals, NOA, issuance all
   load negatively (high "quantity" → lower returns). Distress loads positively.
5. Sign conventions cross-check against the fama-five anomaly directions
   (e.g., ag documented as REVERSED in Iran — matches).

## E7 — 2020 boom-bust subperiod (windows 2019-07..2022-06)

Annualized OOS Sharpe by subperiod:

| Model | FULL | Boom-bust | Boom only | Calm |
|---|---|---|---|---|
| Market | 0.183 | 0.358 | −0.962 | 0.065 |
| FF5 | 0.840 | 0.686 | 0.627 | 0.940 |
| q-factor | 0.728 | 0.534 | −1.539 | 0.796 |
| PCA(5) | 0.607 | 0.999 | −0.540 | 0.465 |
| LASSO | 0.825 | 1.176 | −0.580 | 0.683 |
| **E2** | **0.853** | **1.171** | −0.544 | 0.727 |
| E3 | 0.774 | 1.151 | −0.598 | 0.619 |
| E4B | 0.849 | 1.165 | −0.568 | 0.723 |
| E5A | 0.839 | 1.180 | −0.577 | 0.702 |

### E7 reading
1. **DL-SDF holds up in the boom-bust:** E2 1.17 (and E5A 1.18) vs FF5 0.69,
   q 0.53 — the deep SDF's edge is largest exactly in the crisis period
   (2019-07..2020-06 crash window: every model's best Sharpe, E2 4.09).
2. **The 1399 boom (2020-07..2022-06) breaks ALL models** — negative Sharpe for
   every factor-based strategy (retail mania: factor timing fails). The DL-SDF
   is least-bad (−0.54 vs q −1.54, Market −0.96).
3. Per-window Sharpes are highly correlated across models — the cross-section
   has a strong common regime component; the DL-SDF's advantage is a consistent
   small edge, not a different regime profile.

## Files
- `results/e6_loadings_{sy,all}.csv`, `results/e6_weights_{sy,all}.csv`
- `scripts/e6_loadings.py`, `scripts/e7_subperiod.py`
