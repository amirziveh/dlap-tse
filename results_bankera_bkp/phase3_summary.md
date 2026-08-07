# Phase 3 — E2/E3: Deep SDF Results (2026-08-03)

Deep-learning SDF (CPZ 2024 architecture, PyTorch):
- **M_net**: MLP z → SDF weights w(z) ∈ R^F, hidden [64,64], dropout keep 0.95
- **Z_net**: LSTM(6 macro series → 4 states), trained jointly
- **SDF**: M_{i,t} = 1 − w(z_t)′ x_{i,t}; loss = mean_i (E_t[M R^e])²
- Rolling 60/12/12, same protocol as E1; Adam lr 1e-3, early stopping (patience 25)
- E2 = 11 SY signals · E3 = all 20 characteristics

## Master results (`results/master_results.csv`)

| Model | OOS Sharpe | EV | RMS α % | max α % |
|---|---|---|---|---|
| Market | 0.183 | 0.3866 | 5.99 | 32.4 |
| FF5 | 0.840 | 0.3051 | 7.56 | 56.1 |
| q-factor | 0.728 | 0.2671 | 5.87 | 40.3 |
| PCA(5) | 0.607 | 0.4591 | 5.21 | 31.5 |
| LASSO | 0.825 | 0.4658 | 7.21 | 56.8 |
| **E2 DL-SDF (11 ch)** | **0.853** | 0.4698 | 7.29 | 65.6 |
| **E3 DL-SDF (20 ch)** | 0.774 | **0.4700** | 7.17 | 51.9 |

EV = explained return variation (test-sample OLS of stock returns on the model's
SDF portfolio return — same construction for ALL models). Sharpe = pooled OOS
annualized. α = E[M R^e] per stock, pooled.

## Reading (honest)

1. **Sharpe ranking:** E2 (0.853) > FF5 (0.840) > LASSO (0.825) > E3 (0.774) > q (0.728)
   > PCA (0.607) > Market (0.183). The deep SDF with the 11-signal set edges out
   every benchmark — the paper's headline criterion "DL > benchmarks on Sharpe"
   holds for E2. E3's 20-char version trails FF5/LASSO (more inputs, same capacity —
   overfitting signal on the small TSE cross-section; E6 loading analysis will tell).
2. **EV ranking:** E3 = E2 (0.470) > LASSO (0.466) > PCA (0.459) > Market (0.387)
   > FF5 (0.305) > q (0.267). Deep SDFs top the table; advantage over LASSO/PCA is
   narrow (0.004–0.011), over FF5/q large (0.16–0.20).
3. **Pricing errors:** RMS α 7.2% — mid-pack (PCA 5.2% best). Deep SDF does not win
   this metric; max α 51–66% reflects thin-stock tails.
4. **Regime sensitivity (E7 foreshadowing):** validation loss spikes in the 2019–21
   windows (COVID crash + 1399 boom): val_loss 1.2e-2 and 3.4e-2 vs ~1e-3 elsewhere —
   the SDF struggles to price through regime breaks, exactly the E7 subperiod test.

## Comparability notes (for the manuscript)
- EV: same construction everywhere (test-sample OLS on the SDF portfolio return).
- XS-R² in e1_summary.md is a stricter train-prediction metric (train β·λ vs test
  means): negative for all benchmarks; NOT directly comparable to EV — do not mix.
- Deep SDF trained on CPU (torch 2.13, venv `~/venvs/dlap-tse`), ~10 min for both runs.
- Deterministic: torch seed 42, CV seed 42.

## Files
- `results/e2_results.csv`, `e3_results.csv`, `e2/e3_pooled_series.csv`
- `results/master_results.csv` — canonical combined table
- `scripts/sdf_models.py` — ZNet/MNet · `scripts/train_e2.py` — runner (`--charset sy|all`)
