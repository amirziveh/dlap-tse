# Phase 4 — E4/E5: Macro States & Adversarial Critic (2026-08-03)

Same protocol as E2/E3 (rolling 60/12/12, common period 2008-07..2026-06, 12 windows).

## Experiments
- **E4a/E4b** — `--states const`: learned constant state, NO macro conditioning
  (macro off). Tests whether time-varying SDF weights matter on TSE.
- **E5a/E5b** — `--critic`: adversarial CriticNet (z → portfolio weights in
  [-1,1]^F, tanh) maximizes (E[M r_c])²; SDF minimizes it too (alternating
  Adam, loss_factor 1.0). Tests robustness of the SDF.

## Master results (`results/master_results.csv`, sorted by Sharpe)

| Model | Sharpe | EV | RMS α % |
|---|---|---|---|
| **E2 (11ch, LSTM)** | **0.853** | 0.4698 | 7.29 |
| E4B (11ch, const) | 0.849 | 0.4705 | 7.22 |
| FF5 | 0.840 | 0.3051 | 7.56 |
| E5A (11ch, critic) | 0.839 | 0.4702 | 7.47 |
| LASSO | 0.825 | 0.4658 | 7.21 |
| E3 (20ch, LSTM) | 0.774 | 0.4700 | 7.18 |
| E4A (20ch, const) | 0.771 | 0.4683 | 7.19 |
| E5B (20ch, critic) | 0.751 | 0.4695 | 7.34 |
| q-factor | 0.728 | 0.2671 | 5.87 |
| PCA(5) | 0.607 | 0.4591 | 5.21 |
| Market | 0.183 | 0.3866 | 5.99 |

E5 critic adversarial portfolio pricing error: E5A 0.67% / E5B 0.70% (monthly).

## Findings

1. **E4 — macro conditioning does NOT matter on TSE.** LSTM vs constant states:
   11ch 0.8530 vs 0.8486 (Δ0.004); 20ch 0.7740 vs 0.7708 (Δ0.003). EV moves
   ≤0.002. The SDF weights are effectively time-invariant on TSE — consistent
   with a retail-driven market where macro timing is weak. (Caveat: 6-series
   macro panel; more/lagged series could change this — robustness option.)
2. **E5 — adversarial critic does NOT help on TSE.** Critic on vs off: 11ch
   0.8530 → 0.8394 (−0.014); 20ch 0.7740 → 0.7514 (−0.023). EV unchanged
   (≈0.470). The critic's worst portfolio still carries ~0.7%/month pricing
   error, but forcing the SDF to chase it costs Sharpe OOS. On the small TSE
   cross-section the adversarial loop overfits the critic rather than
   regularizing the SDF (opposite of CPZ on US).
3. **Parsimony wins:** every 11-char model beats its 20-char twin by
   0.02–0.09 Sharpe (E2>E3, E4B>E4A, E5A>E5B). The small cross-section
   (≤350 stocks) punishes extra inputs.
4. **Best model overall: E2** (11 SY signals, LSTM states) — Sharpe 0.853,
   EV 0.470, and the simplest story: the 11 Stambaugh–Yuan-style signals
   already carry the TSE pricing signal; the deep SDF's nonlinearity in
   characteristics is what beats linear benchmarks (EV +0.16 over FF5).

## Manuscript notes
- E4/E5 are honest null results — the paper's contribution stands on E2/E3
  vs benchmarks; E4/E5 become robustness sections ("macro states and the
  adversarial critic add little on TSE — documented, not swept under").
- Compare with CPZ US: states + critic help there; TSE differences are
  institutional (small cross-section, retail, sanctions regimes).

## Files
- `results/e4a/e4b/e5a/e5b_results.csv` (+ pooled series)
- `results/master_results.csv` — canonical 11-model table
- `scripts/sdf_models.py` — ConstZNet, CriticNet, critic_alpha
- `scripts/train_e2.py` — `--states {lstm,const}` `--critic`
