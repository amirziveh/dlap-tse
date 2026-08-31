# DLAP-TSE Revision Memo — 2026-08-31 (v2.0 → v3.0)

## A. Empirical issues resolved (genuinely rerun)

1. **Pakistan net-stock-issuance artifact (P4) — root-caused and removed.** The PK share-count
   series is PAT/EPS-implied for 2022–25 and flat-backfilled before 2022 (`pk_build_shares.py`),
   so YoY share changes (nsi) are extraction artifacts. 430 z-panel entries exceeded 5σ.
   `nsi` is excluded from the PK estimation set (same treatment as unavailable signals). All PK
   models, benchmarks (LASSO, linear SDF), and loadings re-estimated. Consequence: the PK
   dead-window guard no longer triggers (window 0 now trains on noa-coverage months), so PK
   evaluates on **6 windows** (was 5) and stock-months are 47,515 (was 47,518).
2. **Duplicate Asset Growth / Investment (I/A) signal (P5) — removed in all three markets.**
   The builders set `investment = ag` (byte-identical coverage in TR/PK). `investment` is dropped
   from the npz feature sets (IR/TR 19 chars, PK 17). All all-characteristic specifications
   (E3/E4A/E5B/E8B, linear SDF-all, LASSO, loadings) rerun in every market.
3. **Full-battery rerun (deterministic, seeds 42/43/44 × 8 deep specs × 3 markets = 72 runs,
   plus E1, linear SDF, pins, placebos, loadings, bootstraps, SPA).** Zero failures.
4. **P1 — Seed-wide formal inference.** Paired moving-block Sharpe bootstraps, SPA/Reality
   Check, and opposite-orientation loss gaps recomputed for every seed; summary in
   `results/seed_inference_summary.csv`. Loss-gap ordering (PK 0.04–0.22 << IR 1.55–1.65 <<
   TR 6.63–6.79) is stable across seeds.
5. **P2 — Paired inference for RMS pricing-error differences (new).** Per-(window, stock) alpha
   cells saved by all evaluators (`*_alpha_cells.csv`); new `rms_window_bootstrap.py` computes
   paired window-cluster bootstrap intervals (10k resamples) for E2/E3 vs Market, q-factor,
   LASSO, PCA(5), and both linear SDFs in every market.
6. **P6 — Strict predictive OOS R² for the deep models (new).** Train-window betas on the train
   SDF-portfolio return and train portfolio mean are frozen and applied to test months
   (symmetric with the benchmark construction). Deep OOS R²: −0.24…−0.25 (IR), −0.16…−0.18 (TR),
   −0.11…−0.12 (PK) — negative everywhere, but closer to zero than every factor benchmark.
7. **PK audit-chain re-verification against the frozen release:** 4,836 / 3,782 / 23 / 3,759 /
   899 / 174 (173 with retained PDF; AKBL-2015 reviewed without PDF) / 6 rotated-page fixes —
   all match the manuscript exactly.
8. **Loadings recomputed** on the deduped sets for all three markets and seeds 43/44
   (stability check), with block-bootstrapped t-stats.

## B. Main results that changed

- **PK portfolio results changed materially.** E2 seed range is now −0.142…0.694 (was
  −0.405…−0.079); the nsi artifact had polluted every PK deep model's training inputs. E8 at
  seed 42 reaches 1.186 — the highest deep Sharpe in the sample, with bootstrap intervals
  excluding zero vs FF5/q-factor/LASSO — but seed 43 is −0.323 and the Reality Check does not
  reject, so the paper reports this as extreme seed sensitivity, not superiority.
- **PK window count 5 → 6; stock-months 47,518 → 47,515.**
- **IR all-characteristic linear SDF:** Sharpe 1.310 → 1.376, RMS 9.58 → 8.89 (dedup effect).
- **PK LASSO:** 0.275 → 0.509 Sharpe, RMS 4.20 → 4.38 (input-set change).
- **PK linear SDF (all):** 1.636 → 1.523; linear SDF (SY): 0.281 → 0.325.
- **IR deep all-char specs** (E3/E4A/E5B/E8B) moved substantially (e.g., E3 s42 0.479 → 0.630);
  sy-set specs (E2/E4B/E5A/E8) reproduced byte-identically, confirming the rerun isolation.
- **TR:** all-char specs moved within ±0.02; everything else essentially unchanged.
- **New inference:** no pricing-error difference between E2 and any factor benchmark is
  significant; the only significant differences favor benchmarks (LASSO in PK; linear SDFs
  where their RMS is higher). Sign-window beat count is 16 of 24 (PK now 6 windows).

## C. Claims revised

- **"These rankings are descriptive because no inference concerns pricing-error differences"
  → replaced.** Paired window-level bootstrap intervals now exist; the abstract/intro/results
  describe *pricing parity* (all factor-benchmark intervals include zero), not unavailable
  inference.
- **"The pin fails in Pakistan" → recalibrated.** With the corrected inputs, the λ=1 pin
  *raises* the PK mean (0.342 → 0.416) but with the widest cross-seed dispersion (0.761/0.447/
  0.039); λ=10 falls to 0.100 with one negative seed. Framed as "weak orientation separation
  with high seed-to-seed dispersion."
- **PK taxonomy retained but re-evidenced:** closest loss geometry (median gap 0.21, seed
  span 0.04–0.22) + widest pin dispersion → "weak orientation separation plus high seed-to-seed
  dispersion," no longer "negative at all three seeds."
- **Liquidity filter:** PK E8's seed-42 significance is disclosed with its cross-seed flip and
  the Reality Check non-rejection; the "does not generalize" conclusion is unchanged.
- **Loadings:** IR investment-growth block now significant at seed 42 (|t| 2.4–3.0, negative)
  but shown to shrink toward zero at seeds 43/44; TR momentum (−3.8) and short-term
  continuation (+2.2) flagged; stability caveat strengthened.
- **HJ bound:** explicitly monthly, visually separated, "property of the test assets, not a
  model performance estimate."
- **Strict OOS R²:** now symmetric (benchmarks + deep), all negative, deep closer to zero.
- **Claim-discipline sweep:** no "dominates/universally/best-without-qualification" language
  remains; every comparative claim carries its evidence tier.

## D. Remaining limitations (post-revision)

1. **PK share counts remain unusable for issuance-type characteristics.** Repair would require
   the raw annual-report PDFs (decommissioned with the old server). nsi is excluded for PK, not
   corrected; PK `cbop`/`dy` coverage also remains thin.
2. **PK E8 seed-42 strength is unexplainable with available data** (one seed, Reality Check
   non-rejection); longer histories or additional seeds would be needed to say more.
3. **Best-of-specification comparisons remain descriptive.** No valid model-selection
   correction is implemented for the ex-post "best deep" rows; only the prespecified E2 carries
   intervals.
4. **Window-level bootstrap has 6 resampling units in TR/PK** (12 in IR) — intervals are
   accordingly wide; the design respects dependence but is coarse.
5. **Three seeds document, but do not exhaust, initialization sensitivity**; architectures
   beyond the prespecified families are out of scope.
6. **Corresponding-author email is a marked placeholder** pending the authors' institutional
   address.

## Build state

- `paper_gpt_rev/manuscript.tex` + `references.bib` + `figures/` — 23 pp,
  0 errors / 0 overfull / 0 undefined, bibtex clean.
- Verifier: `scripts/verify_manuscript_gptrev.py` → 0 errors (72 per-seed cells + all table
  rows checked against the canonical CSVs; stale-claim greps clean).
- Authors: Amir Ziveh (corresponding) + Sajjad Ghorbanali, Graduate School of Management and
  Economics, Sharif University of Technology (official English name verified via sharif.ir).
- References: 23/23 DOIs resolve on Crossref (talakesh2022 DOI verified via DOAJ;
  kellykuznetsov2026 cited as SFI 26-20 working paper, not as published).
