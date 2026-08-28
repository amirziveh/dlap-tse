# DLAP-TSE — Handoff / Session Status

**Last updated:** 2026-08-04 · **Project dir:** `/home/ubuntu/research/dlap-tse/`
**Read this first in any new session.** CLAUDE.md holds conventions (authoritative); this file holds *state*.

> 📌 **اگر روی دیتای فایننشال بینکشوری (IR/TR/PK) کار میکنی → اول `HANDOFF_FINANCIALS_2026-08-04.md` رو بخون** (ایران و ترکیه کامل شدن؛ پاکستان = قدم بعدی).

---

## TL;DR

**Project:** Replication of Chen, Pelger & Zhu (2024), "Deep Learning in Asset Pricing" (*Management Science* 70(2), 714–750) on the Tehran Stock Exchange (TSE) — a deep-learning stochastic discount factor `M = 1 − w(z)′x` that must beat FF5 / q / PCA / LASSO out-of-sample.

**Status: EXPERIMENTS COMPLETE + MANUSCRIPT v0.3 Q1-LEVEL, REFEREE-CLEAN (2026-08-03).**
- Best model **E2** (11 SY signals, LSTM states): OOS Sharpe **0.853** > FF5 0.840 > LASSO 0.825 > q 0.728 > PCA 0.607 > Market 0.183; EV **0.470** (deep SDFs) vs 0.305 FF5, 0.267 q; negative XS-R² for ALL linear benchmarks. Sharpe gaps vs FF5/LASSO/q are WITHIN bootstrap sampling noise (CIs include 0); significant only vs PCA and Market. Per-window: E2 > FF5 in 2/12 windows (18-19, 19-20); E2 > LASSO in 7/12; FF5's higher mean return (11.7%/mo) compounds to higher wealth (50.2 vs 23.3) despite −112%/−131% months. The paper says "consistently RANKED", not "significantly better".
- **E6 loadings:** turnover dominates (t=22.9), value INVERTED, **1-MONTH CONTINUATION (+0.073, t=10.6) — NOT reversal** (st_rev = raw past-month return; positive w = continuation; the US reversal anomaly does not transfer); q-theory MIXED (IG +0.062 positive, AG nil).
- **E4/E5 nulls:** macro conditioning and the adversarial critic add ~nothing on TSE.
- **E7:** deep SDF best vs FF5/q in 2019–22 boom–bust (1.171 vs 0.686/0.534) but LASSO TIES (1.176); boom window: all models except FF5 (+0.627) lose money, PCA −0.540 marginally beats E2 −0.544.
- **E8:** liquidity filter robust (0.851 vs 0.853).
- **Manuscript:** `paper/manuscript.pdf` (21 pp, English only — user removed the Persian abstract, backup in review/backups/manuscript_with_persian_abstract.tex). Q1 expansion done: desc-stats tables (tab:desc_panel, tab:desc_chars), 3 figures (wealth, cross-section, loadings), per-window Sharpe table, Sharpe-diff moving-block bootstrap (results/sharp_diff_bootstrap.csv), characteristic-definitions appendix (tab:char_defs), research questions, economic-interpretation subsection. Artifacts reproducible via `scripts/q1_artifacts.py` (venv). Audit trail in `review/` (briefs + referee reports + fix briefs + backups). Two full referee loops closed: (1) v0.2 consistency fixes, (2) Q1 expansion review (referee B → fixes → referee D CLEAN). OOS test period 2013-07..2025-06 (NOT 2026-06).
- **Code open-sourced 2026-08-03:** public repo `github.com/amirziveh/dlap-tse` (MIT; scripts now portable via DLAP_ROOT/FAMA_ROOT env vars; results/ included; data/, papers/, litrature review/, review/, code/, tools/, paper/ excluded — data rebuild documented in README). **The manuscript is NOT in the public repo** (user's decision: no author block yet, will be added upon submission); release v0.3.0 (which had carried the PDF) deleted. Manuscript §4.2 says "publicly available at \\url{...}" for the replication package (scripts+results).
- **Persian manuscript v1.0 COMPLETE (2026-08-03):** `paper_fa/manuscript_fa.tex` → `manuscript_fa.pdf` (21 pp, RTL xelatex + Bahij Nazanin). Full translation to Iranian Q1 journal skeleton (چکیده/کلیدواژهها + ۷ بخش + پیوست + منابع + English abstract at end). [n] bracket citations (first-appearance order, `paper_fa/refs_map.csv`), author-year مراجع list (50 entries, corpus-verified). Persian refs use corpus-corrected metadata (corpus-vs-English-bib discrepancies in `review/bib_discrepancies_corpus_vs_english.md`). Referee round 1 (subagent) → 3 fixes applied (osoolian 27(1) 85-113; terminology fixes) → manager re-verification + `review/manager_addendum_round01.md` (5 findings rejected with evidence; 4 new verify-before-submission items). Verification: `python3 /tmp/hermes-verify-persian-manuscript.py` = 14/14 PASS (compile×4, fonts, 50 cites, numbers, bidi, landscape). Figures: `paper_fa/figures_fa/*.pdf` (Persian labels via `scripts/q1_artifacts_fa.py`, arabic_reshaper+python-bidi, LRE-embedded dates). **Font: manuscript now uses patched family `Bahij Nazanin Persian` (~/.fonts/bahij_nazanin_persian*.ttf)** — Bahij's own Persian digits are Arabic-styled (6/7/0/8/9 = identical glyphs to its Arabic digits); the patch swaps in Vazirmatn's true Persian digit outlines (build script: `tools/patch_fa_digits.py`; 10/10 digits IoU 0.96–1.00 vs Vazirmatn). **Bidi traps solved (xelatex):** digit-hyphen-digit runs reverse in RTL prose AND bare digits reverse inside \begin{LTR} tables → wrap ALL date tokens and ALL table numeric cells in `\textenglish{}` (math mode also works); matplotlib labels need \u202a…\u202c LRE marks before get_display + strip controls.

---

## TRUE-CPZ RE-IMPLEMENTATION (2026-08-03, late session) — READ FIRST

**Why:** a GPT-5.6-sol referee review caught a model-identity flaw: the implemented
and manuscript-described SDF was a per-stock conditional linear characteristic
kernel `M_{i,t} = 1 - w(z_t)'x_{i,t}` — NOT the CPZ common SDF
`M_{t+1} = 1 - Σ_i ω_t(i) R^e_{t+1,i}` (ω = MLP over (z_t, x_{i,t}), verified against
`code/torch_port/.../models.py` SDFModel). The reviewer was right; the model has
been **re-implemented faithfully** and every result, table, figure, and both
manuscripts regenerated.

**New core (scripts/sdf_models.py):** `SDFNet` (dense over concat(z,x) → ω per
stock), `common_sdf` (M_t = 1 − (1/N_t)ΣωR, published CPZ form — the official
code's `/N_t × mean(N_t)` rescale was dropped: it cancels only when N_t is
constant and blew up the SDF scale on TSE), `pricing_errors_common` (common M),
`weighted_pricing_loss` (official count-weighted), `MomentsNet` critic (K=8,
tanh), `sdf_portfolio_return` (r_p = ΣωR/Σ|ω|, unit gross leverage).
`train_e2.py --arch cpz` (default) + `--arch charscore` (legacy per-stock,
robustness only, results under results/charscore/). New benchmark:
`linear_sdf_benchmark.py` (common linear SDF ω=θ'x, closed-form weighted LS).
Unit tests: `scripts/test_sdf_models.py` (12 tests, all PASS).

**New results (results/master_results.csv; benchmarks unchanged):**
- Deep specs (common SDF): E2 0.363/0.465/5.82%, E3 0.649/0.466/5.56%,
  E4A 0.712/0.462/5.65%, E4B 0.050/0.466/5.86%, E5A −0.025/0.466/5.82%,
  E5B 0.693/0.461/5.67%, E8 0.819/0.465/5.72%, E8B 0.818/0.463/5.53%
  (sharpe/EV/RMS α). Linear SDF: 11ch 0.374/0.141/6.85%, 20ch 0.713/0.209/13.14%.
- Story: deep SDF's no-arbitrage fit (α, EV) beats every characteristic-based
  benchmark; SDF-portfolio Sharpe fragile (E2 significantly below LASSO per
  bootstrap; E8 liq-filtered indistinguishable from FF5/LASSO, wealth 20.0 vs
  benchmarks 2.5–5.6 at unit leverage). Loadings (univariate OLS of ω on x):
  NONE significant under block bootstrap (strongest: st_rev +1.90, ac −1.82).
  Boom-bust: E2 1.107 (LASSO 1.175 leads). Macro states matter for 11ch
  (E4B 0.050); critic neutral-to-harmful; liquidity first-order (E8).
- EN + FA manuscripts fully rewritten (equations, tables, narrative) and
  compiled clean. `scripts/verify_manuscript_numbers.py` cross-checks every
  number in both manuscripts against the CSVs (PASS). `rerun_audit.sh` is the
  full reproducible re-run (seeded 42).

---

## Project map

| Path | Contents |
|---|---|
| `PLAN.md` | Original plan (E1–E8 catalog, timeline, risks) |
| `PHASE0_FINDINGS.md` | Feasibility answers (official code, macro sources, char set, q-factor) |
| `CLAUDE.md` | Project conventions — formation-year alignment, data sources, status (update it as you progress) |
| `data/` | `characteristics_panel.csv` (74,147 rows, 357 stocks, 20 chars), `characteristics_z.csv`, `macro_panel.csv`, `factors_q.csv`, `Char_all.npz` (303×357×21, official CPZ layout), `Macro_all.npz`, `meta.json`, `macro_raw/`, `README.md` |
| `scripts/` | `build_characteristics.py`, `build_macro_panel.py`, `build_qfactors.py`, `build_npz.py`, `eval_core.py`, `run_e1.py`, `sdf_models.py`, `train_e2.py`, `e6_loadings.py`, `e7_subperiod.py` |
| `results/` | `e1_benchmarks.csv`, `e2..e8*_results.csv`, `*_pooled_series.csv`, `master_results.csv` (**canonical 13-model table**), `e6_loadings_*.csv`, `e6_weights_*.csv`, `e1/phase3/phase4/phase5_summary.md` |
| `paper/` | `manuscript.tex` + `.pdf`, `references.bib` (50 entries, Crossref-verified) |
| `code/` | `official_cpz/` (authors' TF code, reference only), `torch_port/` (PyTorch port, reference only) |
| `litrature review/` | Corpus notes (35 batches), `litreview_synthesis.md`, `gap_analysis.md`, `litreview_section.md` (draft §2, has ⚠️ flags to verify) |
| `papers/` | 88-paper md corpus + `_packs/`, `_IDENTITY_FIXES.md`, style reports |

---

## What was done, phase by phase

- **Phase 0 (feasibility):** authors' official code located (`LouisChen1992/Deep_Learning_Asset_Pricing`, TF 1.12) + PyTorch port (`Darenar/DeepLearningAssetPricing_torch`) — both cloned to `code/` as architecture reference; hyperparameters extracted (hidden [64,64], dropout keep 0.95, LSTM 4 states, 240/60/300 windows, Adam 1e-3). Macro sources verified free & programmatic: World Bank (CPI, USD official), FRED csv / Yahoo BZ=F (Brent), tgju.org embedded chart data (gold coin from 2010-04, market USD from 2011-11). CBI tsd database unreachable even from inside Iran → bypassed. 20-char set confirmed buildable from fama-five data.
- **Phase 1 (data):** 20-char monthly panel (74,147 rows, 2001-03..2026-07, 357 non-financial stocks, formation-year aligned, per-month z-scores winsorized 1/99 + clipped ±10); macro panel (6 series: cbirate, cpi, usd_official, brent, gold_coin, usd_market); q-factors (HXZ 2×3 VW sorts); npz files in official layout with `-99.99` sentinel.
- **Phase 2 (E1):** benchmark battery — rolling 60/12/12, common period 2008-07..2026-06 (214 months, 12 windows, 144 OOS months); max-Sharpe weights with shrunk cov (δ=0.2); HJ bound 2.914 monthly. Findings: XS-R² negative for all linear models.
- **Phase 3 (E2/E3):** torch implementation (M_net MLP z→w, Z_net LSTM 6→4, loss = mean squared pricing errors, early stopping patience 25/400 epochs, Adam 1e-3). E2 (11 SY signals): **0.810/0.469**; E3 (20 chars): 0.796/0.468 (lagged alignment).
- **Phase 4 (E4/E5):** `--states const` (macro off) and `--critic` (adversarial critic, tanh weights, alternating Adam, loss_factor 1.0). Nulls hold on TSE (ΔSharpe ≤ 0.02).
- **Phase 5 (E6–E8 + write-up):** loadings (size dominant, BE/ME value positive, ig/oscore positive, turnover moderate); subperiod (boom-bust 1.14); liquidity filter (robust). Manuscript v0.4 (referee-clean) compiled and delivered.
- **Round-05 audit (2026-08-03):** three fixes applied and everything re-run:
  1. **Alignment (critical):** chars+macro were priced SAME-month (x_t→r_t). Fixed to x_{t-1}→r_t (CPZ convention) in `eval_core.lag_align`, wired into `train_e2.load_data` + `run_e1` (LASSO). Effect: E2 0.853→0.810; headline ranking changes.
  2. **Benchmark symmetry:** FF5/q factors rebuilt from winsorized returns (`scripts/build_winsorized_factors.py` → `data/factors_winsorized/`); raw factors carried capital-increase artifacts (RMW −95.6% in 2009-09). Effect: q 0.728→0.882, FF5 0.840→0.816.
  3. **BE/ME:** bm char rebuilt as book equity / market equity (was TE/TA leverage proxy; units: BE in million Rials ×1e6). Effect: value loading flips −0.079 → +0.059 (positive value premium).
  Plus leverage robustness (`scripts/bench_leverage_check.py` → Table 8): max-Sharpe benchmarks normalized to gross leverage 1 for wealth comparisons (FF5 raw leverage ~15x, min month −112% is a leverage artifact, not a data bug). Old results backed up in `results_legacy_20260803/`.

---

## Critical audit (2026-08-03, post-round-05): fixes + rerun

- **P2A confirmed data error:** the `investment` (I/A) column of `cbop_panel.csv` carried 17 annual unit-mismatch outliers (up to +27,738,800% asset growth; Rahavard unit mismatch per fama-five's own guard in `build_ff5_panels.py`). `Asset growth` (ag) and `Investment (I/A)` are the SAME construct (ΔTA/TA); their distributions differed only because of these outliers (ag 0.29±0.39 vs investment 46.5±3556). Guard `>50` → missing added in `scripts/build_characteristics.py` + `scripts/build_qfactors.py`; data rebuilt (chars panel, npz, winsorized factors) and ALL models rerun deterministically (seed 42). `results/master_results.csv` regenerated.
- **Effect of the fix:** benchmarks/11-char models essentially unchanged (q-factor 0.882→0.885, E2 0.810); 20-char specs moved within ±0.02 (E3 0.796→0.816, E4A 0.816→0.800, E8B 0.816→0.796). **The 20-char loadings changed materially** (e.g., turnover +0.047→+0.006, st_rev +0.023→+0.065, dist +0.059→+0.110): the loadings are sensitive to a 0.13% data perturbation.
- **P4B confirmed:** naive loading t-stats (mean/(sd/√144)) overstate precision — the 12 months of each window share one trained network. New canonical inference: moving-block bootstrap (block 6, 10k, seed 42) on the monthly weight series → `results/e6_loadings_boot_*.csv` (`scripts/loadings_bootstrap.py`). New significant loadings (20-ch, boot t): size −0.176 (−23.5), oscore +0.158 (5.3), ig +0.124 (7.7), dist +0.110 (5.6), st_rev +0.065 (4.9), cei +0.053 (2.6), noa +0.051 (6.4), ac −0.039 (−2.2), bm +0.037 (2.2), roe +0.035 (2.0), mom −0.026 (−2.1). NOT significant anymore: turnover, vol, gp, ita, investment (I/A), dy, cbop, ag, nsi. **Multi-seed check (43/44):** size/ig/dist/oscore/noa/bm/roe/st_rev/investment(−)/vol(−) keep sign; turnover (0.006–0.046), mom, ac, cei, dy flip or wander → loadings reported as EXPLORATORY with a stability caveat.
- **P3B confirmed:** HJ bound 2.914 is MONTHLY; model Sharpes annualized. Now reported as 10.1 annualized (2.914 monthly) everywhere; comparisons on the annualized scale.
- **P3C confirmed:** SDF-portfolio is NOT investable-long — 2.1% of test stock-months have negative SDF values (negative weights). "Investable long portfolio" language removed; described as weights-sum-to-one with small short positions.
- **P3A:** EV already disclosed as test-sample fit; now explicitly labeled "descriptive test-period fit measure" in method/abstract.
- **P1:** terminology fixed — SDF is linear in characteristics with network-learned weights (not "nonlinear SDF"); no-arbitrage restriction stated on excess returns (E[MR^e]=0, scale-free).
- **P5:** 2019–2022 subperiod labeled exploratory; stale English abstract in `paper_fa/sections/refs.tex` (pre-round-05 numbers, Sharpe 0.853) replaced with the new abstract.
- **Canonical Sharpe bootstrap:** `scripts/sharp_diff_bootstrap.py` reproduces `results/sharp_diff_bootstrap.csv` (verified vs pre-audit stored values).
- **Manuscripts revised:** `paper/manuscript.tex` + `paper_fa/` (all sections) recompiled cleanly (EN 2 passes, FA 4 passes xelatex). Figures regenerated (loadings figure now colors by bootstrap t).

## Key results (verified — do not re-derive, just cite `results/master_results.csv`)

| Model | Sharpe | EV | RMS α % | note |
|---|---|---|---|---|
| q-factor | 0.885 | 0.296 | 5.66 | winsorized factors |
| LASSO | 0.835 | 0.466 | 7.26 | lagged chars |
| E5B (20ch critic) | 0.826 | 0.467 | 7.24 | lagged |
| E8 (11ch liq-filter) | 0.820 | 0.467 | 7.19 | lagged |
| E4B (11ch const) | 0.820 | 0.468 | 7.15 | lagged |
| E5A (11ch critic) | 0.819 | 0.469 | 7.19 | lagged |
| E3 (20ch LSTM) | 0.816 | 0.468 | 7.06 | lagged |
| FF5 | 0.816 | 0.311 | 7.44 | winsorized factors |
| E2 (11ch LSTM) | 0.810 | 0.469 | 7.30 | deep SDF, lagged |
| E4A (20ch const) | 0.800 | 0.469 | 7.29 | lagged |
| E8B (20ch liq-filter) | 0.796 | 0.467 | 7.15 | lagged |
| PCA(5) | 0.607 | 0.459 | 5.21 | |
| Market | 0.327 | 0.393 | 5.69 | winsorized Mkt_RF |

Loadings (20-ch model, lagged, **moving-block bootstrap t**, seed 42): size −0.176 (t=−23.5), oscore +0.158 (5.3), ig +0.124 (7.7), dist +0.110 (5.6), st_rev +0.065 (4.9), cei +0.053 (2.6), noa +0.051 (6.4), ac −0.039 (−2.2), bm +0.037 (2.2), roe +0.035 (2.0), mom −0.026 (−2.1); the rest insignificant (turnover +0.006, investment −0.030, gp, vol, ita, dy, cbop, ag, nsi). **Loadings are seed-sensitive (43/44: turnover 0.006–0.046, mom/ac/cei/dy flip) → exploratory.** Subperiod: E2 1.141 (boom-bust) vs LASSO 1.175, FF5 0.494, q 0.749 (exploratory). Bootstrap: E2 beats only PCA(5) [+0.01,+0.54] and Market [+0.03,+1.07]; vs FF5 −0.006 [−0.65,+0.85], LASSO −0.025 [−0.05,+0.00], q −0.075 [−0.50,+0.42]. Wealth (leverage-normalized): E2 18.9, LASSO 21.6, q 5.3, FF5 2.5. Negative-SDF share in test: 2.1% (portfolio long–short in general).

---

## Environment & critical quirks (hard-won)

- **Python for torch:** `/home/ubuntu/venvs/dlap-tse/bin/python` (torch 2.13.0+cpu, numpy 2.5.1). System python has NO torch; PEP 668 blocks system pip. Scripts that need torch MUST use the venv python. Pure-numpy scripts work with either.
- **Hardware:** 2 cores / 3 GB RAM / disk was 100% full → pip cache purged (kept an eye on it). E2/E3 ≈ 5 min each; full E1–E8 re-run ≈ 35–40 min.
- **Deterministic:** torch seed 42, LASSO CV seed 42 → every run reproduces exact pins in `results/`.
- **float32 sentinel trap:** npz stores missing as float32 `-99.99`; converting to float64 gives `-99.98999786` → equality masking FAILS. Always mask with threshold: `arr[arr < -50] = np.nan` (returns ≥ −1 after winsorization; chars ≥ −10).
- **Returns need winsorization:** TSE capital increases produce monthly returns up to +1,692% — `build_characteristics.py` winsorizes per month at 1%/99% (2.41% clipped). Don't "fix" this away.
- **z-chars clipped ±10** (tiny-sd months otherwise explode downstream).
- **formation_year alignment (no look-ahead):** month (y,m) with m≥7 uses formation year y; m<7 uses y−1 (chars known July of fy, held July(fy)..June(fy+1)).
- **LASSO CD:** naive Jacobi updates diverge; must use cyclic Gauss–Seidel with column-energy normalization and incremental residual (see `run_e1.lasso_cv`).
- **Thin stocks:** skip stocks with train return variance < 1e-6 (betas overflow otherwise).
- **Persian tickers:** all CSVs UTF-8; never sort/index by position across files — join on ticker.
- **Telegram delivery:** pattern in the `telegram` skill (Bot API from `~/.hermes/.env`, chat 50471660). The user likes receiving phase reports + deliverables there.

---

## Verification state

- Everything (data → E1–E8 → manuscript) was verified with ad-hoc scripts during the session; all runs deterministic with exact pins. Transcripts hold the evidence.
- **TODO: canonical suite** — `scripts/verify_all.py` (unit tests + pin checks, ~1–2 min) + `make verify` target, mirroring fama-five. The session's verification tracker repeatedly showed stale state; a canonical suite gives it a stable command. Build this when time permits.
- Re-verify after any change: run the affected script, compare `results/*.csv` against the pins above (deterministic).

---

## What's next (TODO, in order)

1. **User reads the revised manuscripts (round-05 audit, 2026-08-03)** — English `paper/manuscript.pdf` + Persian `paper_fa/manuscript_fa.pdf`, both recompiled with the new (lagged-alignment, winsorized-factor, BE/ME) results. NOTE: the headline changed — deep SDF is now competitive-not-dominant (Sharpe 0.810 vs q 0.882/LASSO 0.835/FF5 0.816; EV 0.469 leads; wealth-normalized leads). The paper's framing was rewritten accordingly; read the new abstract first.
2. **DONE (2026-08-03 referee loop):** internal inconsistencies fixed (boom-bust claims, q-theory text, momentum, dates 2013-07..2025-06, pricing-error framing, EV-vs-XS-R² note, 48/12/12 vs 60/12, Table 4 completed to 20 rows, E8B row, bib entries talakesh 27(93):9–54 / taleblou 29(99):49–89 / davallou 21:89–106, all 50 refs cited). Iranian journal volume/page details still worth a final PDF eyeball before submission (corpus ⚠️ flags; authors of talakesh2023 transliterated per corpus note).
3. **Persian journal version** — DONE v1.0 (paper_fa/). Before submission: resolve the 4 verify items in `review/manager_addendum_round01.md` (talakesh first page 9 vs 10, namazi last page 134 vs 135, issue-number convention پیاپی vs within-volume, taleblou2022 journal name) + the 2 translated titles ([28], [43] — corpus has the real Persian titles for both, see referee report verify list). Also decide: fix `references.bib` osoolian 27(4)→27(1)+pp and namazi 9(26)→9(24) (English side).
4. **Optional robustness extensions:** richer macro panel (M2, IP — currently missing); alternative subperiods; delisting discussion (absent data, survivorship-safe rebalancing noted in the paper).
5. **Follow-up research idea (E6 finding):** turnover/attention premium as the dominant priced characteristic on TSE is novel — a standalone anomaly paper is a natural spin-off.
6. Housekeeping: delete stale `papers/_packs/_batches_remaining.json` (done once; it may have regenerated — check), keep `/tmp` clean (disk is tight). Consider promoting `/tmp/hermes-verify-persian-manuscript.py` into `scripts/` as the canonical Persian-manuscript check.

## Quick commands

```bash
cd /home/ubuntu/research/dlap-tse
V=/home/ubuntu/venvs/dlap-tse/bin/python
# data rebuild (pure numpy, any python)
python3 scripts/build_characteristics.py && python3 scripts/build_macro_panel.py \
  && python3 scripts/build_winsorized_factors.py && python3 scripts/build_npz.py
# E1 benchmarks (~4 min)
$V scripts/run_e1.py
# Deep SDF: E2/E3/E4/E5/E8
$V scripts/train_e2.py --charset sy            # E2
$V scripts/train_e2.py --charset all           # E3
$V scripts/train_e2.py --charset all --states const   # E4a
$V scripts/train_e2.py --charset sy --critic          # E5a
$V scripts/train_e2.py --charset sy --liq-filter      # E8
# Loadings / subperiod
$V scripts/e6_loadings.py --charset all
$V scripts/e7_subperiod.py
# Manuscript
cd paper && xelatex manuscript && bibtex manuscript && xelatex manuscript && xelatex manuscript
```

---

*Handoff written by Hermes, 2026-08-03. Future sessions: read this + `CLAUDE.md`, check `results/` for the latest numbers, and pick up the TODO list.*


---

## 2026-08-28 — 3-country manuscript rewrite (v1.0)

- `paper/manuscript.tex` is now the THREE-COUNTRY paper (Iran/Türkiye/Pakistan); the old IR-only
  v0.6 is backed up at `review/backups/manuscript_v06_ir_20260828.tex`.
- All IR analysis artifacts re-run on the bank-free vintage (sharp-diff bootstrap, SPA, leverage,
  loadings bootstrap, e2lag, placebo, e7, linear SDF both charsets, charscore, archsens,
  mechanism dumps + e8_mechanism, q1 figures). TR/PK E1 re-run byte-identical (deterministic).
- Canonical numbers: /tmp path pattern — regenerate via `scripts/render_manuscript_3c.py`
  (template: `paper/manuscript_3c_template.tex`, fail-loud on unresolved placeholders).
- Verifier: `scripts/verify_manuscript_3c.py` (0 errors: all table numbers vs CSVs, citation
  reconciliation, stale-claim greps, abstract-body consistency).
- Key scientific change vs v0.6: on the bank-free IR panel + TR + PK, the liquidity-filter
  stabilization does NOT transfer; the sign-ambiguity mechanism does (sign-normalized E2 above
  every factor benchmark in all 3 markets, stable across seeds; only superior anywhere = PK
  20-char linear SDF). IR linear SDF is now strong (1.07/1.31) — benchmarks tightened.
- Build: pdflatex ×4 + bibtex; 23 pp; 0 errors / 0 overfull / 0 undefined.

---

## 2026-08-28 (later) — Method B: ex-ante sign identification (commit 1780c77e)

- **`train_e2.py --pin-lambda L`**: adds `L·relu(−mean_train(r_p))²` to the TRAINING loss
  only (validation/early stopping stay on the pure pricing loss — no test leakage).
  Output name `e2pin<L>`; per-window `train_rp_mean` diagnostic printed (evidence the
  pin binds: train means ≥ 0; PK windows 0/2 saturate at +0.0000 at λ=1).
- **Sweep (all committed):** IR/TR/PK × seeds 42/43/44 × λ ∈ {1,10}.
  Results in `results{,_tr,_pk}/e2pin{1,10}_{results,pooled_series}.csv` (+ seed43/44 dirs),
  summary in `results/method_b_summary.json` (via `scripts/method_b_summary.py`).
- **Findings (pooled Sharpe, seed mean):**
  | market | raw | λ=1 | λ=10 | reading |
  |---|---|---|---|---|
  | IR | 0.171 | 0.267 (all seeds > 0) | 0.215 | pin repairs; λ=10 over-pins (some windows collapse to ~0 exposure) |
  | TR | 0.574 | 0.568 | 0.616 | already sign-stable; pin neutral |
  | PK | −0.002 | −0.207 | −0.322 | pin binds yet OOS worsens → genuine train→test sign instability |
  Window-sign agreement with the ex-post convention: IR 9/12,11/12,10/12; TR 5/6,6/6,6/6; PK 6/6,5/6,5/6.
  Post-hoc sign-convention Sharpe of pinned seed-42 series: IR 1.345, TR 1.434, PK 1.431.
- **Manuscript:** new `tab:methodb` (per-seed raw/pinned/agreement + SN(42) + λ=10 means),
  `sec:methodb` paragraph in Robustness, Fifth-contribution extension in the intro.
  Renderer reads `results/method_b_summary.json` (placeholders E2PIN_*, E2_MEAN, L10_*);
  verifier 5b audits λ=1 per-seed + all seed means. 24 pp, 0 errors / 0 overfull, verifier 0 errors.
- **TODO next:** Persian `paper_fa/` mirror of the 3-country rewrite (still IR-only v1.0);
  optionally the λ=10 per-seed series could feed a small exposure-collapse figure.

---

## 2026-08-28 (evening) — Referee fixes 1–3 + PK provenance/window-guard (commit 46954ee3)

External referee review rated v1.1 ~7/10, submit-worthy after 4 fixes. Items 1–3 done, item 4 (IPCA benchmark) still open.

1. **Sign-ambiguity math corrected**: flipping ω negates the covariance component of α_i but NOT
   the mean component (α(−ω) ≠ −α(ω) in general); symmetry is approximate. New empirical
   diagnostic in `train_e2.py` (eval-mode, deterministic): per-window L(+ω), L(−ω),
   `|L+−L−|/min` → `results*/e2_sign_symmetry.csv`. Median gaps: PK 0.39 (near-mirror) /
   IR 1.66 / TR 6.79 — discriminates the three identification regimes as predicted.
2. **Abstract/intro/limitations harmonized with Method B**; ex-post sign convention explicitly
   a diagnostic upper bound; limitations' "sign-pinned re-estimation = cleanest next experiment"
   replaced (it is now DONE) with formal-identification future work.
3. **`tab:diagnosis`** added (three identification regimes table).
4. **NOT DONE: IPCA benchmark** (referee: highest-value addition; discusses Kelly et al. but no
   benchmark). Identification wording also tightened (portfolio component up to orientation and
   scale; level fixed by constant 1).

**PK provenance bug (critical, found by accident):** `7a8dc6fa` enriched `data_pk/Char_all.npz`
but stored PK deep results predated it (proven: stored E2 0.112 reproduces ONLY on
`Char_all_orig.npz`). Full PK battery + E1 + linear SDF + placebos + sweep + charscore rerun on
the enriched vintage; everything downstream rebuilt.

**Latent PK dead-window bug (both vintages, incl. published v1.0):** PK window 0 (train
2014-10..2018-09) has **nsi** entirely missing under the all-core-char mask → all-NaN training
loss → untrained network polluting pooled OOS. Fix: disclosed no-coverage window guard in
`train_e2.py` + `done_windows` alignment fix in the EV loop (EV was misaligned after skips) +
tail-overlap alignment in `sharp_diff_bootstrap.py` (deep 60 vs bench 72 months) + rewrite-mode
dedupe in `linear_sdf_benchmark.py` (PK had duplicate lin rows).

**PK numbers now (enriched, guarded, 5 windows):** E2 raw −0.405 (seed spread −0.41/…), E4A 0.883,
E8B 0.445; ex-post SN E2 1.623 / E8 1.691 (artifact removal IMPROVED it); pins λ=1 mean −0.228 /
λ=10 −0.098 → train-to-test instability story unchanged and stronger. PK deep OOS window count
is now 5 vs benchmarks' 6 — ROADMAP/figure caption say so; `build_canonical_3c.py` (NEW script)
derives nwin dynamically from the pooled series.

Manuscript v1.2: 25pp, 0 errors/0 overfull, verifier 0 errors. PDF sent to Telegram (msg 19274).

---

## 2026-08-28 (night) — Referee round 2: 4 statistical-reporting fixes + IPCA exclusion note (commit eee02b8e)

External referee rated the science ~8/10, submit-worthy after 4 inconsistencies. All 5 items done:
1. **PK SPA claim inverted (was a real error):** "advantage is significant in Pakistan (p=0.709)" →
   "neither specification rejects the null in Pakistan either (SPA p 0.709--0.770)". TR nuance now
   reported honestly: E2 Sharpe-basis SPA flags at 5% (0.039/0.045) but Reality Check does not (p>=0.29);
   E8 not flagged (0.111).
2. **PK window count 6→5:** abstract now "6 in Türkiye and 5 in Pakistan";
   `fig_sign_windows` title now DYNAMIC from canonical nwin (was hardcoded "Pakistan (6 windows)");
   caption already used @@PK:NWIN@@.
3. **PK bootstrap contradiction fixed:** was "E8 vs strongest pairs statistically indistinguishable
   from zero"; truth is E2 AND E8 significantly BELOW PCA(5) (−1.28 [−2.33,−0.40] / −1.34 [−2.42,−0.44],
   zero_excluded=1), all other pairs span zero. New @@PK:BOOT_E2_PCA@@ placeholder added to renderer.
4. **"at or above the strongest factor benchmark" removed everywhere** (abstract, intro Fifth
   contribution, conclusion, tab:sign caption): truth = above in IR (E2-SN 1.32–1.35 vs 0.888) and
   PK (1.52–1.62 vs 0.467), essentially TIED in TR (E2-SN 1.476–1.615 vs q-factor 1.6147 — 3-dp tie at
   seed 44, below at 42/43). New wording "competitive with the strongest factor benchmarks in each
   (above them in Iran and Pakistan, essentially tied with the Turkish q-factor)".
   Renderer's SIGN_CLAIM block now classifies above/tied/below from the data (claim auto-updates on
   re-render; template claims are static + verifier-grepped).
5. **IPCA exclusion paragraph** (`\paragraph{Why no IPCA benchmark.}` end of sec:bench):
   three grounds — (a) 316–484 stocks × 48-month windows → nearly-square T×N, estimator unvalidated
   at frontier dimensions; (b) benchmark-set comparability (all benchmarks share the deep model's
   squared-pricing-error estimator class; LASSO/linear-SDF already cover the linear null);
   (c) researcher degrees of freedom in instrument construction. Cites kps2019 (already in bib).

Verifier: 6 new stale-claim regexes (at-or-above, significant-in-Pakistan, indistinguishable-vs-PCA,
exceeds-every-benchmark, PK-6-windows variants). **v1.3: 26pp, pdflatex×4, 0 err / 0 overfull /
0 undefined; verify_manuscript_3c 0 errors.** PDF at /tmp/DLAP-TSE_Manuscript_EN_v1.3_referee-fixes.pdf.
NOTE: paper_fa/ is still the IR-only Persian v1.0 — 3-country mirror remains the main open item.

---

## 2026-08-28 (late night) — Referee round 3: symmetry framing calibrated (commit 858b7199) → v1.4 SUBMISSION-READY

Referee moved readiness to 8.5–9/10 ("polishing round on sign-mechanism framing, then submit").
Two wording fixes applied:
1. **Universal "approximately symmetric" claim removed everywhere** (abstract, intro Fifth
   contribution, sec:signconv): the paper's own diagnostic contradicts it (median gaps
   PK 0.39 / IR 1.66 / TR 6.79). New claim: "weak portfolio-orientation identification,
   though its intensity differs by market" + explicit gaps in the abstract; sec:signconv now
   says "near-mirror loss geometry is one mechanism of portfolio instability rather than a
   universal characterization of all three markets" with dynamic PK median
   (@@PK:SYM_MED@@/@@PK:SYM_PCT@@ new placeholders = 0.39 → ~39% loss gap).
2. **Conclusion claim fixed**: "identification problem before it is an economic one" →
   "identification problem and, in the least persistent market, also a temporal-stability
   problem" (PK Method B shows train-to-test instability, not just identification failure).
Verifier: 2 new stale-claim regexes (universal-symmetry, before-economic-one).
**v1.4: 26pp, 0 err / 0 overfull / 0 undefined, verifier 0 errors.**
PDFs sent to Telegram: v1.3 (msg 19275), v1.4 (msg 19276).
**STATUS: EN manuscript submission-ready per external referee. Remaining: Persian mirror
(paper_fa/ still IR-only v1.0), optional λ=10 exposure-collapse figure.**

---

## 2026-08-28 (latest) — Bibliography verification round → v1.5 (EN, refs-complete)

External referee bibliography audit of v1.4: 9 wrong entries, 2 missing, 1 misattributed
theory claim. ALL resolved (EN `paper/references.bib` + FA `paper_fa/sections/refs.tex`;
details + sources in `review/bib_discrepancies_corpus_vs_english.md` § RESOLVED):

1. **9 entry corrections (EN bib):** cwh2026 (Chen Xing/Wang Jun/Huang Rui + DOI
   10.1016/j.iref.2026.105402), wcw2025 (Shunyao Wang/Ming Cheng/Christina Dan Wang),
   dixon2022 (Goicoechea Kemen; arXiv:2206.10014 — was "Andres", SSRN), kmz2024
   (Zhou Kangying + DOI 10.1111/jofi.13298 — was "Hao"), nabipour2020 (6 authors incl.
   Salwana+Shamshirband, full MDPI title, DOI), osoolian2025 (Mohammad/Ali/Mahdi;
   27(1) 85–113 + DOI; was Maryam/Hossein/Farzaneh 27(4)), raefi2018 (Raoofi/Mohammadi
   Teimour, Iranian Economic Research 23(76) 107–136 + DOI; PDF page-1 footer says
   «از 107 تا 136» — referee's 107 was right, old Persian list had 109), namazi2007
   (9(24) 115–134 per the article PDF itself), esfahanipour2011 (Mardani Parvin;
   INISTA 2011 pp. 44–49 + DOI; now @inproceedings).
2. **Corpus-primary sweep of the rest of the Iranian list** (NoorMags headers = the
   publisher records the referee couldn't find): azar2010 (Financial Research Journal
   11(28) 3–20; Karimi Siros; NOT Accounting and Auditing Review), chavoshi2003
   (Raei+Chavoshi order, 5(15) 97–120), heidari2022 (Mahdi/Hamidreza, 24(4) 602–623),
   hadizadeh2023 (Anita/Tarokh/Majid, Modern Research in Decision Making 7(4) 51–80),
   tehrani2025 (Jafari Seyed Morteza, JFESM 16(62) 24–48), bahmani2024 (Maryam/
   Mohammad Ebrahim/Mehrdad, JFESM 15(58) 1–20), sadeghi2025 (Amir/Amir, JFESM 15(61)
   147–167), soleimanian2020 (Financial Accounting 11(44) 37–62), raei2011 (Quarterly
   Journal of Quantitative Economics 7(4) 101–116), talakesh2023 (journal = Iranian
   Economic Research; author order Talakesh/Mohammadi/Taleblou/Mohajeri), taleblou2022
   (9(2) 83–112), taleblou2024 (Bagheri Todeshki Mohammad Mehdi), davallou2015 (title/journal polish).
   NOTE: soleimanian2020/raei2011/bahmani2024/sadeghi2025 are not cited in the EN
   3-country paper (bibtex drops them); corrections ready for any future citing.
3. **Two missing refs added:** campbell2008 (In Search of Distress Risk, JF 63(6)) and
   ohlson1980 (JAR 18(1)) — now cited in Appendix tab:char_defs via \citealt
   ("simplified Campbell et al. 2008 proxy" / "simplified variant of Ohlson 1980").
4. **Kelly et al. theory claim fixed:** kmz2024 no longer credited with the deep-SDF
   regularization result. Sentence now: kmz2024 = misspecification + optimal shrinkage →
   complexity virtue; NEW \citet{kellykuznetsov2026} (Large and Deep Factor Models,
   arXiv:2402.06635 v3, Kelly/Kuznetsov/Malamud/Zhang 2026) = exact linear factor
   representation of DNN-SDF (portfolio tangent kernel), spectral complexity + regularization
   govern finite-sample pricing. Verified wording against the arXiv v3 abstract.
5. FA mirror: refs.tex [38] pages added, [39] full author list + title, [43] ۱۰۷-۱۳۶ +
   محمدی تیمور. FA rebuild 29pp, 0 overfull.
- **Build EN:** renderer + pdflatex×4 + bibtex → 26pp, 0 err/0 overfull/0 undefined;
  bibtex only pre-existing davallou "no volume" note; verifier 0 errors
  (campbell2008/ohlson1980 aux-notes are benign — they enter via \citealt).
- Backups: `review/backups/references_bib_pre_bibfix_20260828.bib`,
  `review/backups/manuscript_3c_template_pre_bibfix_20260828.tex`.
- Remaining pre-submission (unchanged from v1.4): normalize bibliography style to the
  target journal's format; the ten (13) FA-listed Iranian sources now match corpus
  primaries, so no further metadata work expected there.
