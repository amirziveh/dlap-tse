# PK Data Full-Coverage Plan — DLAP-TSE (Q1-Worthy Data Gate)

> **For agentic workers:** execute phase-by-phase; each phase ends with a STOP gate for user approval.
> Companion docs: `HANDOFF_FINANCIALS_2026-08-04.md`, `notes/3country_data_status.md`, `data_pk/audit/data_appendix.md`, `data_pk/PK_REVIEW_TODO_2026-08-05.md`.

**Goal:** Bring Pakistan (PSX) data to full coverage — 100% core financials, ≥95% all other fields, complete characteristics/macro/returns panels, and a referee-proof audit trail — so no Q1 referee can reject on data grounds.

**Architecture:** Reuse the existing PK pipeline (`scripts/pk_*.py`: download → VLM extract → build → QA). Work in waves: (1) close financial holes, (2) rebuild derived panels from clean financials, (3) expand external verification, (4) cross-country parity with IR/TR, (5) final Q1 gate.

**Tech stack:** Python 3.11 (venv `~/venvs/dlap-tse`), existing scripts in `scripts/`, VLM extraction (qwen 8192-token cap), git in repo root. Costs: ~$3–6 VLM for ~2,500–3,000 page extractions.

---

## Global Constraints

- Schema (16 fields): `country, symbol, year, TA, TL, Eq, CA, CL, Sales, COGS, GP, PAT, Cash, Inv, PPE, dividends, dividends_paid` (+ metadata: units, unit_factor, field_factors, pages, flags, status). PK files use `symbol`; price files use `ticker` — same value.
- Empty = `''`, `None`, `nan`, `NaN`, `NA`, `N/A`, `-`.
- **No silent fixes:** every changed value must be traceable to a source page (page + line) — the audit trail is the deliverable.
- **Data-QA protocol (user rule):** review rows are applied ONE ROW AT A TIME — show page image + extracted values, wait for user confirm before writing to the final CSV.
- Unit policy: PKR millions canonical; `unit_factors.json` per symbol-year (1.0 / 0.001 / 1000.0); every row MUST carry a factor; cross-check with `audit/dps_crosscheck.csv` (already at 1.000 exact for 15/16 checks).
- Banks: decision required from user (see Decisions). Current files: `financials_annual_no_banks.csv` (4,663 rows) and `financials_annual_no_financials.csv` (3,782).
- Phase gates: each phase ends → STOP → user approves → next phase. No exceptions.
- One deliverable file at a time; main CSV finalized before aux files.

---

## Current State (measured 2026-08-08 — baseline numbers)

| Layer | State | Gap |
|---|---|---|
| Financials rows | 4,836 rows, 460 symbols, 2013–2026 | 2026 only 3 rows; 2013 only 204 rows |
| TA / Eq / Sales / PAT | 88.0% / 67.9% / 66.7% / 75.2% filled | target 100% of universe symbol-years |
| Other fields | Cash 79.1%, CA 81.5%, CL 80.4%, COGS 65.0%, GP 62.4%, Inv 57.7%, cfo 49.0%, PPE 42.2% | target ≥95% |
| dividends | 70 rows (1.4%) | target ≥90% via DPS×shares |
| dividends_paid | 1,346 rows (27.8%) | target ≥90% |
| Review rows | 899 rows status=review; flags L2/L3/L5 on 1,377 rows | resolve or document 100% |
| HOLEs (symbol-year missing entirely) | 258 rows, 184 distinct symbols | fill or document (non-compliant/delisted) |
| VALUE defects | 144 negative Eq, 6 negative PPE, 2 negative TA, 10 tiny_TA_unit? | vet each vs source page |
| unit_factor | 95 rows missing | fill; rebuild `unit_norm/` |
| field_factors | empty in 4,832/4,836 rows | decide: document legacy or drop column |
| Characteristics | 47,518 rows, 21 cols, 2014–2026 | 41,596 missing core cells; 5,886 noncore; no 2013 |
| Macro | 168 rows, 5 series (policy_rate, tbill, cpi_ix, cpi_yoy, usd_pkr, brent) | 2026-09..12 empty; target 12 series |
| Risk-free | 164 months 2013-01..2026-08 | complete; extend to 2026-12 |
| Monthly returns | 66,902 symbol-months, 654 tickers, 2013-12..2026-08 | trailing `None` column; universe clean needed |
| Market cap / shares | 61,187 / 6,328 rows, 452 symbols | match to cleaned universe |
| External verification | 3,909 checks in `verification_report.csv` (sources: dps + sarmaaya) | EXACT-level 3,323 (85%), MISMATCH 107, WARN 95 — resolve the 202, target ≥99% clean |
| Excluded rows | 1 (PREMIER TEXTILE FY2018 — 404 at PSX, corrupt Wayback, no company copy) | document; acceptable |
| Universe | `symbols.json` 1,024 entries incl. TFC/debt/ETF | build clean equity universe |

**Referee-risk register (what a Q1 referee can attack):**
1. Missing financials → survivorship-ish selection + unstable characteristics → **fix via Phase 1–2**.
2. No external verification → data "made up" suspicion → **Phase 5**.
3. Unit/scale errors (0.001/1000 mix) → **Phase 1 task 5**.
4. Banks excluded without justification → **Decisions + Phase 1 task 6**.
5. dividends empty everywhere → ROE/investment char distortion → **Phase 1 task 3**.
6. No reproducibility → scripts + git tags + one-command rerun → **Phase 5**.
7. IR/TR treatment differs from PK → inconsistent across countries → **Phase 6**.

---

## Phase 0 — Baseline Freeze (no data changes)

**Deliverable:** `data_pk/GAP_MATRIX_2026-08-08.csv` + git tag `pk-data-baseline`.

- [ ] T1: Generate coverage matrix: 16 fields × year × filled%, per-symbol row counts, flag counts. Script: new `scripts/pk_gap_matrix.py` (reads `financials_annual.csv`, `data_gaps_report.csv`, `characteristics_panel.csv`, `macro_panel.csv`, `processed/monthly_returns.csv`).
- [ ] T2: Copy current files to `data_pk/baseline_2026-08-08/` (financials, characteristics, macro, npz, manifests).
- [ ] T3: `git add -A && git commit -m "pk: baseline freeze 2026-08-08"` + tag `pk-data-baseline`.
- [ ] T4: Print final baseline numbers (table above) into `GAP_MATRIX` header.

**Exit gate:** user reviews matrix → **STOP**.

---

## Phase 1 — Financials to Full Coverage (PK)

### Task 1.1: Resolve 899 review rows (flags L2/L3/L5)
**Files:** `data_pk/vlm_rows/*.json`, `data_pk/review/*.json`, `data_pk/qa_report.csv`, `data_pk/audit/review_summary.csv`
- [ ] Split review queue by flag: L5 (378 rows) = re-extract with VLM at higher-res page crop; L3 (305) = re-extract once; L2 (168) = keep but verify arithmetic (TA−TL=Eq, GP=Sales−COGS) vs `page_manifest/` images.
- [ ] Re-run `scripts/pk_vlm_pipeline.py` on the review-only manifest with `--only-flags L3,L5` (existing flag support), output to `review/v2/`.
- [ ] Each re-extracted row: diff vs old value → if changed, write to `review/resolved_YYYY-MM-DD.csv` with old/new/page/confidence. **User rule:** apply ONE row at a time (show page image + old→new, wait for confirm).
- [ ] Arithmetic re-check for L2 rows via `scripts/pk_gate_check.py` (existing) → resolve mismatches against source page image.
- [ ] Update `financials_annual.csv` status: review→ok only for rows confirmed against page.
- [ ] Commit per batch of 50 confirmed rows: `git commit -m "pk: resolve review rows 1-50"`.

**Exit:** 0 unresolved review rows in final file (any irreducible → documented in `data_gaps_report.csv` with evidence).

### Task 1.2: Fill 258 HOLEs (184 symbols with missing years)
**Files:** `data_pk/pk_manifest.json`, `scripts/pk_download_pdfs.py`, `data_pk/raw_pdfs/`
- [ ] Build hole list: for each symbol in universe, missing years in 2013–2025 (2026 partial OK).
- [ ] Fetch PDFs in priority order: PSX portal → Wayback (`te_wayback_policy.json` exists) → company website. `pk_download_pdfs.py` already implements this chain — extend only if a new source needed.
- [ ] VLM-extract missing symbol-years (est. ~700–1,000 pages, ~$2–4).
- [ ] For irrecoverable holes (delisted, non-compliant, source corrupt — like PREMIER TEXTILE): record symbol/year/evidence in `data_gaps_report.csv` level=HOLE with reason; keep documented, do NOT fabricate.
- [ ] Commit: `git commit -m "pk: fill holes wave N"`.

**Exit:** every hole either filled or has evidence-documented reason; coverage of active universe symbol-years ≥99%.

### Task 1.3: dividends via DPS × shares (target ≥90%)
**Files:** `data_pk/shares_annual.csv`, `data_pk/audit/dps_crosscheck.csv`, new `scripts/pk_build_dividends.py`
- [ ] Source: dps.psx.com.pk per symbol-year DPS (the site already used for `dps_crosscheck.csv`); fallback: annual report "dividend per share" note page.
- [ ] `dividends = DPS × shares_annual(shares)` — mirror IR logic (`build_ir_financials.py`: `pure_dps × capital`).
- [ ] Cross-check computed dividends vs existing 70 extracted values: ≥90% within 1% → else manual per-row review.
- [ ] Also re-scan cash-flow statements for `dividends_paid` (target ≥90%) via targeted VLM pass on CF pages of rows currently empty (3,490 rows, ~$1).
- [ ] Append `dividends_source` and `dps` columns to the CSV (referee traceability).
- [ ] Commit.

**Exit:** dividends ≥90%, dividends_paid ≥90%, dps cross-check table ≥90% rows EXACT/1.000.

### Task 1.4: Vet 162 VALUE defects
**Files:** `data_pk/data_gaps_report.csv` (level=VALUE), `page_manifest/` images
- [ ] 144 negative Eq: for each, open source page image; if genuinely negative (e.g. accumulated losses > equity), keep + flag `neg_eq=ok`; if sign/parse error → fix value from page.
- [ ] 6 negative PPE, 2 negative TA, 10 tiny_TA_unit? (suspect 0.001 factor) — same one-by-one vetting.
- [ ] Commit per resolution batch with page evidence.

**Exit:** 0 unexplained VALUE defects; each resolved or documented.

### Task 1.5: Unit normalization audit
**Files:** `data_pk/unit_factors.json`, `unit_norm/`, `financials_annual.csv`
- [ ] Fill 95 rows missing unit_factor (use `unit_norm/` neighbors + page scan).
- [ ] Rebuild `unit_norm/` via existing normalization script; assert every row has factor.
- [ ] Decide `field_factors` (4,832/4,836 empty): if legacy no-op → remove column and document in README; if meaningful → populate. **User decision needed.**
- [ ] Rebuild `financials_annual_units_normalized.csv` + verify no magnitude jumps (TA ratio year-over-year within [0.2×, 5×] for same symbol, excluding splits/restatements).

**Exit:** 100% rows have unit_factor; unit audit table with 0 unexplained anomalies.

### Task 1.6: Banks & financial-sector policy
**Files:** `data_pk/financials_annual_no_banks.csv`, `financials_annual_no_financials.csv`
- [ ] **User decision first** (see Decisions): (a) include banks with bank schema (TA, TL, Eq, deposits, loans, PAT — different field set; CPZ-style papers usually exclude financials), or (b) keep exclusion + one-paragraph justification in data appendix.
- [ ] If (a): extract bank-specific fields for ~25 bank symbols × years missing; if (b): ensure `no_financials` version used in paper is consistent and documented.
- [ ] Same decision for insurers/REITs (TR already excludes REITs — parity check in Phase 6).

### Task 1.7: v3 merge
- [ ] Merge all resolutions → `financials_annual_v3.csv` (main deliverable; one file at a time).
- [ ] Rebuild variants: `_no_banks`, `_no_financials`, `_partial_rows`.
- [ ] Full identity checks: Eq=TA−TL within 1% tolerance (except documented), GP=Sales−COGS, CA≥Cash+Inv sanity, no negative TA/PPE.
- [ ] Update `PACKAGE_README.md` + commit.

**Exit gate:** v3 coverage matrix: TA/Eq/Sales/PAT = 100% of universe, all 13 real fields ≥95%, dividends ≥90% → **STOP** (user reviews v3 + gap matrix diff).

---

## Phase 2 — Characteristics Rebuild

**Files:** `scripts/pk_build_characteristics.py`, `data_pk/characteristics_panel.csv`, `characteristics_z.csv`, `pk_build_npz.py`, `Char_all.npz`
- [ ] T1: Rebuild characteristics from v3 financials (fixes the 41,596 missing core cells that trace to missing financial inputs). Keep same 21 columns; verify per-year rows: target ≥3,500 symbol-months/year for 2014–2026.
- [ ] T2: Extend to 2013 (returns exist from 2013-12; characteristics need 12-month lag → likely 2014 start is correct — document why, don't force).
- [ ] T3: Re-run z-scoring (`characteristics_z.csv`) with same cross-sectional method; re-run winsorize + `pk_build_npz.py` → new `Char_all.npz`, `Macro_all.npz`.
- [ ] T4: Coverage report: % cells filled per char per year; target ≥95% core chars (size, bm, mom, roe, inv, noa, ac, ag).
- [ ] T5: Confirm npz dims (months × N × K) match returns panel; commit.

**Exit gate:** characteristics ≥95% core cells; npz rebuilt → **STOP**.

---

## Phase 3 — Macro & Risk-Free Completion

**Files:** `scripts/pk_build_macro.py`, `data_pk/macro_panel.csv`, `risk_free_rate.csv`, `macro_sources.md`
- [ ] T1: Fill 2026-09..2026-12 macro rows (SBP policy rate, t-bill, CPI from PBS, USD/PKR, Brent) — sources in `macro_sources.md`.
- [ ] T2: Extend series to 12: add M2, industrial production (or proxy), term spread, KSE-100 index level, FX reserves — document each source + retrieval date in `macro_sources.md`.
- [ ] T3: Extend risk-free to 2026-12 (T-bill 3M series).
- [ ] T4: No trailing empty rows; assert in script (`pk_gate_check.py` extension or new assert).
- [ ] Commit per series batch.

**Exit gate:** macro 12/12 series, full span, sources documented → **STOP**.

---

## Phase 4 — Returns / Market-Cap / Universe QA

**Files:** `data_pk/processed/monthly_returns.csv`, `volume_monthly.csv`, `market_cap_monthly.csv`, `shares_annual.csv`, `symbols.json`
- [ ] T1: Drop trailing `None` column in `monthly_returns.csv`; normalize ticker casing; dedupe.
- [ ] T2: Build clean equity universe from `symbols.json` (exclude isETF/isDebt/TFC; keep listed equities 2013→2026); document count (est. 500–550).
- [ ] T3: Coverage report symbol×month: returns vs universe — target ≥95% of active listings per month; missing = delisted/unlisted (documented, survivorship-safe by construction: daily snapshots `raw/*.txt` are point-in-time).
- [ ] T4: Reconcile market cap = close × shares vs `market_cap_monthly.csv` values; fix mismatches.
- [ ] T5: Write survivorship statement for data appendix (using point-in-time `raw/` snapshots + delisting notes).
- [ ] Commit.

**Exit gate:** clean universe, ≥95% monthly coverage, survivorship statement drafted → **STOP**.

---

## Phase 5 — Referee-Proof Verification & Reproducibility

**Files:** `data_pk/verification_report.csv`, `audit/dps_crosscheck.csv`, `audit/data_appendix.md`, `scripts/`
- [ ] T1: Investigate the 107 MISMATCH + 95 WARN rows in existing `verification_report.csv` (3,909 checks vs dps.psx.com.pk + sarmaaya): each → fix value from source page (one-at-a-time protocol) or document as source discrepancy. Target: ≥99% of checks EXACT-level, 0 unexplained MISMATCH.
- [ ] T2: Extend external checks to new fields after Phase 1 (dividends vs DPS site; more symbol-years) — add ≥500 fresh checks on the v3 file.
- [ ] T3: Extend `audit/data_appendix.md`: coverage tables (all 16 fields × years), unit policy, survivorship statement, exclusion log (each excluded row + evidence), bank policy, macro sources, VLM pipeline description + confidence-flag semantics.
- [ ] T4: Reproducibility: one script `scripts/rerun_pk_all.sh` = download → extract (using cached VLM state `vlm_state.json`) → build → QA; run end-to-end once; record runtime + output hashes.
- [ ] T5: Commit everything + git tag `pk-data-final-v3`; hash manifest (`sha256sums.txt`) into repo.
- [ ] T6: Re-run model battery e1–e8 × 3 seeds on final npz (existing `run_e1.py` + battery scripts); confirm results vs current (E3 SDF 0.93) — unchanged or better; update `notes/3country_data_status.md`.

**Exit gate:** verification ≥500 checks ≥99% EXACT; rerun produces identical/improved results → **STOP**.

---

## Phase 6 — Cross-Country Parity + Final Q1 Gate

**Files:** `data_ir/`, `data_tr/`, `paper/`, `audit/data_appendix.md`
- [ ] T1: IR: fill dividends to ≥90% via Rahavard DPS (pipeline exists in `build_ir_financials.py` — extend same logic as PK); re-run IR battery if financials change.
- [ ] T2: TR: fill dividends 2013–2015 (875 rows — İş Yatırım CF missing; source: KAP annual report PDFs or company sites; else document).
- [ ] T3: Unify bank policy across countries (same exclusion + justification, or same inclusion).
- [ ] T4: Same verification protocol (≥500 checks each country, ≥99% EXACT) — reuse Phase 5 scripts parameterized by country.
- [ ] T5: **Q1 gate checklist** (adversarial, numbers-first):
  - [ ] 100% of universe symbol-years have TA, Eq, Sales, PAT
  - [ ] All 13 real fields ≥95% (excl. documented structural gaps)
  - [ ] dividends ≥90% × 3 countries
  - [ ] 0 unexplained review rows; every exclusion has evidence
  - [ ] Characteristics ≥95% core cells × 3 countries
  - [ ] Macro full span, no trailing empties, sources dated
  - [ ] Verification ≥500 checks/country, ≥99% EXACT
  - [ ] Survivorship statement in appendix
  - [ ] One-command rerun reproduces all numbers (hash log)
  - [ ] Data appendix tables included in manuscript
- [ ] T6: Final commit + tag `data-q1-ready`; update manuscript data section.

**Exit:** checklist 10/10 → plan complete → **STOP**.

---

## Decisions Needed (before Phase 1 execution)

**✅ LOCKED 2026-08-08 (user: "strongest data we can achieve"):**

1. **Banks:** main sample = non-financials (already the case in `pk_build_characteristics.py`) + one-paragraph justification in appendix. **PLUS robustness check**: extract bank-appropriate fields (TA, TL, Eq, deposits, loans, PAT) for ~25 banks × years and re-run key models with banks included using only common fields (size, bm, mom, roe, vol, turnover, st_rev). Est. ~300 pages, ~$1.
2. **`field_factors`:** KEEP and USE — it's the per-field unit-override mechanism written by `pk_vlm_pipeline.py`; record all unit corrections there; document semantics in appendix.
3. **dividends:** canonical method = DPS (dps.psx.com.pk) × `shares_annual.csv`, mirroring IR. Existing 70 hand-extracted values = cross-check sample. Also revive `dy` (currently 1.3% filled) AND `ig` by re-scanning `lt_investments` (BS line) — target both alive at ≥90%.
4. **2013 financials:** probe 20 reports first (mix of caps); if extraction success ≥80% → full chase (~250 reports, ~$1–2); else document "sample starts 2014 (data availability)" + robustness check with 2015 start. OOS window untouched either way.
5. **VLM budget:** approved, ~$3–6 (mixed model strategy: qwen for easy pages, nex for L5/hard pages).

**Ground rule added:** nothing is dropped without evidence; every characteristic must be alive (≥90% cells) or explicitly documented as unavailable in the appendix.

## Estimated Effort
- Compute: 1–2 days (batches run in background with 5-min progress reports, per convention).
- VLM cost: ~$3–6.
- Human review: 899 review rows + 162 value defects + hole evidence — the one-row-at-a-time protocol makes this the bottleneck; estimate 3–5 focused sessions (can batch subagent fan-out for evidence-gathering, user confirms applications).
