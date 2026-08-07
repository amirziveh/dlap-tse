# Data Appendix — Pakistan Financial Statement Panel (PK-FIN v3.1)

*For reviewers: this document describes the construction, verification, and known
limitations of the annual financial-statement panel used in this paper. Every claim
below is backed by machine-readable artifacts in `data_pk/audit/` and
`data_pk/unit_norm/` (file names given inline).*

---

## 1. Sample and Sources

- **Universe**: all companies listed on the Pakistan Stock Exchange (PSX) with
  annual reports filed on the official portal (financials.psx.com.pk, maintained by
  the exchange and the SECP-compliant IME system). 4,946 company-years, 462 companies,
  fiscal years 2013–2025.
- **Primary source**: the **audited annual report PDFs** (balance sheet, profit &
  loss statement, cash flow, auditor's opinion) as filed by each company — i.e.,
  as-reported, unstandardized data.
- **Financial-sector exclusion**: commercial banks, insurers, investment
  banks/companies, leasing, modarabas, and microfinance are excluded from the main
  panel (176 bank-years + 993 financial-years), leaving **3,351 non-financial
  company-years** (`scripts/build_pk_financials.py`).
- **Sample screen**: rows failing the balance-sheet identity after repair (below)
  are excluded to a review directory rather than silently kept.

## 2. Extraction Pipeline

1. **PDF acquisition** from the official portal (all filings archived under their
   exchange-assigned IDs; `data_pk/pk_manifest.json`).
2. **Parser** (`scripts/pk_vlm_pipeline.py`): a deterministic financial-statement
   parser (regex/table logic over text layers) extracts the audited statements and
   produces an *evidence value* (`ev`) per field — used for cross-checks and repairs.
3. **Vision-language extraction**: for scanned reports (no text layer) or when the
   parser fails, a vision model reads the rendered statement pages. **Orientation
   handling**: pages that are rotated 90° are detected and rotated with PIL before
   any read; every vision-read page is checked for upright orientation
   (`data_pk/audit/review_summary.csv`, `rotated` / `rotation_fixed` columns).
4. **QA layers** (`run_qa` in the pipeline): L1 = balance-sheet identity
   (TA = TL + Eq), L2 = prior-year overlap, L3 = year-over-year plausibility,
   L5 = parser-vs-VLM consistency. All flags and their resolutions are logged in
   `data_pk/qa_report.csv`.
5. **L1 repair pass** (`repair_l1`): where the identity fails, the parser's audited
   `ev` value is substituted if it restores the identity; repairs are logged with
   before/after values. Rows still failing L1 (102) are excluded to
   `data_pk/review/` with evidence and are not used in the panel.

## 3. Unit Normalization (the central data issue)

Pakistani reports present figures in **full rupees**, **thousands**, or (rarely)
**millions**, sometimes inconsistently **within a single company across years** and
even **within a single statement**. This is the main source of scale errors in
hand-collected panels. We normalize every monetary field to **thousands of PKR**.

Method (see `references/pakistan-psx-unit-normalization-2026-08.md` and
`data_pk/unit_factors.json` for per-row provenance):

1. **External anchors** (in priority order):
   - **PSX official portal (dps.psx.com.pk)** — as-reported Sales/PAT/EPS,
     2022–2025, stated in thousands; coverage 462/462 companies.
   - **Sarmaaya/FactSet API** (beta-restapi.sarmaaya.pk) — full balance
     sheet/income statement for 2025 (284 companies with real ISINs); consolidated
     basis (see §6), reported in millions.
   - **Market-cap anchor**: June average price × shares outstanding
     (from the portal's equity section) vs. book equity — a 1–500 window identifies
     thousands vs. full-rupees for years without other anchors.
2. **Year-over-year propagation**: once a year is anchored, adjacent years are
   verified/assigned via continuity (growth rates must be economically plausible;
   flagged otherwise).
3. **Outcome** (main panel): 1,824 rows already in thousands (factor 1.0);
   **1,436 rows (43%) were in full rupees and divided by 1,000**; 7 rows were in
   millions and multiplied by 1,000. Without this correction, 43% of the panel
   would have been off by a factor of 1,000.
4. **Mixed-unit fields**: where one field (e.g., PAT) was read from a page
   presented in a different unit than the rest of the row, field-level corrections
   were applied from the audited statement (9 cases; `unit_factors.json`,
   `field_factors`).
5. **Cross-statement consistency**: the parser's `ev` values are scaled with the
   row so the L1 repair operates in normalized space (verified: no row lost to
   repair malfunction; every repaired row re-passes L1).

## 4. External Verification

- **3,896 field-level checks** against external sources
  (`data_pk/verification_report.csv`):
  - 3,312 EXACT (85%) — identical to the source to the reported digit;
  - +211 tight, +171 loose → **94.8% of checks match** within 5% / 10%;
  - 107 residual mismatches: 95 are **documented definitional differences**
    (consolidated vs. standalone for sarmaaya, minority-interest attribution for
    PAT), 12 are flagged rows in the review list (below).
- **Sales vs. PSX portal (as-reported)**: 920/994 EXACT (93%) for 2022–2025.
- **Aggregator year-shift errors — three documented cases**: the PSX portal's
  aggregator (Capital Stake) occasionally mislabels the comparative column as the
  current year: KAPCO 2023 (portal "Sales" 136,599,624 = audited 2022),
  HTL 2024/2025 PAT (portal values = audited 2022/2024), AEL 2022 (portal "Sales"
  2,927 = audited 2021 column). In all three cases the audited statement was
  retained and the aggregator error documented. This is why every portal mismatch
  is re-verified against the audited statement rather than auto-accepted.
- **Independent re-review of unresolved rows**: 185 company-years whose unit could
  not be anchored automatically were re-extracted from the audited PDFs by parallel
  review agents (8 + 5 workers, one report per row:
  `data_pk/audit/subagent_reports/`, `review_summary.csv`). Each report records:
  PDF and page numbers, unit header text, rotation status, L1 and GP arithmetic
  checks, and confidence. 174/185 rows completed; of these, **0 fail L1 and 0 fail
  the GP identity**; every 2022+ row (16/17 fields) matches the PSX portal exactly
  (1 miss detected and corrected: AEL 2022 Sales, `audit/dps_crosscheck.csv`).
- **Spot verification against audited statements**: a random subset (EXIDE 2016,
  ADAMS 2019, AEL 2013–2015, HTL 2022/2024/2025, JSML 2025, KAPCO 2023) was
  re-verified manually against the signed financial statements, including
  **rotated pages** — all values match to the rupee; where the external aggregator
  disagreed with the audited statement (KAPCO 2023, HTL 2024/25 PAT), the audited
  statement was retained and the aggregator error documented.

## 5. Error Correction Process

All corrections are **logged with provenance** (`src` per field in `vlm_rows/`,
`qa_report.csv`, `verification_report.csv`):

- 31 L1 repairs from parser evidence;
- 168 PAT/Sales substitutions from the as-reported PSX portal (2022–2025);
- 82 PAT substitutions for definitional consistency (1–3× differences,
  attributable vs. total profit) — one definition across the panel;
- 12 fields filled/verified per audited statements in joint human review
  (EXIDE 2016, ADAMS 2019, AEL 2013–2015);
- sign corrections (e.g., PSEL 2023 PAT − → +);
- 4 rows recovered after a normalization bug (MCBAH 2013, OLPL 2022, SAPT 2014,
  SHCM 2018) — regression-tested: the panel row count is identical after
  normalization.

## 6. Known Limitations (stated openly)

1. **External coverage is recent-year only**: the PSX portal covers 2022–2025 and
   Sarmaaya/FactSet (free tier) 2025. Pre-2022 years rest on audited-report
   extraction + unit propagation + market-cap anchors; where an anchor was
   impossible, the row is flagged (84 rows in the review list) rather than guessed.
2. **Consolidated vs. standalone**: Sarmaaya/FactSet data are consolidated; our
   panel is as-reported standalone. Ratios between them can exceed the tight match
   window (up to ~4.5×, e.g., HUBC); these are documented, not errors.
3. **Definitional PAT differences** (attributable vs. total, discontinued ops):
   flagged rows are documented in `data_pk/value_errors.csv` /
   `pass2_review_list.csv` (156 unique rows, 112 in the main CSV, mostly 1–3× PAT
   definitional or consolidation differences).
4. **Audited-statement caveats carried over**: 2013 AEL qualified audit opinion;
  going-concern notes (AEL 2014–2015, KASBM 2022); liquidator's statements
   (BPBL 2021); name changes (Ideal Energy → Arshad Energy → AEL Textiles;
   symbol reuse) — all noted per row in `subagent_reports/`.
5. **Symbol reuse / name changes**: e.g., AEL changed names twice; the symbol's
   price history and fundamentals are matched by the symbol's filings, and the
   entity change is documented rather than silently merged.
6. **11/185 review rows remain in progress** (scanned-only reports, e.g.,
   AGP 2021/2023/2024, ARPAK 2024, AWWAL 2018): their PDFs are staged and they will
   be completed with the same documented workflow before the final dataset freeze.

## 7. Reproducibility

- Pipeline: `scripts/pk_vlm_pipeline.py` (finalize/L1/repair/QA, deterministic);
  `scripts/build_pk_financials.py` (panel assembly); `/tmp/pk_*.py` review scripts
  (kept in repo `scripts/review_archive/`).
- Inputs: `data_pk/pk_manifest.json` (filing IDs), raw price files
  `data_pk/raw/*.txt`, external-source dumps `data_pk/unit_norm/`
  (pk_dps_financials.json, pk_sarmaaya_fundamentals.json).
- Artifacts: `data_pk/financials_annual.csv` (full L1-passing panel, 4,844 rows),
  `data_pk/unit_norm/combined_financials_v3.csv` (final merged panel),
  `unit_factors.json` (per-row unit provenance), `verification_report.csv`,
  `qa_report.csv`, `audit/` (this appendix + subagent evidence).
- Raw backup of pre-normalization extraction: `/tmp/vlm_rows_raw_backup/` (4,946
  files) — the entire normalization and correction chain can be replayed from it.

*Prepared August 2026. All numbers in this appendix are computed by deterministic
scripts from the named artifacts; re-running them reproduces the panel.*
