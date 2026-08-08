# DLAP-TSE — Pakistan (PSX) Data Package
Created: 2026-08-08 | Source: /home/ubuntu/research/dlap-tse/data_pk/

## 1. FINANCIALS (annual, 16-field schema)
- financials_annual.csv — MAIN file: 4,836 rows | 460 symbols | years 2013–2026
  Fields: symbol, year, TA (total assets), TL (total liabilities), Eq (equity),
  CA (current assets), CL (current liabilities), Sales, COGS, GP (gross profit),
  PAT (profit after tax), Cash, Inv (inventory), PPE, lt_investments, dividends,
  dividends_paid, cfo, units, unit_factor, field_factors, pages, flags, status
  Coverage (of 4,836 rows): TA 88.0% | CA 81.5% | CL 80.4% | Cash 79.1% | PAT 75.2%
  | Eq 67.9% | Sales 66.7% | TL 66.0% | COGS 65.0% | GP 62.4% | Inv 57.7% | cfo 49.0%
  | PPE 42.2% | dividends_paid 27.8% | dividends 1.4% | lt_investments 0.3%
  Quality flags: status=ok 3,937 rows | status=review 899 rows (flags: L2/L3/L5 =
  confidence levels of VLM extraction; empty flag = clean)
- financials_annual_no_banks.csv — 4,663 rows (banks removed)
- financials_annual_no_financials.csv — 3,782 rows (all financial sector removed)
- financials_annual_with_empty.csv — includes rows where extraction produced no values
- vlm_rows_all.csv — 2,567 per-row VLM extractions (raw extraction layer, one row per
  financial-statement page processed)
- vlm_rows/ — 4,945 raw per-row JSON extractions (per symbol-year-statement)
- review/ — 2,429 review JSONs (disputed/low-confidence rows with reference values)
- unit_factors.json + unit_norm/ — unit normalization factors + normalized values
- pk_manifest.json — per-symbol manifest (PDFs, pages, extraction status)
- symbols.json — symbol universe

## 2. PANELS (model inputs)
- characteristics_panel.csv — 47,518 rows (raw characteristics)
- characteristics_z.csv — cross-sectionally z-scored characteristics
- market_cap_monthly.csv — 61,187 rows
- shares_annual.csv — 6,328 rows
- processed/monthly_returns.csv — monthly returns
- processed/volume_monthly.csv — monthly volume
- processed/coverage_summary.csv — coverage per symbol
- macro_panel.csv + macro_sources.md — macro series
- risk_free_rate.csv — PK risk-free rate
- Char_all.npz / Macro_all.npz / meta.json — final model input panels (winsorized 1%/99%)

## 3. QA / AUDIT
- qa_report.csv — VLM-vs-parser conflicts (899 review rows)
- verification_report.csv — external cross-checks (vs dps.psx.com.pk values; EXACT/ratio levels)
- data_gaps_report.csv — holes & missing fields per symbol-year
- excluded_rows.csv — rows dropped during cleaning
- value_errors_review.csv — unit/scale error candidates
- pass2_review_list.csv — second-pass review list
- audit/data_appendix.md — data appendix writeup (for paper)
- audit/review_summary.csv — summary of all reviews
- audit/dps_crosscheck.csv — dividends-per-share cross-check vs PSX
- audit/subagent_manifest.json + subagent_reports/ — 174 subagent QA reports

## 4. NOTES
- PK_REVIEW_TODO_2026-08-05.md — remaining review todo
- ../../HANDOFF_FINANCIALS_2026-08-04.md — 3-country financials handoff
- ../../notes/3country_data_status.md — latest 3-country status (PK experiments done:
  E3 SDF Sharpe 0.93 = best model for PK)
