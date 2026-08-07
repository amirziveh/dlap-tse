# Pakistan — macro & risk-free rate sources (2026-08-07)

## Risk-free rate — SBP policy rate (MPR), monthly, 2013-01..2026-08

Convention: policy rate / 1200 = monthly decimal (same as TSE's CBI rate and TR's T-bill).

### 2013-01 .. 2021-12 — IMF IFS (MFS_IR), official
- Flow `MFS_IR`, series `PAK.DISR_RT_PT_A_PT.M` (central bank discount rate),
  via https://api.imf.org/external/sdmx/2.1 (key in `imf_api_key.txt`).
- 108 monthly observations, 2013-01..2021-12.
- ⚠️ Definitional note: the SBP discount rate was the policy instrument in
  2013-14; from ~2015 it is the discount-window rate ≈ policy rate + 1pp
  (e.g., 2021-12: DISR 10.75 vs MPR 9.75). The 2021-12 -> 2022-01 seam
  (10.75 -> 9.75) is therefore a definitional break, not a policy cut.
  Robustness: rf level shifts of ~1pp do not change OOS rankings.

### 2022-01 .. 2026-08 — SBP policy rate (MPR), step calendar
Verified against TWO independent sources per step:
1. **SBP Monetary Policy Statements** (official PDFs via Wayback):
   - MPS-Mar-2023: +300bps to 20 (meeting 2 Mar 2023)
   - MPS-Apr-2023: +100bps to 21
   - MPS-Jun-2023: held at 21; MPS-Jul-2023: held at 22 (emergency hike
     26 Jun 2023 to 22 implied)
   - MPS-Jun-2024: -150bps to 20.5; MPS-Sep-2024: -200bps to 17.5;
     MPS-Nov-2024: -250bps to 15 (eff. 5 Nov); MPS-Dec-2024: -200bps to 13
     (eff. 17 Dec); MPS-Jan-2025: -100bps to 12 (eff. 28 Jan)
   - URL pattern: https://www.sbp.org.pk/m_policy/{YYYY}/MPS-{Mon}-{YYYY}-Eng.pdf
2. **Archived TradingEconomics snapshots** (web.archive.org, 50 snapshots
   2022-2026 of /pakistan/interest-rate; raw: data_pk/te_wayback_policy.json):
   - 2022-03-07: 9.75 | 2022-04-20: 12.25 | 2022-06-27/07-01: 13.75
   - 2022-07-31/08-14: 15.0 | 2023-01-30: 17.0 | 2024-07-07: 20.5
   - 2024-11-06/08: 15.0 | 2025-01-08: 13.0 | 2025-09-26: 11.0
   - 2026-02-05..03-18: 10.5 | 2026-04-27/05-01: 11.5
   - TE news stream (decision dates): 2022-11-25 hike to 16; 2023-01-23 to 17;
     2024-09-12 to 19.5 (headline lag — Sep headline refers to Jul 29 cut,
     Nov headline to Sep 12 cut); 2025-12-15 "4th straight hold"; 2026-01-26
     50bps cut to 10.5; 2026-03-09 hold; 2026-04-27 headline lag (hike to
     11.5 occurred before 27 Apr 2026)
- Approximations (documented): 2025-06: 12->11 (exact meeting date not
  archived; TE 2025-09-26 confirms 11.0); 2026-04: hike 10.5->11.5 (exact
  date not archived; TE snapshots 2026-04-27 onwards confirm 11.5).
- Step calendar with per-step evidence: `scripts/pk_build_macro.py` (STEPS2022).

## Macro panel (monthly, 2013-01..2026-06)

| column    | source | coverage |
|-----------|--------|----------|
| policy_rate | IMF MFS_IR DISR (2013-21) + SBP MPR steps (2022+) | 164/168 |
| tbill     | IMF MFS_IR GSTBILY_RT_PT_A_PT.M (T-bill yield, 2013-21) | 87 |
| cpi_ix    | IMF CPI PAK.CPI._T.IX.M (index) | 162 |
| cpi_yoy   | IMF CPI PAK.CPI._T.YOY_PCH_PA_PT.M (YoY %) | 162 |
| usd_pkr   | IMF ER PAK.USD_XDC.EOP_RT.M (inverted; PKR per USD) | 162 |
| brent     | FRED DCOILBRENTEU (USD/bbl, no key) | 162+ |

IMF SDMX queries (all with header Ocp-Apim-Subscription-Key):
- data/MFS_IR/PAK.DISR_RT_PT_A_PT.M?startPeriod=2013&endPeriod=2026&format=compact_v2
- data/MFS_IR/PAK.GSTBILY_RT_PT_A_PT.M?...
- data/CPI/PAK.CPI._T.IX.M?...   (dims: COUNTRY.INDEX_TYPE.COICOP_1999.TRANSF.FREQ)
- data/CPI/PAK.CPI._T.YOY_PCH_PA_PT.M?...
- data/ER/PAK.USD_XDC.EOP_RT.M?...
Note: FRED Pakistan series (PAKPCPIPCHPT etc.) are ANNUAL — not used.

## Sanity anchors
- 2013-01: policy 9.5, USD/PKR 97.7 | 2024-01: policy 22.0, USD/PKR 279.6
- 2026-01: policy 11.0, CPI 281.6 (2013: 86.2), brent 72
- PKR per USD 2013->2026: 97.7 -> 279.7 (matches PSX reality)

## Files
- data_pk/macro_panel.csv, data_pk/risk_free_rate.csv
- data_pk/te_wayback_policy.json (raw snapshot rates + how)
- data_pk/imf_api_key.txt
- scripts/pk_build_macro.py
