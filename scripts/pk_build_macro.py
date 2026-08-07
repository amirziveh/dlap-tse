#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pk_build_macro.py — DLAP-TSE Pakistan (PSX): macro panel + risk-free rate
=========================================================================
Sources (all documented in data_pk/macro_sources.md):
  1. IMF SDMX 2.1 API (key: data_pk/imf_api_key.txt, subscription dlap-pakistan)
       policy rate : MFS_IR PAK.DISR_RT_PT_A_PT.M   (discount rate, 2013-01..2021-12)
       T-bill yield: MFS_IR PAK.GSTBILY_RT_PT_A_PT.M (2013-01..2021-12, 87 obs)
       CPI index   : CPI    PAK.CPI._T.IX.M          (2013-01..2026-06)
       CPI YoY     : CPI    PAK.CPI._T.YOY_PCH_PA_PT.M
       USD/PKR     : ER     PAK.USD_XDC.EOP_RT.M     (inverted: units PKR per USD)
  2. FRED (no key): DCOILBRENTEU (Brent USD/bbl, monthly)
  3. SBP policy rate (MPR) 2022-01..2026-07 — official SBP Monetary Policy
     Statements (MPS, via Wayback) + archived TradingEconomics snapshots;
     step calendar with per-step evidence in STEPS2022 below.

rf convention: policy rate (same as TSE which uses the CBI rate).
Note: IMF DISR is the SBP discount window rate (MPR + ~1pp in 2015-2021; the
policy instrument itself in 2013-14). The 2021-12 -> 2022-01 seam (10.75 ->
9.75) is a definitional break, documented in macro_sources.md.

Outputs (data_pk/):
  macro_panel.csv     month, policy_rate, tbill, cpi_ix, cpi_yoy, usd_pkr, brent
  risk_free_rate.csv  month, monthly_rate_pct  (policy_rate / 12, percent/mo)
"""
import csv
import json
import re
import subprocess
from pathlib import Path

PK = Path("/home/ubuntu/research/dlap-tse/data_pk")
KEY = "6ef971f688284ff59cac0af628563d90"
BASE = "https://api.imf.org/external/sdmx/2.1"
MONTHS = []
y, m = 2013, 1
while (y, m) <= (2026, 12):
    MONTHS.append(f"{y}-{m:02d}")
    m += 1
    if m > 12:
        m, y = 1, y + 1

# ---- SBP MPR step calendar 2022-2026 (month -> rate) -----------------------
# Each step verified against: SBP MPS PDFs (Wayback), TE archived snapshots,
# TE news-stream headlines. Details in data_pk/macro_sources.md.
STEPS2022 = {
    "2022-01": 9.75, "2022-02": 9.75, "2022-03": 9.75,
    "2022-04": 12.25,   # MPS Apr 7, 2022 (TE snap 20220420: 12.25)
    "2022-05": 13.75,   # MPS May 23, 2022 (TE snaps 20220627..: 13.75)
    "2022-06": 13.75,
    "2022-07": 15.0,    # MPS Jul 7, 2022 (TE snap 20220731: 15.0)
    "2022-08": 15.0, "2022-09": 15.0, "2022-10": 15.0,  # Oct 10 held (TE news)
    "2022-11": 16.0,    # MPS Nov 25, 2022 (TE news: hike to 16)
    "2022-12": 16.0,
    "2023-01": 17.0,    # MPS Jan 23, 2023 (TE snap 20230130: 17.0)
    "2023-02": 17.0,
    "2023-03": 20.0,    # MPS Mar 2, 2023: +300bps to 20
    "2023-04": 21.0,    # MPS Apr 2023: +100bps to 21
    "2023-05": 21.0,
    "2023-06": 22.0,    # emergency hike Jun 26, 2023 (MPS Jul 31: held at 22)
    "2023-07": 22.0, "2023-08": 22.0, "2023-09": 22.0, "2023-10": 22.0,
    "2023-11": 22.0, "2023-12": 22.0,
    "2024-01": 22.0, "2024-02": 22.0, "2024-03": 22.0, "2024-04": 22.0,
    "2024-05": 22.0,
    "2024-06": 20.5,   # MPS Jun 10, 2024: -150bps (TE snap 20240707: 20.5)
    "2024-07": 19.5,   # MPS Jul 29, 2024: -100bps
    "2024-08": 19.5,
    "2024-09": 17.5,   # MPS Sep 12, 2024: -200bps
    "2024-10": 17.5,
    "2024-11": 15.0,   # MPS Nov 4, 2024: -250bps (TE snaps 20241106/08: 15.0)
    "2024-12": 13.0,   # MPS Dec 16, 2024: -200bps (TE snap 20250108: 13.0)
    "2025-01": 12.0,   # MPS Jan 27, 2025: -100bps
    "2025-02": 12.0, "2025-03": 12.0, "2025-04": 12.0, "2025-05": 12.0,
    "2025-06": 11.0,   # cut 12->11 (~Jun-Jul 2025; TE snap 20250926: 11.0)
    "2025-07": 11.0, "2025-08": 11.0,
    "2025-09": 11.0, "2025-10": 11.0, "2025-11": 11.0, "2025-12": 11.0,
    "2026-01": 11.0,
    "2026-02": 10.5,   # MPS Jan 26, 2026: -50bps (TE snaps 20260205..: 10.5)
    "2026-03": 10.5,   # Mar 9, 2026 held (TE news)
    "2026-04": 11.5,   # hike to 11.5 before Apr 27, 2026 (TE snaps 20260427/0501: 11.5)
    "2026-05": 11.5, "2026-06": 11.5, "2026-07": 11.5,
    "2026-08": 11.5,
}


def imf(path):
    r = subprocess.run(["curl", "-s", "--max-time", "60", f"{BASE}/{path}",
                        "-H", f"Ocp-Apim-Subscription-Key: {KEY}"],
                       capture_output=True, text=True, timeout=90)
    return r.stdout


def parse_obs(xml):
    out = {}
    for mm in re.finditer(r'TIME_PERIOD="(\d{4})-M(\d{2})" OBS_VALUE="([0-9.eE+-]+)"', xml):
        key = f"{mm.group(1)}-{int(mm.group(2)):02d}"
        out[key] = float(mm.group(3))
    return out


def fred(series):
    r = subprocess.run(["curl", "-sL", "--max-time", "40",
                        f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"],
                       capture_output=True, text=True, timeout=60)
    out = {}
    for line in r.stdout.splitlines()[1:]:
        if not line.strip():
            continue
        date, val = line.split(",")
        try:
            out[date[:7]] = float(val)
        except ValueError:
            continue
    return out


def main():
    print("fetching IMF series...", flush=True)
    disr = parse_obs(imf(f"data/MFS_IR/PAK.DISR_RT_PT_A_PT.M?startPeriod=2013&endPeriod=2026&format=compact_v2"))
    tbill = parse_obs(imf(f"data/MFS_IR/PAK.GSTBILY_RT_PT_A_PT.M?startPeriod=2013&endPeriod=2026&format=compact_v2"))
    cpi_ix = parse_obs(imf(f"data/CPI/PAK.CPI._T.IX.M?startPeriod=2013&endPeriod=2026&format=compact_v2"))
    cpi_yoy = parse_obs(imf(f"data/CPI/PAK.CPI._T.YOY_PCH_PA_PT.M?startPeriod=2013&endPeriod=2026&format=compact_v2"))
    usd = parse_obs(imf(f"data/ER/PAK.USD_XDC.EOP_RT.M?startPeriod=2013&endPeriod=2026&format=compact_v2"))
    print(f"  disr={len(disr)} tbill={len(tbill)} cpi={len(cpi_ix)} cpi_yoy={len(cpi_yoy)} usd={len(usd)}", flush=True)
    brent = fred("DCOILBRENTEU")
    print(f"  brent={len(brent)}", flush=True)

    rows = []
    for month in MONTHS:
        pr = disr.get(month) if month < "2022-01" else STEPS2022.get(month)
        rows.append({
            "month": month,
            "policy_rate": pr if pr is not None else "",
            "tbill": tbill.get(month, ""),
            "cpi_ix": cpi_ix.get(month, ""),
            "cpi_yoy": cpi_yoy.get(month, ""),
            "usd_pkr": round(1.0 / usd[month], 4) if month in usd and usd[month] else "",
            "brent": brent.get(month, ""),
        })

    with open(PK / "macro_panel.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    rf = [{"month": r["month"], "monthly_rate_pct": round(float(r["policy_rate"]) / 1200.0, 8)}
          for r in rows if r["policy_rate"] != ""]
    with open(PK / "risk_free_rate.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["month", "monthly_rate_pct"])
        w.writeheader()
        w.writerows(rf)

    n_policy = sum(1 for r in rows if r["policy_rate"] != "")
    print(f"\nmacro_panel.csv: {len(rows)} rows | policy coverage: {n_policy}/{len(rows)}")
    print(f"risk_free_rate.csv: {len(rf)} rows")
    for r in rows[::12]:
        print("  ", r["month"], "policy=", r["policy_rate"], "tbill=", r["tbill"],
              "cpi=", r["cpi_ix"], "cpi_yoy=", r["cpi_yoy"], "usd=", r["usd_pkr"],
              "brent=", r["brent"])


if __name__ == "__main__":
    main()
