#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Re-run specific validation years with the current pipeline code, then
recompute the validation gate from the stored rows (corrected thresholds:
trusted gt >= 10,000; index-table junk below)."""
import json
import sys
from pathlib import Path

ROOT = Path("/home/ubuntu/research/dlap-tse")
OUT = ROOT / "data_pk"
sys.path.insert(0, str(ROOT / "scripts"))
import pk_vlm_pipeline as P  # noqa: E402

YEARS = []
for sym in ("HUBC",):
    YEARS += [(sym, y) for y in range(2016, 2026)]
YEARS += [("ABOT", 2020)]

for sym, y in YEARS:
    pdf = P.E2E_DIR / sym / f"{y}.pdf"
    if not pdf.exists():
        print(f"{sym} {y}: PDF missing — skip")
        continue
    res = P.process_company_year(sym, y, pdf)
    if res is None:
        print(f"{sym} {y}: FAILED/None")
        continue
    row, _, _, flags = res
    P.ROWS_DIR.mkdir(exist_ok=True)
    (P.ROWS_DIR / f"{sym}_{y}.json").write_text(json.dumps(row, indent=1))
    P.mark(sym, y, "done", ";".join(x[0] for x in flags))
    print(f"{sym} {y}: fields={sum(1 for k in P.PANEL_HDR if row.get(k) not in (None, ''))} "
          f"flags={[(f[0], f[1]) for f in flags]}")
    print(f"    TA={row['TA']} TL={row['TL']} Eq={row['Eq']} CA={row['CA']} CL={row['CL']} "
          f"Sales={row['Sales']} Cash={row['Cash']} Inv={row['Inv']}")

# ---- gate recomputation from stored rows ----
print("\n=== GATE (rows vs p0_e2e, trusted |gt|>=10000) ===")
totals = {"pairs": 0, "match": 0, "trap_ok": 0, "trap_bad": 0, "flags": 0}
for sym in ["ABOT", "NML", "OGDC", "HUBC"]:
    gt_raw = json.loads((OUT / f"p0_e2e_{sym}_raw.json").read_text())
    gt = {int(k): v for k, v in gt_raw.items()}
    for y in sorted(gt):
        rf = OUT / "vlm_rows" / f"{sym}_{y}.json"
        if not rf.exists():
            continue
        row = json.loads(rf.read_text())
        for fld, key in (("TA", "total_assets"), ("TL", "total_liabilities"),
                         ("Eq", "total_equity"), ("CA", "current_assets"),
                         ("CL", "current_liabilities"), ("Sales", "revenue"),
                         ("GP", "gross_profit"), ("PAT", "net_income"),
                         ("Cash", "cash"), ("Inv", "inventory")):
            gv = gt.get(y, {}).get(key)
            vv = row.get(fld)
            if gv is None or vv is None:
                continue
            if abs(gv) >= 10000:
                totals["pairs"] += 1
                if P.approx(vv, gv):
                    totals["match"] += 1
                else:
                    print(f"  !! {sym} {y} {fld}: VLM={vv} GT={gv} MISMATCH")
            else:
                if abs(vv) >= 10000:
                    totals["trap_ok"] += 1
                else:
                    totals["trap_bad"] += 1
                    print(f"  !! {sym} {y} {fld}: VLM={vv} still trapped (GT={gv})")
        fds = row.get("flag_details") or []
        totals["flags"] += len(fds)
rate = 100 * totals["match"] / max(totals["pairs"], 1)
print(f"GATE: pairs={totals['pairs']} match={totals['match']} ({rate:.1f}%) "
      f"trap_ok={totals['trap_ok']} trap_bad={totals['trap_bad']} flags={totals['flags']}")
