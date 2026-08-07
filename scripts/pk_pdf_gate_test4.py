#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PK-P0.3d: PDF gate test v4 — pdfplumber table extraction.

Find the standalone balance sheet + income statement pages, extract tables
with pdfplumber (label + value columns), match fields from table rows.
"""
import json
import re
import subprocess
from pathlib import Path

import pdfplumber

ROOT = Path("/home/ubuntu/research/dlap-tse")
OUT = ROOT / "data_pk"
TMP = Path("/tmp/pk_pdf_test")

SAMPLE = [
    ("ABOT", "PHARMA"), ("COLG", "CONSUMER"), ("DGKC", "CEMENT"),
    ("EFERT", "FERTILIZER"), ("HUBC", "POWER"), ("LUCK", "CEMENT"),
    ("MCB", "BANK"), ("NESTLE", "FOOD"), ("NML", "TEXTILE"),
    ("OGDC", "E&P"), ("PSO", "OIL"), ("HBL", "BANK"),
]

BS_PAT = re.compile(r"Statement of Financial Position|Balance Sheet", re.I)
PL_PAT = re.compile(r"Statement of Profit or Loss|Statement of Profit and Loss|Profit or Loss Account|Profit and Loss Account", re.I)

FIELDS = [
    ("revenue", [r"^(?:Net )?(?:Sales|Revenue|Turnover)\b", r"^Sales \u2013 net"]),
    ("cost_of_sales", [r"^Cost of (?:Goods |Sales )?Sales", r"^Cost of Revenue"]),
    ("gross_profit", [r"^Gross (?:Profit|Loss)"]),
    ("net_income", [r"^Profit (?:for the year|before tax)", r"^(?:Net )?Profit after", r"^Loss (?:for the year|before tax)"]),
    ("total_assets", [r"^TOTAL ASSETS", r"^Total Assets\b", r"^TOTAL EQUITY AND LIABILITIES"]),
    ("total_equity", [r"^TOTAL EQUITY", r"^Total Equity\b", r"^Total equity and", r"^TOTAL CAPITAL AND RESERVES"]),
    ("total_liabilities", [r"^TOTAL LIABILITIES", r"^Total Liabilities\b"]),
    ("current_assets", [r"^TOTAL CURRENT ASSETS", r"^Total Current Assets", r"^Total current assets"]),
    ("current_liabilities", [r"^TOTAL CURRENT LIABILITIES", r"^Total Current Liabilities", r"^Total current liabilities"]),
    ("ppe", [r"^Property, Plant and Equipment\b", r"^Property and equipment\b"]),
    ("cash", [r"^Cash and (?:Cash )?Equivalents\b", r"^Cash and Bank"]),
    ("inventory", [r"^(?:Inventories|Stock-in-Trade)\b"]),
]


def pdf_text_pages(path, first=None, last=None):
    cmd = ["pdftotext", "-layout"]
    if first:
        cmd += ["-f", str(first), "-l", str(last)]
    cmd += [str(path), "-"]
    r = subprocess.run(cmd, capture_output=True, timeout=180)
    return r.stdout.decode("utf-8", errors="ignore")


def find_page(pdf, pat):
    """Return 1-based page number of the first page containing a match."""
    for i, page in enumerate(pdf.pages, 1):
        text = page.extract_text() or ""
        if pat.search(text) and not re.search(r"consolidated", text, re.I):
            return i
    return None


def clean_num(s):
    s = s.replace(",", "").replace(" ", "")
    try:
        return float(s)
    except ValueError:
        return None


def row_value(cells):
    """Best numeric candidate from a table row: prefer last cell with a
    plausible magnitude; skip ratio-looking values."""
    nums = [clean_num(c) for c in cells if clean_num(c) is not None]
    if not nums:
        return None
    big = [n for n in nums if abs(n) >= 1000]
    return (big or nums)[-1]


def process(code, sector):
    out = {"code": code, "sector": sector}
    pdf_path = TMP / f"{code}.pdf"
    if not pdf_path.exists():
        out["status"] = "NO_PDF"
        return out
    try:
        with pdfplumber.open(pdf_path) as pdf:
            bs_pg = find_page(pdf, BS_PAT)
            if bs_pg is None:
                out["status"] = "NO_BS_HEADING"
                return out
            fields = {}
            # scan pages bs_pg .. bs_pg+3 for tables
            for pg in pdf.pages[bs_pg - 1:bs_pg + 3]:
                for table in pg.extract_tables():
                    for row in table:
                        cells = [c.strip() if c else "" for c in row]
                        label = cells[0]
                        if not label or len(label) > 70:
                            continue
                        for name, pats in FIELDS:
                            if name in fields:
                                continue
                            for p in pats:
                                if re.match(p, label, re.I):
                                    v = row_value(cells[1:])
                                    if v is not None:
                                        fields[name] = v
                                    break
                    if len(fields) >= 10:
                        break
                if len(fields) >= 10:
                    break
            # income statement: P&L heading page
            pl_pg = find_page(pdf, PL_PAT)
            if pl_pg is not None and pl_pg != bs_pg:
                for pg in pdf.pages[pl_pg - 1:pl_pg + 2]:
                    for table in pg.extract_tables():
                        for row in table:
                            cells = [c.strip() if c else "" for c in row]
                            label = cells[0]
                            if not label or len(label) > 70:
                                continue
                            for name in ("revenue", "cost_of_sales",
                                         "gross_profit", "net_income"):
                                if name in fields:
                                    continue
                                for p in FIELDS[[f[0] for f in FIELDS].index(name)][1]:
                                    if re.match(p, label, re.I):
                                        v = row_value(cells[1:])
                                        if v is not None:
                                            fields[name] = v
                                        break
                    if len(fields) >= 12:
                        break
        out.update({"status": "OK", "size_mb": round(pdf_path.stat().st_size / 1e6, 1),
                    "fields": {k: round(v, 2) for k, v in fields.items()}})
    except Exception as e:
        out["status"] = f"ERR {type(e).__name__}: {str(e)[:90]}"
    return out


def main():
    results = [process(c, s) for c, s in SAMPLE]
    json.dump(results, open(OUT / "p0_pdf_gate_test_v4.json", "w"), indent=1)
    for r in sorted(results, key=lambda x: x["code"]):
        n = len(r.get("fields", {}))
        print(f"{r['code']:6s} {r['sector']:12s} {r['status']:16s} fields={n} "
              f"size={r.get('size_mb', '-')}MB")
    ok = [r for r in results if r["status"] == "OK"]
    print(f"\nOK: {len(ok)}/{len(results)}")
    for f in FIELDS:
        cnt = sum(1 for r in ok if f[0] in r.get("fields", {}))
        print(f"  {f[0]:22s} {cnt}/{len(ok)}")
    for r in ok:
        if r.get("fields"):
            print(f"\nexample {r['code']}:", json.dumps(r["fields"]))


if __name__ == "__main__":
    main()
