#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PK-P0.3c: PDF gate test v3 — heading-first-occurrence extraction.

Find the FIRST standalone statement section (not consolidated) by locating
the "Statement of Financial Position" / "Balance Sheet" heading in the full
text, then extract the following lines and match fields. No TOC needed.
"""
import json
import re
import subprocess
from pathlib import Path

ROOT = Path("/home/ubuntu/research/dlap-tse")
OUT = ROOT / "data_pk"
TMP = Path("/tmp/pk_pdf_test")

SAMPLE = [
    ("ABOT", "PHARMA"), ("COLG", "CONSUMER"), ("DGKC", "CEMENT"),
    ("EFERT", "FERTILIZER"), ("HUBC", "POWER"), ("LUCK", "CEMENT"),
    ("MCB", "BANK"), ("NESTLE", "FOOD"), ("NML", "TEXTILE"),
    ("OGDC", "E&P"), ("PSO", "OIL"), ("HBL", "BANK"),
]

# heading markers for the standalone balance sheet (skip consolidated ones)
BS_HEADINGS = [
    r"Statement of Financial Position",
    r"Balance Sheet",
    r"Statement of Financial Position (?:as at|As at)",
]
PL_HEADINGS = [
    r"Statement of Profit or Loss",
    r"Statement of Profit and Loss",
    r"Profit or Loss Account",
    r"Profit and Loss Account",
]

FIELDS = [
    ("revenue", [r"^(?:Net )?(?:Sales|Revenue|Turnover)\b", r"^Sales \u2013 net",
                 r"^(?:Sales|Revenue) from (?:contracts )?with customers"]),
    ("cost_of_sales", [r"^Cost of (?:Goods |Sales )?Sales", r"^Cost of Revenue"]),
    ("gross_profit", [r"^Gross (?:Profit|Loss)"]),
    ("net_income", [r"^Profit (?:for the year|before tax)", r"^(?:Net )?Profit after",
                    r"^Loss (?:for the year|before tax)"]),
    ("total_assets", [r"^TOTAL ASSETS", r"^Total Assets\b", r"^Total assets\b",
                      r"^TOTAL ASSETS AND"]),
    ("total_equity", [r"^TOTAL EQUITY", r"^Total Equity\b", r"^Total equity and",
                      r"^TOTAL CAPITAL AND RESERVES", r"^Total equity attributable"]),
    ("total_liabilities", [r"^TOTAL LIABILITIES", r"^Total Liabilities\b", r"^Total liabilities\b"]),
    ("current_assets", [r"^TOTAL CURRENT ASSETS", r"^Total Current Assets", r"^Total current assets"]),
    ("current_liabilities", [r"^TOTAL CURRENT LIABILITIES", r"^Total Current Liabilities", r"^Total current liabilities"]),
    ("ppe", [r"^Property, Plant and Equipment\b", r"^Property and equipment\b"]),
    ("cash", [r"^Cash and (?:Cash )?Equivalents\b", r"^Cash and Bank"]),
    ("inventory", [r"^(?:Inventories|Stock-in-Trade)\b"]),
]


def pdf_text(path, first=None, last=None):
    cmd = ["pdftotext", "-layout"]
    if first:
        cmd += ["-f", str(first), "-l", str(last)]
    cmd += [str(path), "-"]
    r = subprocess.run(cmd, capture_output=True, timeout=180)
    return r.stdout.decode("utf-8", errors="ignore")


def find_section(lines, markers, skip_consolidated=True):
    """Return index of first heading line that is NOT consolidated."""
    for i, line in enumerate(lines):
        s = line.strip()
        for m in markers:
            if re.search(m, s, re.I):
                if skip_consolidated and re.search(r"consolidated", s, re.I) \
                        and not re.search(r"unconsolidated", s, re.I):
                    continue
                return i
    return None


def extract_fields(lines, start, n=400):
    found = {}
    for name, patterns in FIELDS:
        for line in lines[start:start + n]:
            s = line.strip()
            if not s or re.match(r"^[\d,\s.\-]+$", s):
                continue
            for pat in patterns:
                if re.match(pat, s, re.I):
                    nums = re.findall(r"[\d,]+\.?\d*", s)
                    if nums:
                        found[name] = nums[-1]
                        break
            if name in found:
                break
    return found


def process(code, sector):
    out = {"code": code, "sector": sector}
    pdf = TMP / f"{code}.pdf"
    if not pdf.exists():
        out["status"] = "NO_PDF"
        return out
    try:
        full = pdf_text(pdf)
        if len(full) < 2000:
            out["status"] = "NO_TEXT"
            return out
        lines = full.splitlines()
        bi = find_section(lines, BS_HEADINGS)
        if bi is None:
            out["status"] = "NO_BS_HEADING"
            return out
        fields = extract_fields(lines, bi)
        # income statement: find P&L heading after the balance sheet
        pli = find_section(lines[bi:], PL_HEADINGS)
        if pli is not None:
            pl_fields = extract_fields(lines, bi + pli)
            for k in ("revenue", "cost_of_sales", "gross_profit", "net_income"):
                if k in pl_fields and k not in fields:
                    fields[k] = pl_fields[k]
        out.update({"status": "OK", "size_mb": round(pdf.stat().st_size / 1e6, 1),
                    "fields": fields, "bs_line": bi})
    except Exception as e:
        out["status"] = f"ERR {type(e).__name__}: {str(e)[:80]}"
    return out


def main():
    results = [process(c, s) for c, s in SAMPLE]
    json.dump(results, open(OUT / "p0_pdf_gate_test_v3.json", "w"), indent=1)
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
            print(f"\nexample {r['code']}:", json.dumps(r["fields"], ensure_ascii=False))


if __name__ == "__main__":
    main()
