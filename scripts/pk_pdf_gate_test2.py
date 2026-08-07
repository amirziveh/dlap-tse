#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PK-P0.3b: PDF gate test v2 — TOC-guided page extraction.

Annual reports have a table of contents with page numbers for the
statements. Parse the TOC, extract ONLY those pages, match fields there.
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

# section heading -> page number in TOC
TOC_MARKERS = [
    ("sfp", r"Statement of Financial Position"),
    ("pl", r"Statement of Profit or Loss"),
]

FIELDS = [
    ("revenue", [r"^(?:Net )?(?:Sales|Revenue|Turnover)\b", r"^Sales \u2013 net",
                 r"^(?:Sales|Revenue) (?:from )?(?:contracts )?with customers"]),
    ("cost_of_sales", [r"^Cost of (?:Goods |Sales )?Sales", r"^Cost of Revenue"]),
    ("gross_profit", [r"^Gross (?:Profit|Loss)"]),
    ("net_income", [r"^Profit (?:for the year|before tax)", r"^(?:Net )?Profit after",
                    r"^Loss (?:for the year|before tax)"]),
    ("total_assets", [r"^TOTAL ASSETS", r"^Total Assets\b", r"^Total assets\b"]),
    ("total_equity", [r"^TOTAL EQUITY", r"^Total Equity\b", r"^Total equity and",
                      r"^TOTAL CAPITAL AND RESERVES", r"^Total equity attributable"]),
    ("total_liabilities", [r"^TOTAL LIABILITIES", r"^Total Liabilities\b", r"^Total liabilities\b"]),
    ("current_assets", [r"^TOTAL CURRENT ASSETS", r"^Total Current Assets", r"^Current Assets\b"]),
    ("current_liabilities", [r"^TOTAL CURRENT LIABILITIES", r"^Total Current Liabilities", r"^Current Liabilities\b"]),
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


def toc_page_numbers(full_text):
    """Find report page numbers for statement sections via the TOC."""
    pages = {}
    for name, marker in TOC_MARKERS:
        # TOC lines look like: "Statement of Financial Position ..... 44"
        for line in full_text.splitlines():
            if marker in line:
                m = re.search(r"(\d{1,3})\s*$", line.strip())
                if m:
                    pages[name] = int(m.group(1))
                    break
    return pages


def extract_fields(pages_text):
    found = {}
    lines = pages_text.splitlines()
    for name, patterns in FIELDS:
        for line in lines:
            s = line.strip()
            if not s or re.match(r"^[\d,\s.\-]+$", s):
                continue
            for pat in patterns:
                if re.match(pat, s, re.I):
                    nums = re.findall(r"[\d,]+\.?\d*", s)
                    if nums:
                        # last number = current-year value (layout: label note cur prev)
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
        toc = toc_page_numbers(full)
        out["toc"] = toc
        if "sfp" not in toc:
            out["status"] = "NO_TOC_SFP"
            return out
        # statements: SFP page .. SFP+8 (P&L follows; offset ~6 pdf pages from
        # report page due to front matter)
        start = max(1, toc["sfp"] - 1)
        pages_text = pdf_text(pdf, start, toc["sfp"] + 10)
        fields = extract_fields(pages_text)
        out.update({"status": "OK", "size_mb": round(pdf.stat().st_size / 1e6, 1),
                    "fields": fields})
    except Exception as e:
        out["status"] = f"ERR {type(e).__name__}: {str(e)[:80]}"
    return out


def main():
    results = [process(c, s) for c, s in SAMPLE]
    json.dump(results, open(OUT / "p0_pdf_gate_test_v2.json", "w"), indent=1)
    for r in sorted(results, key=lambda x: x["code"]):
        n = len(r.get("fields", {}))
        print(f"{r['code']:6s} {r['sector']:12s} {r['status']:14s} "
              f"fields={n} toc={r.get('toc', '-')} size={r.get('size_mb', '-')}MB")
    ok = [r for r in results if r["status"] == "OK"]
    print(f"\nOK: {len(ok)}/{len(results)}")
    for f in FIELDS:
        cnt = sum(1 for r in ok if f[0] in r.get("fields", {}))
        print(f"  {f[0]:22s} {cnt}/{len(ok)}")
    # show one example extraction
    for r in ok:
        if r.get("fields"):
            print(f"\nexample {r['code']}:", json.dumps(r["fields"], ensure_ascii=False))
            break


if __name__ == "__main__":
    main()
