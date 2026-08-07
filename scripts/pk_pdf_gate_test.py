#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PK-P0.3: PDF gate test — extract key fields from 10 companies' annual PDFs.

Companies across sectors: banks, cement, textile, oil/gas, insurance, sugar,
chemical, pharma, automobile, power.
For each: get annual PDF links from financials.psx.com.pk, download (via proxy),
pdftotext, find statement sections, extract key fields.
"""
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path("/home/ubuntu/research/dlap-tse")
OUT = ROOT / "data_pk"
TMP = Path("/tmp/pk_pdf_test")
TMP.mkdir(exist_ok=True, parents=True)

API = "https://financials.psx.com.pk/annQtrStmts.php"
DL = "https://financials.psx.com.pk/lib/DownloadPDF.php"
PROXIES = ["http://127.0.0.1:12000", "http://127.0.0.1:12001"]
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/126.0"

# company code, sector
SAMPLE = [
    ("HBL", "BANK"), ("MCB", "BANK"),
    ("DGKC", "CEMENT"), ("LUCK", "CEMENT"),
    ("NML", "TEXTILE"), ("PSO", "OIL"),
    ("EFERT", "FERTILIZER"), ("NESTLE", "FOOD"),
    ("COLG", "CONSUMER"), ("OGDC", "E&P"),
    ("ABOT", "PHARMA"), ("HUBC", "POWER"),
]

# IFRS label patterns for key fields
FIELDS = [
    ("revenue", r"^(Sales|Revenue|Net Sales|Turnover)[\s-]"),
    ("cost_of_sales", r"^Cost of (?:Goods )?Sales"),
    ("gross_profit", r"^Gross Profit"),
    ("net_income", r"^Profit(?: for the year)? (?:for the year )?(?:before|after)"),
    ("total_assets", r"^TOTAL ASSETS|^Total Assets$"),
    ("total_equity", r"^TOTAL EQUITY AND LIABILITIES|^Total Equity$|^TOTAL CAPITAL AND"),
    ("total_liabilities", r"^Total Liabilities"),
    ("current_assets", r"^Total Current Assets"),
    ("current_liabilities", r"^Total Current Liabilities"),
    ("ppe", r"^Property, Plant and Equipment"),
    ("cash", r"^Cash and (?:Cash )?Equivalents"),
    ("inventory", r"^Inventories|^Stock-in-Trade"),
]


def get_annual_links(code, year="2024"):
    r = requests.post(API, data={"name": "get_comp_y_data", "smbCode": code,
                                 "year": year}, headers={"User-Agent": UA}, timeout=30)
    rows = r.json()
    annuals = [x for x in rows if "Annual" in x.get("Reports", "")]
    urls = []
    for a in annuals:
        m = re.search(r'href="([^"]+)"', a["Reports"])
        if m:
            urls.append(DL + "?" + m.group(1).split("?")[-1] if "?" in m.group(1)
                        else DL + "?" + m.group(1).split("lib/DownloadPDF.php?id=")[-1])
    return urls


def fetch_pdf(url, dest):
    for px in PROXIES:
        try:
            r = requests.get(url, headers={"User-Agent": UA},
                             proxies={"http": px, "https": px}, timeout=90)
            r.raise_for_status()
            dest.write_bytes(r.content)
            return True
        except Exception as e:
            print(f"    dl fail via {px}: {e}", flush=True)
    return False


def extract(text, label_re):
    for line in text.splitlines():
        s = line.strip()
        if re.match(label_re, s, re.I):
            nums = re.findall(r"[\d,]+\.?\d*", s)
            if nums:
                return nums[-1]
    return None


def process(code, sector):
    out = {"code": code, "sector": sector}
    try:
        links = get_annual_links(code)
        if not links:
            out["status"] = "NO_ANNUAL_LINK"
            return out
        pdf = TMP / f"{code}.pdf"
        if not fetch_pdf(links[0], pdf):
            out["status"] = "DL_FAIL"
            return out
        size_mb = pdf.stat().st_size / 1e6
        r = subprocess.run(["pdftotext", "-layout", str(pdf), "-"],
                           capture_output=True, timeout=120)
        text = r.stdout.decode("utf-8", errors="ignore")
        if len(text) < 2000:
            out["status"] = "NO_TEXT"
            return out
        # limit to first 150k chars (statements come before notes usually)
        text = text[:150000]
        found = {}
        for name, pat in FIELDS:
            v = extract(text, pat)
            if v is not None:
                found[name] = v
        out.update({"status": "OK", "size_mb": round(size_mb, 1),
                    "text_len": len(text), "fields": found})
    except Exception as e:
        out["status"] = f"ERR {type(e).__name__}: {str(e)[:80]}"
    return out


def main():
    results = []
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = {ex.submit(process, c, s): c for c, s in SAMPLE}
        for f in futs:
            results.append(f.result())
    json.dump(results, open(OUT / "p0_pdf_gate_test.json", "w"), indent=1)
    for r in sorted(results, key=lambda x: x["code"]):
        status = r["status"]
        n = len(r.get("fields", {}))
        print(f"{r['code']:6s} {r['sector']:12s} {status:12s} "
              f"fields={n} size={r.get('size_mb', '-')}MB "
              f"text={r.get('text_len', '-')}")
    ok = [r for r in results if r["status"] == "OK"]
    print(f"\nOK: {len(ok)}/{len(results)}")
    if ok:
        allf = set()
        for r in ok:
            allf.update(r["fields"].keys())
        print("field coverage across OK companies:")
        for f in FIELDS:
            cnt = sum(1 for r in ok if f[0] in r.get("fields", {}))
            print(f"  {f[0]:22s} {cnt}/{len(ok)}")


if __name__ == "__main__":
    main()
