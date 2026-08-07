#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pk_download_pdfs.py — download ALL annual-report PDFs for the PK universe
=========================================================================
Universe = symbols in financials_annual_no_financials.csv (355, paper sample).
For each symbol: fetch its annual-report records (company_annuals) and
download every year's PDF to <OUT>/<SYMBOL>/<YEAR>.pdf.

Resumable: existing files (size > 200KB) are skipped; progress in
<OUT>/downloads.json. Parallel downloads (I/O bound — 16 workers fine).
"""
import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pk_vlm_pipeline import company_annuals, download_pdf  # noqa: E402

OUT = Path.home() / "pk_pdfs"
FIN = Path.home() / "pk_financials.csv"   # rsynced from dlap-tse/data_pk
STATE = OUT / "downloads.json"
LOG = Path("/tmp/pk_download.log")

MIN_SIZE = 200_000  # bytes; smaller = failed/empty download


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def main():
    symbols = sorted({r["symbol"] for r in csv.DictReader(open(FIN))})
    log(f"universe: {len(symbols)} symbols")
    OUT.mkdir(parents=True, exist_ok=True)
    st = {}
    if STATE.exists():
        st = json.loads(STATE.read_text())

    # collect tasks (cache annual records per symbol)
    tasks = []
    ann_cache = {}
    for sym in symbols:
        ann = ann_cache.get(sym)
        if ann is None:
            ann = company_annuals(sym)
            ann_cache[sym] = ann
            time.sleep(0.2)  # be gentle with the portal
        for rec in ann:
            year = rec["year"]
            key = f"{sym}|{year}"
            pdf = OUT / sym / f"{year}.pdf"
            if st.get(key, {}).get("status") == "ok" and pdf.exists() and pdf.stat().st_size > MIN_SIZE:
                continue
            tasks.append((sym, year, rec["id"]))

    log(f"pending downloads: {len(tasks)}")

    def work(t):
        sym, year, pid = t
        pdf = OUT / sym / f"{year}.pdf"
        pdf.parent.mkdir(parents=True, exist_ok=True)
        ok = False
        for attempt in range(3):
            if download_pdf(sym, year, pid, pdf):
                if pdf.exists() and pdf.stat().st_size > MIN_SIZE:
                    ok = True
                    break
            time.sleep(1.5)
        return t, ok

    done = 0
    with ThreadPoolExecutor(max_workers=16) as ex:
        futs = {ex.submit(work, t): t for t in tasks}
        for fut in as_completed(futs):
            t, ok = fut.result()
            st[f"{t[0]}|{t[1]}"] = {"status": "ok" if ok else "failed"}
            done += 1
            if done % 50 == 0:
                STATE.write_text(json.dumps(st, indent=1))
                log(f"progress: {done}/{len(tasks)}")
    STATE.write_text(json.dumps(st, indent=1))

    ok_n = sum(1 for v in st.values() if v.get("status") == "ok")
    fail_n = sum(1 for v in st.values() if v.get("status") == "failed")
    total_bytes = sum(f.stat().st_size for f in OUT.rglob("*.pdf"))
    log(f"DOWNLOAD DONE: ok={ok_n} failed={fail_n} total={total_bytes/1e9:.1f}GB")


if __name__ == "__main__":
    main()
