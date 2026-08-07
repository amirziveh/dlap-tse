#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pk_page_manifest.py — PK Phase 1: statement-page manifest for ALL company-years
================================================================================
For every symbol-year in the panel universe (financials_annual_no_financials.csv):
  1. download the annual-report PDF (stream)
  2. find the standalone SFP / PL / CFS pages:
       - text-layer reports: candidate_pages (statement markers) + per-page
         nex classification (fast, no qwen)
       - scanned reports: nex wide-scan over pages 15%-80%
  3. save  page_manifest/<SYM>_<YEAR>.json   (page numbers, markers, heuristics)
  4. save  page_images/<SYM>/<YEAR>_p<NN>_<kind>.jpg  (JPEG @110 DPI renders of
     every found statement page — for the human review contact sheets)
  5. delete the PDF (disk-light; ~5GB free)

No extraction happens here — this is ONLY page finding + evidence images.
"""
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pk_vlm_pipeline import (  # noqa: E402
    ANALYSIS, HEADING, TE_UPPER, BS_START, UNITS_K,
    company_annuals, download_pdf, parser_extract, page_count,
    candidate_pages, nex_classify, wide_scan, render_page,
    TMP_WORK, log, EXT_MODEL, llm_call,
)

PK = Path(os.environ.get("DLAP_PK", str(Path.home() / "research/dlap-tse/data_pk")))
MANIFEST_DIR = PK / "page_manifest"
IMAGE_DIR = PK / "page_images"
STATE_FILE = PK / "page_state.json"
LOG_FILE = Path("/tmp/pk_manifest.log")

FIN_FILE = PK / "financials_annual_no_financials.csv"

COST = {"calls": 0, "cost": 0.0}


def jpeg_render(pdf_path, idx, sym, year, kind, dpi=110):
    """Render one page to JPEG (pdftoppm -jpeg). Returns path or None."""
    out_dir = IMAGE_DIR / sym
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / f"{year}_tmp"
    for stale in out_dir.glob(f"{year}_tmp-*"):
        stale.unlink()
    r = subprocess.run(["pdftoppm", "-f", str(idx + 1), "-l", str(idx + 1),
                        "-r", str(dpi), "-jpeg", str(pdf_path), str(prefix)],
                       capture_output=True, timeout=120)
    if r.returncode != 0:
        return None
    hits = sorted(out_dir.glob(f"{year}_tmp-*"))
    if not hits:
        return None
    final = out_dir / f"{year}_p{idx + 1:02d}_{kind}.jpg"
    shutil.move(str(hits[0]), str(final))
    return str(final)


def process_row(sym, year, pdf_id):
    """Find statement pages for one company-year. Returns manifest dict or None."""
    wd = TMP_WORK / sym
    wd.mkdir(parents=True, exist_ok=True)
    pdf = wd / f"{year}.pdf"
    if not download_pdf(sym, year, pdf_id, pdf):
        log(f"  {sym} {year}: download FAILED")
        return {"symbol": sym, "year": year, "status": "download_failed"}
    try:
        pages, (si, ai, pi, ci), parser, summary = parser_extract(pdf)
        text_based = pages is not None
        found = {"sfp": [], "pl": [], "cfs": []}
        nex_pages = []
        heuristic = {"si": si, "ai": ai, "pi": pi, "ci": ci}
        if text_based:
            cands = candidate_pages(pages)
            for h in (ai, pi, ci):
                if h is not None:
                    cands.extend(range(max(0, h - 1), min(len(pages), h + 2)))
            cands = sorted({c for c in cands if not ANALYSIS.search(pages[c])})
            if cands:
                found = nex_classify(sym, year, pdf, cands, f"{TMP_WORK}/png/{sym}_{year}_m")
                nex_pages = cands
                if not any(found.values()):
                    found = {"sfp": [ai] if ai is not None else [],
                             "pl": [pi] if pi is not None else [],
                             "cfs": [ci] if ci is not None else []}
            else:
                found = {"sfp": [ai] if ai is not None else [],
                         "pl": [pi] if pi is not None else [],
                         "cfs": [ci] if ci is not None else []}
        else:
            n = page_count(pdf)
            if n <= 0:
                return {"symbol": sym, "year": year, "status": "no_pages"}
            found = wide_scan(sym, year, n, pdf, f"{TMP_WORK}/png/{sym}_{year}_m")
            nex_pages = ["wide"]

        # save JPEG evidence for found pages (cap 7 unique)
        imgs = []
        seen = set()
        for kind in ("sfp", "pl", "cfs"):
            for idx in sorted(found.get(kind) or []):
                if idx in seen or len(imgs) >= 7:
                    continue
                seen.add(idx)
                p = jpeg_render(pdf, idx, sym, year, kind)
                if p:
                    imgs.append(str(p))
        manifest = {
            "symbol": sym, "year": year, "status": "ok",
            "text_based": text_based, "n_pages": len(pages) if text_based else page_count(pdf),
            "sfp": sorted(found.get("sfp") or []),
            "pl": sorted(found.get("pl") or []),
            "cfs": sorted(found.get("cfs") or []),
            "heuristic": heuristic, "nex_pages": nex_pages,
            "images": imgs,
        }
        return manifest
    except Exception as e:
        log(f"  {sym} {year}: EXCEPTION {type(e).__name__}: {e}")
        return {"symbol": sym, "year": year, "status": "exception", "err": str(e)[:200]}
    finally:
        try:
            pdf.unlink()
        except Exception:
            pass


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", type=str, default="")
    a = ap.parse_args()
    only = {s.strip() for s in a.symbols.split(",") if s.strip()}
    rows = list(csv.DictReader(open(FIN_FILE)))
    if only:
        rows = [r for r in rows if r["symbol"] in only]
    ann_map = {}
    # collect pdf ids per symbol-year
    st = {}
    if STATE_FILE.exists():
        st = json.loads(STATE_FILE.read_text())

    tasks = []
    for r in rows:
        sym, year = r["symbol"], int(r["year"])
        key = f"{sym}|{year}"
        if st.get(key, {}).get("status") == "ok":
            continue
        ann = ann_map.get(sym)
        if ann is None:
            ann = company_annuals(sym)
            ann_map[sym] = ann
        rec = next((a for a in ann if a["year"] == year), None)
        if rec is None:
            st[key] = {"status": "no_record"}
            continue
        tasks.append((sym, year, rec["id"]))

    log(f"=== PAGE MANIFEST: {len(tasks)} pending of {len(rows)} rows ===")
    MANIFEST_DIR.mkdir(exist_ok=True)
    IMAGE_DIR.mkdir(exist_ok=True)
    done = 0

    def work(t):
        sym, year, pid = t
        m = process_row(sym, year, pid)
        if m:
            (MANIFEST_DIR / f"{sym}_{year}.json").write_text(json.dumps(m, indent=1))
            st[f"{sym}|{year}"] = {"status": m.get("status", "ok")}
        return t

    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(work, t): t for t in tasks}
        for fut in as_completed(futs):
            try:
                fut.result()
            except Exception as e:
                t = futs[fut]
                log(f"  worker exception {t}: {type(e).__name__}: {e}")
                st[f"{t[0]}|{t[1]}"] = {"status": "exception"}
            done += 1
            if done % 25 == 0:
                json.dump(st, open(STATE_FILE, "w"), indent=1)
                log(f"progress: {done}/{len(tasks)} done")

    json.dump(st, open(STATE_FILE, "w"), indent=1)
    ok = sum(1 for v in st.values() if v.get("status") == "ok")
    no_rec = sum(1 for v in st.values() if v.get("status") == "no_record")
    fail = sum(1 for v in st.values() if v.get("status") != "ok" and v.get("status") != "no_record")
    log(f"MANIFEST DONE: ok={ok} no_record={no_rec} failed={fail}")


if __name__ == "__main__":
    LOG_FILE.parent.mkdir(exist_ok=True)
    main()
