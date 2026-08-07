#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tr_download_financials.py — DLAP-TSE Turkey (BIST) Phase 2
============================================================
Download annual IFRS financial statements for all BIST stocks from
İş Yatırım's public MaliTablo endpoint.

Dispatch rule (verified): banks/specialized use financialGroup=UFRS,
others XI_29. Probe 2023 with UFRS; if empty, use XI_29 for all years.

One call per (ticker, year) with period1=12 (annual).
Years: 2013..current (2010-2012 only for banks, covered by the probe rule).

Output: data_tr/financials/{ticker}_{year}.json  (resumable)
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import requests

ROOT = Path(os.environ.get("DLAP_ROOT", str(Path.home() / "research/dlap-tse")))
DATA = ROOT / "data_tr"
FIN = DATA / "financials"
FIN.mkdir(exist_ok=True, parents=True)

URL = ("https://www.isyatirim.com.tr/_layouts/15/Isyatirim.Website/Common/Data.aspx/"
       "MaliTablo")
FIRST_YEAR = 2013
LAST_YEAR = datetime.now().year  # 2026
SLEEP = 0.25
TIMEOUT = 45

# exit-IP rotation (same as tr_download_prices.py): direct + local Xray proxies
def proxy_list():
    p = [x.strip() for x in os.environ.get("DLAP_PROXIES", "").split(",") if x.strip()]
    if p and "direct" not in p:
        p = ["direct"] + p
    if not p:
        found = []
        for port in range(12000, 12010):
            import socket
            s = socket.socket()
            s.settimeout(0.15)
            try:
                s.connect(("127.0.0.1", port))
                found.append(f"http://127.0.0.1:{port}")
            except OSError:
                pass
            finally:
                s.close()
        p = ["direct"] + found
    return p


PROXIES = proxy_list()
_proxy_idx = [0]
_proxy_lock = __import__("threading").Lock()


def next_proxy():
    with _proxy_lock:
        p = PROXIES[_proxy_idx[0] % len(PROXIES)]
        _proxy_idx[0] += 1
        return p


def fetch_financials(ticker, group, year):
    params = {
        "companyCode": ticker, "exchange": "TRY", "financialGroup": group,
        "year1": year, "period1": 12,
        "year2": year, "period2": 12,
        "year3": year, "period3": 12,
        "year4": year, "period4": 12,
    }
    for attempt in (1, 2):
        proxy = next_proxy()
        try:
            if proxy == "direct":
                r = requests.get(URL, params=params, timeout=TIMEOUT)
            else:
                r = requests.get(URL, params=params, timeout=TIMEOUT,
                                 proxies={"http": proxy, "https": proxy})
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"  [{ticker}/{year}] via {proxy} attempt {attempt}: {e}",
                  flush=True)
            time.sleep(1.5)
    return None


def main():
    workers = int(os.environ.get("DLAP_WORKERS", "1"))
    tickers = [t.strip() for t in open(DATA / "tickers.txt", encoding="utf-8")
               if t.strip()]
    print(f"{len(tickers)} tickers, years {FIRST_YEAR}..{LAST_YEAR}, "
          f"{workers} workers, proxies: {PROXIES}", flush=True)
    lock = __import__("threading").Lock()
    n_done = n_empty = n_fail = 0

    def work(tkr):
        nonlocal n_done, n_empty, n_fail
        # determine chart group by probing 2023 (or LAST_YEAR if 2023 < FIRST_YEAR)
        probe_year = 2023 if 2023 >= FIRST_YEAR else LAST_YEAR
        group = None
        for g in ("UFRS", "XI_29"):
            probe = fetch_financials(tkr, g, probe_year)
            if probe and probe.get("value"):
                group = g
                break
        if group is None:
            with lock:
                n_empty += 1
            return
        out_dir = FIN / tkr
        out_dir.mkdir(exist_ok=True, parents=True)
        for year in range(FIRST_YEAR, LAST_YEAR + 1):
            out = out_dir / f"{year}.json"
            if out.exists():
                continue
            data = fetch_financials(tkr, group, year)
            if data is None:
                with lock:
                    n_fail += 1
                continue
            with open(out, "w", encoding="utf-8") as f:
                json.dump(data, f)
            time.sleep(SLEEP)
        with lock:
            n_done += 1
            if n_done % 25 == 0:
                print(f"  {n_done} tickers with data (empty {n_empty}, "
                      f"fail {n_fail})", flush=True)

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(work, tickers))
    else:
        for tkr in tickers:
            work(tkr)
    print(f"\nDONE: {n_done} tickers with data, {n_empty} empty, "
          f"{n_fail} failed requests", flush=True)


if __name__ == "__main__":
    sys.exit(main())
