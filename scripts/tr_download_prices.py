#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tr_download_prices.py — DLAP-TSE Turkey (BIST) Phase 1
========================================================
Download daily price data for all BIST stocks from İş Yatırım's public
HisseTekil endpoint (2008-01-01 .. today) and save raw JSON per ticker.

Endpoint (public, no key):
  https://www.isyatirim.com.tr/_layouts/15/Isyatirim.Website/Common/Data.aspx/HisseTekil?hisse=<TICKER>&startdate=dd-mm-yyyy&enddate=dd-mm-yyyy

Output: data_tr/raw/{ticker}_prices.json  (resumable: existing files skipped)
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path

import requests

ROOT = Path(os.environ.get("DLAP_ROOT", str(Path.home() / "research/dlap-tse")))
DATA = ROOT / "data_tr"
RAW = DATA / "raw"
RAW.mkdir(exist_ok=True, parents=True)

URL = ("https://www.isyatirim.com.tr/_layouts/15/Isyatirim.Website/Common/Data.aspx/"
       "HisseTekil")
START = "01-01-2008"
END = datetime.now().strftime("%d-%m-%Y")
SLEEP = 0.25
TIMEOUT = 60

# exit-IP rotation: "direct" plus local Xray HTTP proxies (proxy-pool skill)
def proxy_list():
    p = [x.strip() for x in os.environ.get("DLAP_PROXIES", "").split(",") if x.strip()]
    if p and "direct" not in p:
        p = ["direct"] + p
    if not p:
        # auto-detect xray http proxies on 12000+
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


def fetch(ticker):
    for attempt in (1, 2):
        proxy = next_proxy()
        try:
            if proxy == "direct":
                r = requests.get(URL, params={"hisse": ticker, "startdate": START,
                                              "enddate": END}, timeout=TIMEOUT)
            else:
                r = requests.get(URL, params={"hisse": ticker, "startdate": START,
                                              "enddate": END}, timeout=TIMEOUT,
                                 proxies={"http": proxy, "https": proxy})
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"  [{ticker}] via {proxy} attempt {attempt} failed: {e}",
                  flush=True)
            time.sleep(1.5)
    return None


def main():
    workers = int(os.environ.get("DLAP_WORKERS", "1"))
    tickers = [t.strip() for t in open(DATA / "tickers.txt", encoding="utf-8")
               if t.strip()]
    pending = [t for t in tickers if not (RAW / f"{t}_prices.json").exists()]
    print(f"{len(tickers)} tickers, {len(pending)} pending, {workers} workers, "
          f"range {START}..{END}", flush=True)
    done, failed, empty = len(tickers) - len(pending), [], 0
    lock = __import__("threading").Lock()
    total = len(pending)

    def work(tkr):
        nonlocal empty
        data = fetch(tkr)
        if data is None:
            with lock:
                failed.append(tkr)
            return
        if not data.get("value"):
            with lock:
                empty += 1
        with open(RAW / f"{tkr}_prices.json", "w", encoding="utf-8") as f:
            json.dump(data, f)
        with lock:
            nonlocal done
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(tickers)} done (failed {len(failed)}, "
                      f"empty {empty})", flush=True)
        time.sleep(SLEEP)

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(work, pending))
    else:
        for tkr in pending:
            work(tkr)
    print(f"\nDONE: {done} files, failed {len(failed)}, empty {empty}", flush=True)
    if failed:
        print("failed:", ",".join(failed), flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
