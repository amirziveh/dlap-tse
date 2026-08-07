#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pk_download_prices.py — DLAP-TSE Pakistan (PSX) Phase 1
========================================================
Download daily mkt_summary files (OHLCV for ALL symbols) from the
official PSX data portal for 2010-01-01 .. today.

Endpoint (public, no key):
  https://dps.psx.com.pk/download/mkt_summary/{YYYY-MM-DD}.Z
  (ZIP containing a pipe-delimited .txt: DATE|SYMBOL|CODE|NAME|OPEN|HIGH|LOW|CLOSE|VOLUME|AVG|...)

Output: data_pk/raw/{date}.txt   (resumable: existing files skipped)
"""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import requests

ROOT = Path(os.environ.get("DLAP_ROOT", str(Path.home() / "research/dlap-tse")))
DATA = ROOT / "data_pk"
RAW = DATA / "raw"
RAW.mkdir(exist_ok=True, parents=True)

URL = "https://dps.psx.com.pk/download/mkt_summary/{d}.Z"
START = date(2013, 12, 1)   # mkt_summary data begins ~Dec 2013 (verified)
END = date.today()
SLEEP = 0.15
TIMEOUT = 45
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"


def proxy_list():
    p = [x.strip() for x in os.environ.get("DLAP_PROXIES", "").split(",") if x.strip()]
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
        p = found
    # direct is throttled by PSX after ~30 reqs — use proxies only when available
    if not p:
        p = ["direct"]
    return p


PROXIES = proxy_list()
_proxy_idx = [0]
_proxy_lock = __import__("threading").Lock()


def next_proxy():
    with _proxy_lock:
        p = PROXIES[_proxy_idx[0] % len(PROXIES)]
        _proxy_idx[0] += 1
        return p


def all_weekdays(start, end):
    d = start
    while d <= end:
        if d.weekday() < 5:  # Mon-Fri
            yield d
        d += timedelta(days=1)


def fetch(day):
    url = URL.format(d=day.isoformat())
    out = RAW / f"{day.isoformat()}.txt"
    if out.exists():
        return "skip"
    for attempt in (1, 3):
        proxy = next_proxy()
        try:
            if proxy == "direct":
                r = requests.get(url, headers={"User-Agent": UA}, timeout=TIMEOUT)
            else:
                r = requests.get(url, headers={"User-Agent": UA}, timeout=TIMEOUT,
                                 proxies={"http": proxy, "https": proxy})
            if r.status_code == 404:
                return "none"  # holiday/weekend — not an error
            r.raise_for_status()
            content = r.content
            # .Z is a container: ZIP (recent) or GZIP (older) — extract inner text
            if content[:2] == b"PK":  # ZIP
                import io, zipfile
                z = zipfile.ZipFile(io.BytesIO(content))
                out.write_bytes(z.read(z.namelist()[0]))
            elif content[:2] == b"\x1f\x8b":  # GZIP
                import gzip
                out.write_bytes(gzip.decompress(content))
            elif content[:1] == b"0" and b"|" in content[:200]:
                out.write_bytes(content)  # already plain text
            else:
                raise ValueError(f"unknown container: {content[:8]!r}")
            return "ok"
        except Exception as e:
            print(f"  [{day}] via {proxy} attempt {attempt}: {e}", flush=True)
            time.sleep(0.8)
    return "fail"


def main():
    workers = int(os.environ.get("DLAP_WORKERS", "1"))
    days = list(all_weekdays(START, END))
    pending = [d for d in days if not (RAW / f"{d.isoformat()}.txt").exists()]
    print(f"{len(days)} weekdays {START}..{END}, {len(pending)} pending, "
          f"{workers} workers", flush=True)
    counts = {"ok": 0, "none": 0, "skip": len(days) - len(pending), "fail": 0}
    lock = __import__("threading").Lock()
    done = [0]

    def work(day):
        res = fetch(day)
        with lock:
            counts[res] += 1
            done[0] += 1
            if done[0] % 200 == 0:
                print(f"  {done[0]}/{len(pending)} (ok {counts['ok']}, "
                      f"none {counts['none']}, fail {counts['fail']})", flush=True)

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(work, pending))
    else:
        for day in pending:
            work(day)
    print(f"\nDONE: {counts}", flush=True)
    sys.exit(1 if counts["fail"] else 0)


if __name__ == "__main__":
    main()
