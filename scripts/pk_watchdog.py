#!/usr/bin/env python3
"""Watchdog: every 30 min, append a status line to data_pk/vlm_watchdog.log.
Detects stalls (row count not advancing)."""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/ubuntu/research/dlap-tse")
LOG = ROOT / "data_pk" / "vlm_watchdog.log"

last_rows = None
while True:
    time.sleep(1800)
    try:
        rows = len(list((ROOT / "data_pk" / "vlm_rows").glob("*.json")))
        cost = 0.0
        cf = ROOT / "data_pk" / "vlm_cost.json"
        if cf.exists():
            cost = json.loads(cf.read_text()).get("total", 0.0)
        st = {}
        sf = ROOT / "data_pk" / "vlm_state.json"
        if sf.exists():
            st = json.loads(sf.read_text())
        done = sum(1 for v in st.values() if v.get("status") == "done")
        line = (f"{time.strftime('%Y-%m-%d %H:%M:%S')} rows={rows} done={done} "
                f"cost=${cost:.3f}")
        if last_rows is not None and rows == last_rows:
            line += " *** STALL? ***"
        last_rows = rows
        with open(LOG, "a") as f:
            f.write(line + "\n")
        print(line, flush=True)
    except Exception as e:
        with open(LOG, "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} watchdog ERR {e}\n")
