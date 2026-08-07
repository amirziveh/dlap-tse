#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PK VLM pipeline — PSX annual financial statements via VLM.

Division of labor (proven 2026-08-04):
  qwen/qwen3.7-flash  -> MULTI-IMAGE page arbitration (reasoning model:
                        max_tokens >= 2048 else content=null)
  nex-agi/nex-n2-mini -> per-page numeric extraction (SHORT prompt only;
                        long prompts return literal null)

Flow per company-year: manifest id -> download PDF (proxy round-robin) ->
pdftotext -layout -> candidate pages (free text signals) -> render 150 DPI
-> qwen arbitration (chunks of 10) -> nex extraction (sfp page(s), pl, cfs)
-> merge (aud page priority) -> QA L1/L2.5/L3/L4/L5 -> state.json + log.
Universe: 8 workers, stream-extract-delete, cost cap $5, resume-safe.

Usage:
  python pk_vlm_pipeline.py --validate          # ABOT,NML,OGDC,HUBC vs p0_e2e
  python pk_vlm_pipeline.py --universe          # full liquid universe
  python pk_vlm_pipeline.py --finalize          # L2/L3 pass + CSVs from rows
  python pk_vlm_pipeline.py --cost              # print cumulative cost
"""
import argparse
import base64
import collections
import csv
import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ROOT = Path("/home/ubuntu/research/dlap-tse")
OUT = ROOT / "data_pk"
TMP_WORK = Path("/tmp/pk_vlm")
E2E_DIR = Path("/tmp/pk_e2e")
REVIEW_DIR = OUT / "review"
STATE_FILE = OUT / "vlm_state.json"
LOG_FILE = OUT / "vlm_run.log"
COST_FILE = OUT / "vlm_cost.json"
MANIFEST_FILE = OUT / "pk_manifest.json"
ROWS_DIR = OUT / "vlm_rows"

TMP_WORK.mkdir(parents=True, exist_ok=True)
(TMP_WORK / "png").mkdir(parents=True, exist_ok=True)
REVIEW_DIR.mkdir(parents=True, exist_ok=True)
ROWS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "scripts"))
from pk_e2e_test import (find_statement_pages, parse_page_text,  # noqa: E402
                         parse_summary_aligned, section_sum, clean_num, approx,
                         load_opendoors_sample, BS_FIELDS, PL_FIELDS)

# ---------------------------------------------------------------- config ---
API = "https://financials.psx.com.pk/annQtrStmts.php"
PDF_URL = "https://financials.psx.com.pk/lib/DownloadPDF.php?id={}"
OR_URL = "https://openrouter.ai/api/v1/chat/completions"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")
ARB_MODEL = "qwen/qwen3.7-flash"
EXT_MODEL = "nex-agi/nex-n2-mini"
ARB_MAX_TOK = 4096          # reasoning model: <1024 -> content=null; headroom for multi-image batches
EXT_MAX_TOK = 20000          # cap, not cost: model stops at EOS anyway; 4096 was
                             # truncating big JSON extraction answers (user: «راحت
                             # باشه» 2026-08-06) — read at call time by extract_page
ARB_DPI = 100               # classification doesn't need 150 DPI (10s vs 50s)
EXT_DPI = 150               # extraction reads exact digits — keep 150
DPI = EXT_DPI
COST_LIMIT = 5.0
CHUNK = 15                  # arbitration images per call (cap ~20; 15 keeps most reports to ONE call)
TOL = 0.01

CFS_PAT = re.compile(r"Statement of Cash Flows|Cash Flow Statement", re.I)
ANALYSIS = re.compile(r"Vertical Analysis|Horizontal Analysis|Analysis of Financial"
                      r"|AT A GLANCE|Notes to the (Consolidated )?Financial Statements", re.I)
UNITS_K = re.compile(r"(Rupees?|PKR|Rs\.?)\s*(in\s+)?['\u2018\u2019]?0*000|Rupees? in thousand", re.I)
TE_UPPER = re.compile(r"TOTAL EQUITY AND LIABILITIES")
BS_START = re.compile(r"^\s*SHARE CAPITAL AND RESERVES\s*$|^\s*EQUITY AND LIABILITIES\s*$", re.M)
HEADING = re.compile(r"Statement of Financial Position|Balance Sheet"
                     r"|Statement of Profit or Loss|Statement of Profit and Loss"
                     r"|Profit or Loss Account|Profit and Loss Account"
                     r"|Statement of Cash Flows|Cash Flow Statement", re.I)

# SHORT field-group prompts — REQUIRED for nex-n2-mini (long prompts ->
# literal null + cross-page hallucinated values, see plan: "keep prompts
# field-specific"). The old single 13-field EXT_PROMPT was the top failure
# cause of extraction garbage in the batch run.
EXT_PROMPT = ("Extract from this page (current fiscal year, numbers as printed): "
              "total_assets, total_liabilities, total_equity, current_assets, "
              "current_liabilities, cash, inventory, ppe, long_term_investments, units. "
              "JSON only, integers, null if absent. units = 'thousands' or 'millions' as printed.")
PL_PROMPT = ("Extract from this page (current fiscal year, numbers as printed): "
             "revenue, cost_of_sales, gross_profit, net_income, units. "
             "JSON only, integers, null if absent. units = 'thousands' or 'millions' as printed.")
CFO_PROMPT = ("Extract from this page (current fiscal year, numbers as printed): "
              "operating_cash_flow, dividends_paid, units. "
              "JSON only, integers, null if absent. units = 'thousands' or 'millions' as printed.")

BS_KEYS = ["total_assets", "total_liabilities", "total_equity", "current_assets",
           "current_liabilities", "cash", "inventory", "dividends", "ppe",
           "long_term_investments"]
PL_KEYS = ["revenue", "cost_of_sales", "gross_profit", "net_income"]
CFS_KEYS = ["operating_cash_flow", "dividends_paid"]

# Which pages may carry a field: label hints (parser-verified). A field is
# only taken from a page whose text contains one of its hints — stops the
# VLM from hallucinating a BS field on a page that cannot contain it (e.g.
# "cash" on the equity side). If NO page matches, the filter is lifted.
LABEL_HINTS = {
    "total_assets": [r"TOTAL ASSETS", r"Total assets"],
    "total_liabilities": [r"TOTAL LIABILITIES", r"Total liabilities"],
    "total_equity": [r"TOTAL EQUITY", r"Total equity", r"TOTAL CAPITAL AND RESERVES"],
    "current_assets": [r"TOTAL CURRENT ASSETS", r"Total current assets", r"Current assets"],
    "current_liabilities": [r"TOTAL CURRENT LIABILITIES", r"Total current liabilities", r"Current liabilities"],
    "cash": [r"Cash and bank balances",
             r"Cash and cash equivalents(?!\s+at\s+(?:the\s+)?(?:end|beginning)\s+of\s+the\s+year)",
             r"Cash at bank", r"Cash in hand"],
    "inventory": [r"Inventories", r"Stock-in-trade"],
    "ppe": [r"Property, plant and equipment", r"Property and equipment",
            r"Operating fixed assets", r"Fixed assets"],
    "long_term_investments": [r"Long[ -]?term investments", r"Long term investment",
                              r"Investments[^,;]*at (?:fair|amortised)", r"Long term loans and advances"],
    "dividends": [r"Ordinary cash dividends", r"Cash dividends(?! paid)"],
    "revenue": [r"Net sales", r"Sales - net", r"Sales\s*\(net\)", r"R?REVENUE", r"Net revenue", r"Gross sales"],
    "cost_of_sales": [r"Cost of (?:goods |sales )?sales", r"Cost of goods sold", r"Cost of revenue"],
    "gross_profit": [r"Gross (?:profit|loss)"],
    "net_income": [r"Profit after taxation", r"Profit after tax", r"Profit for the year",
                   r"Net profit", r"Loss after taxation", r"Loss for the year"],
    "operating_cash_flow": [r"Net cash (?:\(outflow\) / inflow|generated from|from|generated by|used in|provided by) operating",
                            r"Cash generated from operations"],
    "dividends_paid": [r"Dividends? paid"],
}

PANEL_HDR = ["symbol", "year", "TA", "TL", "Eq", "CA", "CL", "Sales", "COGS", "GP",
             "PAT", "Cash", "Inv", "PPE", "lt_investments", "dividends", "dividends_paid",
             "cfo", "units", "unit_factor", "field_factors", "pages", "flags", "status"]

# ---------------------------------------------------------------- env/key ---
def api_key():
    k = os.environ.get("OPENROUTER_API_KEY")
    if k:
        return k
    try:
        for line in Path.home().joinpath(".bashrc").read_text().splitlines():
            if "OPENROUTER_API_KEY" in line and "=" in line:
                m = re.search(r"OPENROUTER_API_KEY\s*=\s*[\"']?([^\"'\s]+)", line)
                if m:
                    return m.group(1)
    except Exception:
        pass
    return None


KEY = api_key()
if not KEY:
    sys.exit("OPENROUTER_API_KEY not found (env or ~/.bashrc)")

# ------------------------------------------------------- state / log / cost ---
_lock = threading.Lock()
_stop = threading.Event()

_cost = {"total": 0.0, "prompt_tokens": 0, "completion_tokens": 0, "calls": 0,
         "models": {}}


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    with _lock:
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
    print(line, flush=True)


def load_state():
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {}


def save_state(st):
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=0, sort_keys=True))
    tmp.replace(STATE_FILE)


def mark(symbol, year, status, note=""):
    with _lock:
        st = load_state()
        st[f"{symbol}|{year}"] = {"status": status, "note": note}
        save_state(st)


def record_cost(model, cost, pt, ct):
    with _lock:
        _cost["total"] += cost
        _cost["prompt_tokens"] += pt
        _cost["completion_tokens"] += ct
        _cost["calls"] += 1
        m = _cost["models"].setdefault(model, {"calls": 0, "cost": 0.0})
        m["calls"] += 1
        m["cost"] += cost
        if _cost["calls"] % 10 == 0:
            COST_FILE.write_text(json.dumps(_cost, indent=1))
        if _cost["total"] > COST_LIMIT:
            log(f"COST LIMIT ${_cost['total']:.2f} exceeded — stopping")
            _stop.set()


# ------------------------------------------------------------ proxies ---
def proxy_list():
    p = [x.strip() for x in os.environ.get("DLAP_PROXIES", "").split(",") if x.strip()]
    if p:
        return p
    live = []
    for port in range(12000, 12010):
        try:
            r = requests.get("https://api.ipify.org?format=json",
                             proxies={"http": f"http://127.0.0.1:{port}",
                                      "https": f"http://127.0.0.1:{port}"},
                             timeout=4)
            if r.status_code == 200:
                live.append(f"http://127.0.0.1:{port}")
        except Exception:
            continue
    return live


PROXIES = proxy_list()
_proxy_idx = [0]


def next_proxy():
    if not PROXIES:
        return None
    with _lock:
        p = PROXIES[_proxy_idx[0] % len(PROXIES)]
        _proxy_idx[0] += 1
        return p


def fetch_url(url, data=None, timeout=90, tries=4, direct=False):
    """Proxy round-robin fetch via curl (the portal rejects Python-requests
    TLS fingerprints with an empty 200 body; curl passes). direct=True skips
    the proxy pool — it can stall on large files."""
    last = None
    if direct:
        tries = 2
    for i in range(tries):
        px = None if direct else next_proxy()
        cmd = ["curl", "-sL", "--max-time", str(timeout),
               "-H", f"User-Agent: {UA}",
               "-H", "X-Requested-With: XMLHttpRequest"]
        if px:
            cmd += ["-x", px]
        if data is not None:
            cmd += ["-d", data]
        cmd.append(url)
        try:
            r = subprocess.run(cmd, capture_output=True, timeout=timeout + 20)
            if r.returncode == 0 and r.stdout:
                return r.stdout
            last = f"rc={r.returncode} {r.stderr.decode(errors='ignore')[:100]}"
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
        time.sleep(1.5 * (i + 1))
    log(f"  fetch_url FAIL {url[:80]} ({last})")
    return None


# ------------------------------------------------------------ manifest ---
def load_manifest():
    if MANIFEST_FILE.exists():
        try:
            return json.loads(MANIFEST_FILE.read_text())
        except Exception:
            pass
    return {}


def company_annuals(symbol):
    """[{id, year, period}] for a symbol via get_comp_data (cached)."""
    man = load_manifest()
    if symbol in man:
        return man[symbol]
    recs = None
    for _ in range(3):
        body = fetch_url(API, data=f"name=get_comp_data&smbCode={symbol}")
        if body:
            try:
                recs = json.loads(body.decode("utf-8", errors="ignore"))
                break
            except Exception:
                recs = None
        time.sleep(2)
    ann = []
    if recs:
        for rec in recs:
            m = re.search(r"DownloadPDF\.php\?id=([^\"']+)", rec.get("Reports", ""))
            if not m:
                continue
            label = re.sub(r"<[^>]+>", "", rec.get("Reports", ""))
            if "annual" not in label.lower():
                continue
            ym = re.search(r"(20\d{2})", rec.get("period_ended", ""))
            ann.append({"id": m.group(1),
                        "year": int(ym.group(1)) if ym else None,
                        "period": rec.get("period_ended", "")})
    ann = [a for a in ann if a["year"]]
    ann.sort(key=lambda x: x["year"])
    with _lock:  # reload inside the lock — concurrent writers must not clobber
        man = load_manifest()
        man[symbol] = ann
        MANIFEST_FILE.write_text(json.dumps(man))
    return ann


# ------------------------------------------------------------ LLM ---
def b64(path):
    return base64.b64encode(Path(path).read_bytes()).decode()


def llm_call(model, text, images, max_tokens, json_mode=True, timeout=300):
    content = [{"type": "text", "text": text}]
    for p in images:
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64(p)}"}})
    payload = {"model": model,
               "messages": [{"role": "user", "content": content}],
               "max_tokens": max_tokens}
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    last = None
    for attempt in range(5):
        try:
            r = requests.post(OR_URL,
                              headers={"Authorization": f"Bearer {KEY}",
                                       "Content-Type": "application/json"},
                              json=payload, timeout=timeout)
            if r.status_code == 429:
                last = "429 rate-limited"
                wait = 8 * (attempt + 1) + 2 * attempt * attempt
                log(f"  llm 429 ({model}) — sleeping {wait}s")
                time.sleep(wait)
                continue
            r.raise_for_status()
            j = r.json()
            msg = j["choices"][0]["message"]
            usage = j.get("usage") or {}
            cost = float(usage.get("cost") or 0.0)
            record_cost(model, cost, usage.get("prompt_tokens", 0),
                        usage.get("completion_tokens", 0))
            return msg.get("content")
        except requests.exceptions.HTTPError as e:
            last = f"HTTP {e}"
            if "429" in str(e):
                wait = 8 * (attempt + 1) + 2 * attempt * attempt
                log(f"  llm 429 ({model}) — sleeping {wait}s")
                time.sleep(wait)
                continue
            if attempt < 4:
                time.sleep(3 * (attempt + 1))
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
            if attempt < 4:
                time.sleep(3 * (attempt + 1))
    log(f"  llm_call FAIL {model} ({last})")
    return None


# ------------------------------------------------------------ render ---
_render_seq = [0]  # unique suffix per render_page call (concurrency-safe)


def render_page(pdf_path, page_idx, prefix, dpi=None):
    """Render one 0-based page. Returns the PNG path (pdftoppm pads the
    suffix — glob it). Prefix gets a unique counter suffix so CONCURRENT
    renders of the same logical page (parallel extraction tasks) can never
    delete each other's PNG mid-read (was: FileNotFoundError race)."""
    dpi = dpi or DPI
    _render_seq[0] += 1
    prefix = f"{prefix}_{_render_seq[0]:04d}"
    for stale in Path(prefix).parent.glob(f"{Path(prefix).name}-*.png"):
        stale.unlink()
    subprocess.run(["pdftoppm", "-f", str(page_idx + 1), "-l", str(page_idx + 1),
                    "-r", str(dpi), "-png", str(pdf_path), prefix],
                   capture_output=True, timeout=120)
    hits = sorted(Path(prefix).parent.glob(f"{Path(prefix).name}-*.png"))
    return str(hits[0]) if hits else None


# ------------------------------------------------------------ page finding ---
def candidate_pages(pages):
    cands = []
    for i, p in enumerate(pages):
        if ANALYSIS.search(p):
            continue
        if (HEADING.search(p) and UNITS_K.search(p)) or TE_UPPER.search(p) or BS_START.search(p):
            cands.append(i)
    return cands


def heuristic_verdict(pages):
    """Fallback when arbitration fails — the e2e-hardened finder."""
    _, ai, pi, ci = find_statement_pages(pages)
    return {"sfp": [ai] if ai is not None else [],
            "pl": [pi] if pi is not None else [],
            "cfs": [ci] if ci is not None else []}


def parse_page_num(v):
    if v is None:
        return None
    s = str(v).strip()
    m = re.search(r"Page\s*(\d+)", s, re.I)
    if m:
        return int(m.group(1))
    if re.fullmatch(r"\d+", s):
        return int(s)
    return None


def parse_verdict(content):
    out = {"sfp": [], "pl": [], "cfs": []}
    units = {}
    if not content:
        return out, units
    try:
        d = json.loads(content)
    except Exception:
        return out, units
    if not isinstance(d, dict):
        return out, units
    for key in ("sfp", "pl", "cfs"):
        for v in d.get(key) or []:
            n = parse_page_num(v)
            if n is not None:
                out[key].append(n)
    u = d.get("units") or {}
    if isinstance(u, dict):
        for k, v in u.items():
            n = parse_page_num(k)
            if n is not None and isinstance(v, str):
                units[n] = v.lower().strip()
    return out, units


ARB_PROMPT = (
    "Images below are pages of a Pakistani listed company's annual report "
    "(fiscal year {year}). Each image is labeled 'Page N (report page M)'. "
    "Classify each page as one of: SFP = audited standalone statement of "
    "financial position (balance sheet with numbers), PL = statement of profit "
    "or loss, CFS = statement of cash flows. "
    "Reply with JSON: {{\"sfp\": [\"Page N\", ...], \"pl\": [\"Page N\", ...], "
    "\"cfs\": [\"Page N\", ...], \"units\": {{\"Page N\": \"thousands\" or "
    "\"millions\"}}}}. Use exactly the 'Page N' labels given. Do NOT include "
    "summary tables, vertical/horizontal analysis (percentage tables), "
    "index/base-100 tables, notes pages, covers or TOC. When in doubt, "
    "INCLUDE the page.")


def nex_classify(symbol, year, pdf_path, cands, tag):
    """Per-page nex classification fallback — used when qwen arbitration
    comes back EMPTY (the 127-row failure bucket). nex-n2-mini classifies
    single pages reliably (proven: ACPL 2018 SFP/PL/CFS)."""
    verdict = {"sfp": [], "pl": [], "cfs": []}
    prompt = ("What financial statement is this page: balance sheet (sfp), "
              "income statement (pl), cash flow (cfs), or other? "
              "Answer exactly one word.")
    for j, idx in enumerate(cands):
        png = render_page(pdf_path, idx, f"{tag}_nex_{j}", dpi=ARB_DPI)
        if not png:
            continue
        content = None
        for att in range(2):
            try:
                content = llm_call(EXT_MODEL, prompt, [png], 300)
            except Exception:
                content = None
            if content and content.strip().lower() not in ("null", "none", "{}", "[]", ""):
                break
            time.sleep(2)
        word = (content or "").strip().lower()
        for key, alts in (("sfp", ("sfp", "balance", "sheet")),
                          ("pl", ("pl", "income", "profit", "loss")),
                          ("cfs", ("cfs", "cash", "flow"))):
            if any(a in word for a in alts):
                verdict[key].append(idx)
                log(f"  {symbol} {year} nex-classified p{idx + 1} as {key}")
                break
        time.sleep(0.3)
    return verdict


def pdf_locate_4omini(symbol, year, pdf_path, n_pages, tag):
    """ONE+ calls: send the PDF (chunked to ≤40 pages / ≤18MB sub-PDFs) to
    gpt-4o-mini → page numbers of SFP/PL/CFS. Cheap ($0.0004-0.002/rep,
    ~10-30s) and ±1-2 pages accurate (printed vs PDF-index offset possible —
    caller refines with nex on a ±3 window). Returns verdict
    {sfp/pl/cfs: [0-based idx]} or empty (caller then uses wide_scan).
    Chunking fixes OpenRouter's 'Too many images (max 50)' + file-size caps."""
    verdict = {"sfp": [], "pl": [], "cfs": []}

    def locate_one(sub_pdf, offset):
        """Locate in one sub-PDF; map 'Page N' back by offset. Returns bool."""
        try:
            size = Path(sub_pdf).stat().st_size
            if size > 18 * 1024 * 1024:
                log(f"  {symbol} {year}: chunk {Path(sub_pdf).name} {size/1e6:.0f}MB too big")
                return False
            b64 = base64.b64encode(Path(sub_pdf).read_bytes()).decode()
            prompt = ("This is part of an annual report PDF. Find the pages containing: "
                      "1) the audited STANDALONE balance sheet / statement of financial position (SFP), "
                      "2) the STANDALONE income statement / profit or loss account (PL), "
                      "3) the STANDALONE cash flow statement (CFS). NOT consolidated, NOT notes, "
                      "NOT summary tables. Answer STRICT JSON: "
                      "{\"sfp\": [page numbers], \"pl\": [page numbers], \"cfs\": [page numbers]}.")
            content = [{"type": "text", "text": prompt},
                       {"type": "file", "file": {"filename": f"{symbol}_{year}.pdf",
                                                 "file_data": f"data:application/pdf;base64,{b64}"}}]
            payload = {"model": "openai/gpt-4o-mini",
                       "messages": [{"role": "user", "content": content}],
                       "max_tokens": 1024,
                       "provider": {"order": ["OpenAI"], "allow_fallbacks": False}}
            r = requests.post(OR_URL,
                              headers={"Authorization": f"Bearer {KEY}",
                                       "Content-Type": "application/json"},
                              json=payload, timeout=240)
            r.raise_for_status()
            j = r.json()
            if "choices" not in j:  # 200-with-error-body (provider limits)
                log(f"  {symbol} {year}: locator provider error: {str(j.get('error'))[:120]}")
                return False
            usage = j.get("usage") or {}
            record_cost("openai/gpt-4o-mini", usage.get("cost") or 0.0,
                        usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
            content = j["choices"][0]["message"].get("content")
            d = _salvage_json(content)
            if not d:
                log(f"  {symbol} {year}: locator bad JSON: {(content or '')[:100]!r}")
                return False
            for key in ("sfp", "pl", "cfs"):
                for v in d.get(key) or []:
                    n = parse_page_num(v)
                    if n is not None and 1 <= n <= n_pages - offset:
                        verdict[key].append(offset + n - 1)
            return True
        except Exception as e:
            log(f"  {symbol} {year}: locator ERR {type(e).__name__}: {e}")
            return False

    try:
        # ---- attempt 1: the ORIGINAL file (cheapest: ~75-90 tok/page text
        # mode when the embedded OCR layer is readable; $0.0004-0.002/rep)
        size = Path(pdf_path).stat().st_size
        if size <= 18 * 1024 * 1024 and locate_one(pdf_path, 0):
            for key in ("sfp", "pl", "cfs"):
                verdict[key] = sorted(set(verdict[key]))
            log(f"  {symbol} {year}: 4o-mini locator (original) "
                f"sfp={[p + 1 for p in verdict['sfp']]} "
                f"pl={[p + 1 for p in verdict['pl']]} "
                f"cfs={[p + 1 for p in verdict['cfs']]}")
            return verdict

        # ---- attempt 2: chunks via pdfseparate/pdfunite (poppler) — copies
        # original streams, keeps the cheap text-extraction path (qpdf's
        # rewrite forces IMAGE billing: 804K tok vs 23K tok per 40 pages!)
        # Chunk ≤35 pages (pdfunite inflates size; stays under 50-image cap).
        chunks = []
        start = 0
        CH = 35
        while start < n_pages:
            end = min(n_pages, start + CH)
            chunks.append((start + 1, end))
            start = end
        tmpdir = Path(f"/tmp/pk_loc_{symbol}_{year}")
        tmpdir.mkdir(exist_ok=True)
        any_ok = False
        for ci, (a, b) in enumerate(chunks):
            sub = tmpdir / f"c{ci}.pdf"
            r = subprocess.run(["pdfseparate", "-f", str(a), "-l", str(b),
                                str(pdf_path), str(tmpdir / "pg_%d.pdf")],
                               capture_output=True, timeout=180)
            if r.returncode != 0:
                continue
            pgs = sorted((tmpdir / "pg_*.pdf").parent.glob("pg_*.pdf"))
            if not pgs:
                continue
            r2 = subprocess.run(["pdfunite"] + [str(p) for p in pgs] + [str(sub)],
                                capture_output=True, timeout=180)
            for p in pgs:
                p.unlink(missing_ok=True)
            if r2.returncode != 0 or not sub.exists():
                continue
            if locate_one(str(sub), a - 1):
                any_ok = True
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)
        for key in ("sfp", "pl", "cfs"):
            verdict[key] = sorted(set(verdict[key]))
        if any_ok:
            log(f"  {symbol} {year}: 4o-mini locator (chunked) "
                f"sfp={[p + 1 for p in verdict['sfp']]} "
                f"pl={[p + 1 for p in verdict['pl']]} "
                f"cfs={[p + 1 for p in verdict['cfs']]}")
        else:
            log(f"  {symbol} {year}: 4o-mini locator empty (all attempts failed)")
    except Exception as e:
        log(f"  {symbol} {year}: locator chunking ERR {type(e).__name__}: {e}")
    return verdict


def wide_scan(symbol, year, n_pages, pdf_path, tag):
    """Vision-only page finding over a window of the PDF (pages ~15%-72%:
    covers/TOC early, notes late). Rescues scanned & stub-text reports where
    text-based candidate finding finds nothing (ADAMS/ADOS 2018-2022 class).
    Uses nex per-PAGE classification (single image) — qwen multi-image
    chunks came back EMPTY here (the 127-row flake), while nex per-page has
    been 100% reliable (ACPL 2018 test). ~30 pages x 2s ≈ 1 min ≈ $0.0003."""
    lo, hi = int(n_pages * 0.15), int(n_pages * 0.80)
    cands = list(range(lo, min(hi, n_pages)))
    verdict = {"sfp": [], "pl": [], "cfs": []}
    if not cands:
        return verdict
    log(f"  {symbol} {year}: wide-scan {len(cands)} pages ({lo + 1}-{min(hi, n_pages)}) via nex")
    prompt = ("What financial statement is this page: balance sheet (sfp), "
              "income statement (pl), cash flow (cfs), or other? "
              "Answer exactly one word.")
    done_idx = set()

    def classify_one(idx):
        if idx in done_idx or not (0 <= idx < n_pages):
            return
        done_idx.add(idx)
        png = render_page(pdf_path, idx, f"{tag}_wide_{idx}", dpi=ARB_DPI)
        if not png:
            return
        content = None
        for att in range(2):
            try:
                content = llm_call(EXT_MODEL, prompt, [png], 300)
            except Exception:
                content = None
            if content and content.strip().lower() not in ("null", "none", "{}", "[]", ""):
                break
            time.sleep(1.5)
        word = (content or "").strip().lower()
        for key, alts in (("sfp", ("sfp", "balance", "sheet")),
                          ("pl", ("pl", "income", "profit", "loss")),
                          ("cfs", ("cfs", "cash", "flow"))):
            if any(a in word for a in alts):
                verdict[key].append(idx)
                log(f"  {symbol} {year} wide-nex p{idx + 1} -> {key}")
                break

    for idx in cands:
        classify_one(idx)
    # CFS often sits just after PL but outside the window on short reports
    if not verdict["cfs"] and verdict["pl"]:
        pl = verdict["pl"][-1]
        log(f"  {symbol} {year}: CFS missing — scanning p{pl + 2}-{min(pl + 5, n_pages)} after PL")
        for idx in range(pl + 1, min(pl + 5, n_pages)):
            classify_one(idx)
    for key in ("sfp", "pl", "cfs"):
        verdict[key] = sorted(set(verdict[key]))
    log(f"  {symbol} {year}: wide-scan verdict sfp={[p + 1 for p in verdict['sfp']]} "
        f"pl={[p + 1 for p in verdict['pl']]} cfs={[p + 1 for p in verdict['cfs']]}")
    return verdict


def page_count(pdf_path):
    try:
        r = subprocess.run(["pdfinfo", str(pdf_path)], capture_output=True,
                           text=True, timeout=60)
        m = re.search(r"^Pages:\s+(\d+)", r.stdout, re.M)
        return int(m.group(1)) if m else 0
    except Exception:
        return 0


def arbitrate(symbol, year, pages, cands, pdf_path, tag, heur=None):
    """qwen multi-image arbitration over candidates (chunks of CHUNK).
    heur = (ai, pi, ci) heuristic indices: injected into the verdict for a
    chunk that exhausts retries (avoids 90s retry storms; QA still gates).
    Returns (verdict{sfp,pl,cfs: [page idx]}, units{page_idx: 'thousands'})."""
    verdict = {"sfp": [], "pl": [], "cfs": []}
    units = {}
    if not cands:
        return verdict, units
    for start in range(0, len(cands), CHUNK):
        chunk = cands[start:start + CHUNK]
        pngs, labels = [], []
        for j, idx in enumerate(chunk):
            png = render_page(pdf_path, idx, f"{tag}_arb_{start + j}", dpi=ARB_DPI)
            if png:
                pngs.append(png)
                labels.append(f"Page {j} (report page {idx + 1})")
        if not pngs:
            continue
        prompt = ARB_PROMPT.format(year=year) + "\n" + "\n".join(labels)
        got = None
        # one retry for transient failures; empty verdicts after that get the
        # heuristic injection immediately (retrying 4x burns $0.0012 for the
        # same empty answer — QA gates the injection anyway)
        for attempt in range(2):
            try:
                content = llm_call(ARB_MODEL, prompt, pngs, ARB_MAX_TOK)
                v, u = parse_verdict(content)
                if any(v.values()):
                    got = (v, u)
                    break
                if attempt == 0:
                    time.sleep(2.5)
            except Exception as e:
                log(f"  {symbol} {year} arbitrate attempt {attempt} ERR {type(e).__name__}: {e}")
                if attempt == 0:
                    time.sleep(2.5)
        if got:
            v, u = got

            def map_n(n):
                # chunk-local label first; if it points outside the chunk,
                # interpret as the REPORT page number (1-based)
                if 0 <= n < len(chunk):
                    return chunk[n]
                for c in chunk:
                    if c + 1 == n:
                        return c
                return None

            for key in ("sfp", "pl", "cfs"):
                for n in v[key]:
                    c = map_n(n)
                    if c is not None:
                        verdict[key].append(c)
            for n, unit in u.items():
                c = map_n(n)
                if c is not None:
                    units[c] = unit
        else:
            log(f"  {symbol} {year} arbitration chunk empty — heuristic injection")
            if heur:
                for key, hidx in (("sfp", heur[1]), ("pl", heur[2]), ("cfs", heur[3])):
                    if hidx is not None and hidx in chunk and hidx not in verdict[key]:
                        verdict[key].append(hidx)
                        log(f"  {symbol} {year} injected heuristic p{hidx + 1} into {key}")
    for key in ("sfp", "pl", "cfs"):
        verdict[key] = sorted(set(verdict[key]))
    if not any(verdict.values()):
        log(f"  {symbol} {year} arbitration EMPTY -> heuristic fallback")
        verdict = heuristic_verdict(pages)
        if not any(verdict.values()):
            log(f"  {symbol} {year} heuristic empty too -> nex per-page fallback")
            verdict = nex_classify(symbol, year, pdf_path, cands, tag)
    return verdict, units


# ------------------------------------------------------------ extraction ---
def _salvage_json(content):
    """Try strict parse, then strip markdown fences / surrounding prose,
    then extract the first balanced {...} block. Returns dict or None."""
    if not content:
        return None
    try:
        d = json.loads(content)
        return d if isinstance(d, dict) else None
    except Exception:
        pass
    s = content.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", s)
        try:
            d = json.loads(s)
            return d if isinstance(d, dict) else None
        except Exception:
            pass
    m = re.search(r"\{.*\}", s, re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            return d if isinstance(d, dict) else None
        except Exception:
            return None
    return None


def extract_page(symbol, year, pdf_path, page_idx, prompt, tag, max_tok=None):
    max_tok = max_tok or EXT_MAX_TOK  # read at CALL time, not def time
    png = render_page(pdf_path, page_idx, f"{tag}_x{page_idx}")
    if not png:
        return None
    for attempt in range(3):
        try:
            content = llm_call(EXT_MODEL, prompt, [png], max_tok)
            if content and content.strip().lower() not in ("null", "none", "{}", ""):
                d = _salvage_json(content)
                if d is not None:
                    return {str(k).strip().lower().replace(" ", "_"): v
                            for k, v in d.items()}
                log(f"  {symbol} {year} p{page_idx} attempt {attempt} JSON parse FAIL: {content[:120]!r}")
        except Exception as e:
            log(f"  {symbol} {year} p{page_idx} extract attempt {attempt} ERR {type(e).__name__}: {e}")
        if attempt < 2:
            time.sleep(2.5)
    return None


def num(v):
    n = clean_num(v)
    if n is None:
        return None
    return int(round(n)) if abs(n - round(n)) < 1e-6 else n


def merge_fields(symbol, year, pdf_path, tag, pages, bs_primary, bs_extra,
                 pl_primary, pl_extra, cfs_primary, cfs_extra, units,
                 parser_cur=None):
    """Extract per page (in parallel), merge with aud-priority + label-hint
    filtering.

    - Primary pages (aud spread, PL page, CFS page) are always extracted.
    - Extra pages (qwen verdicts beyond the spread — usually the CONSOLIDATED
      statements section) are extracted lazily, only to fill still-missing
      fields, and never when they carry Consolidated-marked text.
    - A field is only taken from a page whose TEXT contains the field's label
      hint (kills hallucinations like 'cash' on the equity side); if no page
      matches, the filter is lifted.
    Returns (fields dict, src dict {field: page})."""
    f, src = {}, {}
    consol = re.compile(r"\bConsolidated\b", re.I)

    def page_has_hint(pg, keys):
        if not pages:  # vision mode (scanned): no text to hint — extract anyway
            return True
        if not (0 <= pg < len(pages)):
            return False
        hints = [h for k in keys for h in LABEL_HINTS.get(k, [])]
        return not hints or any(re.search(r"(?m)^\s*" + h, pages[pg], re.I)
                                for h in hints)

    def eligible(field, pgs):
        hints = LABEL_HINTS.get(field)
        if not hints:
            return None
        hits = [pg for pg in pgs if 0 <= pg < len(pages)
                and any(re.search(r"(?m)^\s*" + h, pages[pg], re.I) for h in hints)]
        return hits if hits else None

    def extra_filter(pgs, primary):
        if not primary:
            return pgs  # standalone primary missing -> verdict pages ARE the statements
        return [p for p in pgs if not consol.search(pages[p] if 0 <= p < len(pages) else "")]

    def take(d, keys, pg, scale, pgs):
        for k in keys:
            v = num(d.get(k))
            if v is None or k in f:
                continue
            # statement totals at thousands scale are never < 10,000 — kills
            # base-100 index-table junk (100/1,130/1,060.49) at the source
            if abs(v) < 10000:
                continue
            elig = eligible(k, pgs)
            if elig is not None and pg not in elig:
                continue
            f[k] = v * scale
            src[k] = pg

    # ---- extraction plan: phase 1 = primaries (parallel), phase 2 = extras
    # only if fields still missing ----
    def plan_pass(pgs, keys, prompt):
        return [(pg, prompt, keys) for pg in pgs if page_has_hint(pg, keys)]

    results = {}

    def run_plan(tasks):
        if not tasks:
            return
        seen, uniq = set(), []
        for pg, prompt, keys in tasks:
            if (pg, prompt) in seen:
                continue
            seen.add((pg, prompt))
            uniq.append((pg, prompt, keys))

        def do_extract(pg, prompt):
            results[pg] = extract_page(symbol, year, pdf_path, pg, prompt, tag)

        if len(uniq) > 1:
            with ThreadPoolExecutor(max_workers=min(4, len(uniq))) as ex:
                futs = [ex.submit(do_extract, pg, prompt) for pg, prompt, keys in uniq]
                for fut in futs:
                    fut.result()
        else:
            for pg, prompt, keys in uniq:
                do_extract(pg, prompt)

    run_plan(plan_pass(bs_primary, BS_KEYS, EXT_PROMPT)
             + plan_pass(pl_primary, PL_KEYS, PL_PROMPT)
             + plan_pass(cfs_primary, CFS_KEYS, CFO_PROMPT))
    # phase 2: extras for still-missing fields (never Consolidated-marked)
    bs_missing = [k for k in BS_KEYS if k not in f]
    pl_missing = [k for k in PL_KEYS if k not in f]
    cfs_missing = [k for k in CFS_KEYS if k not in f]
    if bs_missing and bs_extra:
        run_plan(plan_pass(extra_filter(bs_extra, bs_primary), bs_missing, EXT_PROMPT))
    if pl_missing and pl_extra:
        run_plan(plan_pass(extra_filter(pl_extra, pl_primary), pl_missing, PL_PROMPT))
    if cfs_missing and cfs_extra:
        run_plan(plan_pass(extra_filter(cfs_extra, cfs_primary), cfs_missing, CFO_PROMPT))

    # ---- merge in priority order ----
    # units reported by the VLM itself (vision mode has no arbitration units)
    ext_units = {}
    for pg, d in results.items():
        if isinstance(d, dict) and d.get("units"):
            u = str(d["units"]).strip().lower()
            if u.startswith("m"):
                ext_units[pg] = "millions"
            elif u.startswith("t") or u.startswith("k"):
                ext_units[pg] = "thousands"

    def scale_of(pg):
        u = units.get(pg) or ext_units.get(pg)
        return 1000 if u == "millions" else 1

    for pg in bs_primary:
        d = results.get(pg)
        if not d:
            continue
        scale = scale_of(pg)
        take(d, BS_KEYS, pg, scale, bs_primary)
        if pg in pl_primary:
            take(d, PL_KEYS, pg, scale, pl_primary)  # same page doubles as PL
    for pg in pl_primary:
        if pg in bs_primary:
            continue
        d = results.get(pg)
        if not d:
            continue
        scale = scale_of(pg)
        take(d, PL_KEYS, pg, scale, pl_primary)
    for pg in cfs_primary:
        if pg in bs_primary or pg in pl_primary:
            continue
        d = results.get(pg)
        if not d:
            continue
        scale = scale_of(pg)
        take(d, CFS_KEYS, pg, scale, cfs_primary)
    for pg in bs_extra + pl_extra + cfs_extra:
        d = results.get(pg)
        if not d:
            continue
        scale = scale_of(pg)
        take(d, [k for k in BS_KEYS + PL_KEYS + CFS_KEYS if k not in f],
             pg, scale, [pg])
    # pass 3: parser-fill — the e2e-verified pdftotext parser's values for
    # fields the VLM could not extract (|v| >= 1000 keeps index-table junk out)
    if parser_cur:
        for k in list(BS_KEYS) + list(PL_KEYS) + list(CFS_KEYS):
            if k in f or k not in parser_cur:
                continue
            v = parser_cur[k]
            # >=10000 keeps index-table junk out; not a year token (20xx)
            if v is not None and abs(v) >= 10000 and not (1900 <= v <= 2100):
                f[k] = v
                src[k] = "parser-fill"
                log(f"  {symbol} {year} parser-fill {k}={v}")
    # dividends < 1000 = per-share figure, not a statement total; negative
    # dividends = CFS "Dividends paid" cash outflow -> dividends_paid column
    for k in ("dividends", "dividends_paid"):
        if k in f and abs(f[k]) < 1000:
            log(f"  {symbol} {year} {k}={f[k]} <1000 -> per-share, dropped")
            del f[k]
            src.pop(k, None)
    if f.get("dividends") is not None and f["dividends"] < 0:
        log(f"  {symbol} {year} dividends={f['dividends']} <0 -> cash outflow, "
            f"moved to dividends_paid")
        if "dividends_paid" not in f:
            f["dividends_paid"] = f["dividends"]
            src["dividends_paid"] = src.get("dividends")
        del f["dividends"]
        src.pop("dividends", None)
    return f, src


# ------------------------------------------------------------ QA ---
def run_qa(symbol, year, f, parser_cur, parser_sum):
    """-> list of [layer, field, vlm, ref, detail]. L1/L2.5/L4/L5 (L2/L3 run
    company-level in finalize)."""
    fl = []
    ta, tl, eq = f.get("total_assets"), f.get("total_liabilities"), f.get("total_equity")
    if ta is not None and tl is not None and eq is not None:
        if not approx(ta, tl + eq):
            fl.append(["L1", "TA=TL+Eq", ta, tl + eq, ""])
    sale, cogs, gp = f.get("revenue"), f.get("cost_of_sales"), f.get("gross_profit")
    if sale is not None and cogs is not None and gp is not None:
        if not approx(sale - abs(cogs), gp):
            fl.append(["L1", "Sales-COGS=GP", sale - abs(cogs), gp, ""])
    # L2.5: 6-year summary (parser, units vary) vs audited VLM
    for fld in ("total_assets", "total_equity", "total_liabilities", "revenue", "net_income"):
        s, a = parser_sum.get(fld), f.get(fld)
        if s is None or a is None:
            continue
        ok = approx(a, s) or approx(a, s * 1000) or approx(a * 1000, s)
        if not ok:
            fl.append(["L2.5", fld, a, s, "summary-vs-audited"])
    # L4: opendoors anchors
    sample = load_opendoors_sample()
    ext = sample.get((symbol, year)) if sample else None
    if ext:
        for fld, ev in ext.items():
            v = f.get(fld)
            if v is not None and ev is not None and not approx(v, ev):
                fl.append(["L4", fld, v, ev, "vs opendoors"])
    # L5: VLM vs pdftotext parser on the same page (parser-missing = allowed)
    for k, v in f.items():
        pv = parser_cur.get(k)
        if pv is None:
            continue
        if k == "cost_of_sales" and approx(abs(v), abs(pv)):
            continue  # negative-expense format: sign convention varies
        if not approx(v, pv):
            fl.append(["L5", k, v, pv, "VLM vs parser"])
    return fl


# ------------------------------------------------------------ company-year ---
def parser_extract(pdf_path):
    """pdftotext parser A (from pk_e2e_test) returning pages + fields."""
    r = subprocess.run(["pdftotext", "-layout", str(pdf_path), "-"],
                       capture_output=True, timeout=180)
    text = r.stdout.decode("utf-8", errors="ignore")
    if len(text.strip()) < 200:
        return None, (None, None, None, None), {}, {}
    pages = text.split("\f")
    si, ai, pi, ci = find_statement_pages(pages)
    fields = {}
    if ai is not None:
        for pg in range(max(0, ai - 1), min(len(pages), ai + 2)):
            for k, v in parse_page_text(pages[pg], BS_FIELDS).items():
                fields.setdefault(k, {"cur": v[0], "prior": v[1], "ev": v[2], "src": "aud"})
        if "current_assets" not in fields:
            s = section_sum(pages, ai, r"^Current Assets$")
            if s is not None:
                fields["current_assets"] = {"cur": s, "prior": None, "ev": "section sum", "src": "sum"}
        if "current_liabilities" not in fields:
            s = section_sum(pages, ai, r"^Current Liabilities$")
            if s is not None:
                fields["current_liabilities"] = {"cur": s, "prior": None, "ev": "section sum", "src": "sum"}
    if pi is not None:
        for pg in range(pi, min(pi + 2, len(pages))):
            for k, v in parse_page_text(pages[pg], PL_FIELDS).items():
                fields.setdefault(k, {"cur": v[0], "prior": v[1], "ev": v[2], "src": "pl"})
    if ci is not None:
        for pg in range(ci, min(ci + 2, len(pages))):
            for k, v in parse_page_text(pages[pg], ("cfo",)).items():
                fields.setdefault(k, {"cur": v[0], "prior": v[1], "ev": v[2], "src": "cfs"})
    summary = {}
    if si is not None and si != ai:
        try:
            summary = parse_summary_aligned(pdf_path, si)
        except Exception:
            summary = {}
    return pages, (si, ai, pi, ci), fields, summary


def download_pdf(symbol, year, pdf_id, dest):
    if dest.exists() and dest.stat().st_size > 10000:
        return True
    for _ in range(3):
        body = fetch_url(PDF_URL.format(pdf_id), timeout=120)
        if body and len(body) > 10000:
            dest.write_bytes(body)
            return True
        time.sleep(2)
    # proxy pool can stall on large files — direct fallback
    body = fetch_url(PDF_URL.format(pdf_id), timeout=120, direct=True)
    if body and len(body) > 10000:
        dest.write_bytes(body)
        return True
    return False


def process_company_year(symbol, year, pdf_path, gt=None):
    """Full pipeline for one company-year. Returns (row, parser_cur,
    parser_prior, flags) or None on failure."""
    tag = f"{TMP_WORK}/png/{symbol}_{year}"
    try:
        pages, (si, ai, pi, ci), parser, summary = parser_extract(pdf_path)
        scanned = pages is None
        if scanned:
            n = page_count(pdf_path)
            if n <= 0:
                mark(symbol, year, "failed", "SCANNED")
                return None
            log(f"  {symbol} {year}: no text layer — wide-scan page finding "
                f"({n} pages)")
            pages = []  # vision mode: no text hints, label filter lifted
            cands = []
        else:
            cands = candidate_pages(pages)
        verdict, units = arbitrate(symbol, year, pages, cands, pdf_path, tag,
                                   heur=(si, ai, pi, ci))
        if not verdict["sfp"] and ai is None:
            # no heuristic aud page to fall back on — retry arbitration once
            log(f"  {symbol} {year}: sfp empty & no heuristic aud — retrying arbitration")
            verdict, units = arbitrate(symbol, year, pages, cands, pdf_path, tag,
                                       heur=(si, ai, pi, ci))
        if not any(verdict.values()):
            log(f"  {symbol} {year}: no verdict from candidates — 4o-mini locator")
            n = len(pages) or page_count(pdf_path)
            if n > 0:
                v4 = pdf_locate_4omini(symbol, year, pdf_path, n, tag)
                if any(v4.values()):
                    # refine with nex on a window around the located pages
                    # (±5 absorbs printed-vs-PDF-index offset up to ~4 and
                    # catches both spread pages)
                    win = sorted({p for pg in v4["sfp"] + v4["pl"] + v4["cfs"]
                                  for p in range(max(0, pg - 5), min(n, pg + 6))})
                    log(f"  {symbol} {year}: nex refine window {[p + 1 for p in win]}")
                    verdict = nex_classify(symbol, year, pdf_path, win, tag)
                    if not any(verdict.values()):
                        verdict = v4
                    units = {}
                    scanned = True
                    pages = []
                else:
                    log(f"  {symbol} {year}: 4o-mini locator empty — wide-scan fallback")
                    verdict = wide_scan(symbol, year, n, pdf_path, tag)
                    units = {}  # nex classification has no units — extraction defaults to thousands
                    if any(verdict.values()):
                        scanned = True
                        pages = []
            else:
                mark(symbol, year, "failed", "SCANNED")
                return None
        if not any(verdict.values()):
            mark(symbol, year, "failed", "SCANNED" if scanned else "EMPTY_EXTRACTION")
            return None
        assert pages is not None  # vision mode sets pages=[]; never None here

        def spread(idx, r=1):
            return [p for p in range(idx - r, idx + r + 1) if 0 <= p < len(pages)]

        # BS page selection: heuristic aud page (ai) vs qwen sfp verdict.
        # qwen wins ONLY when its picks carry statement markers (TE_UPPER /
        # BS_START / thousands-units) — otherwise its verdict is highlights/
        # summary pages (ENGRO 2019 lesson) and the heuristic stays.
        def strong_markers(p):
            return TE_UPPER.search(p) or BS_START.search(p) or UNITS_K.search(p)

        vsfp = verdict["sfp"]
        if scanned:
            # vision mode: no text to verify markers — trust the VLM verdict
            bs_primary, bs_extra = vsfp, []
            pl_primary, pl_extra = verdict["pl"], []
            cfs_primary, cfs_extra = verdict["cfs"], []
        elif vsfp and ai is not None and ai not in vsfp and \
                any(strong_markers(pages[p]) for p in vsfp):
            strong = [p for p in vsfp if strong_markers(pages[p])]
            bs_primary = strong + [p for p in vsfp if p not in strong]
            bs_extra = []
            log(f"  {symbol} {year}: qwen sfp (marker-verified) replaces "
                f"heuristic aud p{ai + 1} -> {bs_primary}")
        elif vsfp and ai is not None and ai not in vsfp:
            bs_primary = [ai] + [p for p in spread(ai) if p != ai]
            bs_extra = []
            log(f"  {symbol} {year}: qwen sfp lacks statement markers "
                f"-> keep heuristic aud p{ai + 1}")
        elif ai is not None and strong_markers(pages[ai]):
            bs_primary = [ai] + [p for p in spread(ai) if p != ai]
            bs_extra = [p for p in vsfp if p not in bs_primary]
        elif ai is not None:
            # content-only heuristic aud = highlights/summary page, not a real
            # statement (image-based statement PDFs, ENGRO 2019 lesson)
            bs_primary = [p for p in vsfp if strong_markers(pages[p])]
            bs_extra = []
            if not bs_primary:
                log(f"  {symbol} {year}: heuristic aud p{ai + 1} lacks "
                    f"statement markers (image-based statements?) — clean fail")
        else:
            bs_primary = [p for p in vsfp if strong_markers(pages[p])]
            bs_extra = [p for p in vsfp if p not in bs_primary]
            if not bs_primary:
                log(f"  {symbol} {year}: no marker-verified SFP pages "
                    f"(verdict {vsfp}) — extraction will fail cleanly")
        # PL: qwen verdict wins only with units+content markers, else heuristic
        if not scanned:
            PL_STRONG = re.compile(r"Statement of Profit|Profit and Loss|Net sales|"
                                   r"profit after|Cost of sales|gross profit", re.I)
            CFS_STRONG = re.compile(r"Net cash|Dividends? paid", re.I)
            vpl = verdict["pl"]
            pl_page = vpl[0] if vpl else (pi if pi is not None else None)
            pi_strong = pi is not None and UNITS_K.search(pages[pi]) and PL_STRONG.search(pages[pi])
            if vpl and pi_strong and pi not in vpl and \
                    any(UNITS_K.search(pages[p]) and PL_STRONG.search(pages[p]) for p in vpl):
                strong = [p for p in vpl if UNITS_K.search(pages[p]) and PL_STRONG.search(pages[p])]
                pl_primary = strong + [p for p in vpl if p not in strong]
                pl_extra = []
            elif vpl and pi_strong and pi not in vpl:
                pl_primary = [pi] + [p for p in spread(pi) if p != pi]
                pl_extra = []
            elif pi_strong:
                pl_primary = [pi] + [p for p in spread(pi) if p != pi]
                pl_extra = [p for p in vpl if p not in pl_primary]
            else:
                pl_primary = [p for p in vpl if UNITS_K.search(pages[p]) and PL_STRONG.search(pages[p])]
                pl_extra = [p for p in vpl if p not in pl_primary]
            # CFS: same rule
            vcf = verdict["cfs"]
            cfs_page = vcf[0] if vcf else (ci if ci is not None else None)
            ci_strong = ci is not None and UNITS_K.search(pages[ci]) and CFS_STRONG.search(pages[ci])
            if vcf and ci_strong and ci not in vcf and \
                    any(UNITS_K.search(pages[p]) and CFS_STRONG.search(pages[p]) for p in vcf):
                strong = [p for p in vcf if UNITS_K.search(pages[p]) and CFS_STRONG.search(pages[p])]
                cfs_primary = strong + [p for p in vcf if p not in strong]
                cfs_extra = []
            elif vcf and ci_strong and ci not in vcf:
                cfs_primary = [ci] + [p for p in spread(ci) if p != ci]
                cfs_extra = []
            elif ci_strong:
                cfs_primary = [ci] + [p for p in spread(ci) if p != ci]
                cfs_extra = [p for p in vcf if p not in cfs_primary]
            else:
                cfs_primary = [p for p in vcf if UNITS_K.search(pages[p]) and CFS_STRONG.search(pages[p])]
                cfs_extra = [p for p in vcf if p not in cfs_primary]
        parser_cur = {k: v["cur"] for k, v in parser.items()}
        f, src = merge_fields(symbol, year, pdf_path, tag, pages,
                              bs_primary, bs_extra, pl_primary, pl_extra,
                              cfs_primary, cfs_extra, units, parser_cur=parser_cur)
        if not f:
            log(f"  {symbol} {year}: extraction EMPTY — failed")
            mark(symbol, year, "failed", "EMPTY_EXTRACTION")
            return None
        parser_prior = {k: v["prior"] for k, v in parser.items()}
        parser_sum = {k: v[0] for k, v in summary.items()}
        flags = run_qa(symbol, year, f, parser_cur, parser_sum)
        # disk-light: drop render PNGs for clean rows (keep flagged ones for review)
        if not flags:
            for stale in Path(f"{TMP_WORK}/png/{symbol}_{year}_").parent.glob(
                    f"{Path(f'{TMP_WORK}/png/{symbol}_{year}_').name}*"):
                try:
                    stale.unlink()
                except Exception:
                    pass
        row = {"symbol": symbol, "year": year,
               "TA": f.get("total_assets"), "TL": f.get("total_liabilities"),
               "Eq": f.get("total_equity"), "CA": f.get("current_assets"),
               "CL": f.get("current_liabilities"), "Sales": f.get("revenue"),
               "COGS": f.get("cost_of_sales"), "GP": f.get("gross_profit"),
               "PAT": f.get("net_income"), "Cash": f.get("cash"),
               "Inv": f.get("inventory"), "PPE": f.get("ppe"),
               "lt_investments": f.get("long_term_investments"),
               "dividends": f.get("dividends"),
               "dividends_paid": f.get("dividends_paid"), "cfo": f.get("operating_cash_flow"),
               "units": "thousands",
               "pages": json.dumps({"sfp": bs_primary + bs_extra,
                                    "pl": pl_primary + pl_extra,
                                    "cfs": cfs_primary + cfs_extra}),
               "flags": ";".join(x[0] for x in flags),
               "status": "review" if flags else "ok",
               "src": json.dumps(src), "ev": json.dumps(parser),
               "flag_details": flags}
        return row, parser_cur, parser_prior, flags
    except Exception as e:
        log(f"  {symbol} {year}: EXCEPTION {type(e).__name__}: {e}")
        return None


# ------------------------------------------------------------ finalize ---
FLD_MAP = {"TA": "total_assets", "TL": "total_liabilities", "Eq": "total_equity",
           "Sales": "revenue", "PAT": "net_income", "Cash": "cash",
           "PPE": "ppe"}


def l1_ok(d):
    ta, tl, eq = d.get("TA"), d.get("TL"), d.get("Eq")
    s, c, g = d.get("Sales"), d.get("COGS"), d.get("GP")
    ok1 = not (ta and tl and eq) or approx(ta, tl + eq)
    ok2 = not (s and c and g) or approx(s - abs(c), g)
    return ok1 and ok2


def repair_l1(rows, flags_out):
    """L1-repair: replace L1-breaking fields with the parser's own audited
    value (documented). Strips the repaired L1/L5 flags; adds a REPAIR row."""
    n = 0
    for (sym, y), d in rows.items():
        if l1_ok(d):
            continue
        try:
            ev = json.loads(d.get("ev", "{}"))
        except Exception:
            ev = {}
        for col, key in (("TA", "total_assets"), ("TL", "total_liabilities"),
                         ("Eq", "total_equity"), ("Sales", "revenue"),
                         ("COGS", "cost_of_sales"), ("GP", "gross_profit")):
            pv = ev.get(key, {}).get("cur")
            if pv is None or abs(pv) < 10000 or (1900 <= pv <= 2100):
                continue
            old = d.get(col)
            if old is not None and approx(old, pv):
                continue
            d[col] = pv
            if l1_ok(d):
                try:
                    src = json.loads(d.get("src", "{}"))
                except Exception:
                    src = {}
                src[key] = "parser-repair"
                d["src"] = json.dumps(src)
                d["flag_details"] = [fd for fd in d.get("flag_details") or []
                                     if fd[0] != "L1" and fd[1] != key]
                d["flags"] = ";".join(fd[0] for fd in d.get("flag_details") or [])
                flags_out.append((sym, y, "REPAIR", key, old, pv,
                                  "L1 repaired via parser audited value"))
                log(f"  L1-repair {sym} {y} {key}: {old} -> {pv}")
                n += 1
                break
            d[col] = old  # revert
    return n


def finalize():
    """L2 (cross-year overlap) + L3 (YoY) pass; write financials_annual.csv,
    qa_report.csv, review/ evidence files."""
    rows = {}
    for jf in sorted(ROWS_DIR.glob("*.json")):
        d = json.loads(jf.read_text())
        rows[(d["symbol"], d["year"])] = d
    if not rows:
        log("finalize: no cached rows")
        return
    by_comp = {}
    for (sym, y), d in rows.items():
        by_comp.setdefault(sym, {})[y] = d

    flags_out = []  # (symbol, year, layer, field, vlm, ref, detail)
    n_rep = repair_l1(rows, flags_out)
    for (sym, y), d in rows.items():
        for fd in d.get("flag_details") or []:
            if len(fd) >= 5:
                flags_out.append((sym, y, fd[0], fd[1], fd[2], fd[3], fd[4]))
            else:
                flags_out.append((sym, y, fd[0], fd[1], fd[2], None, ""))

    def add_flag(sym, y, layer, fld, v, ref, det):
        d = rows[(sym, y)]
        cur = d["flags"].split(";") if d["flags"] else []
        if layer not in cur:
            cur.append(layer)
            d["flags"] = ";".join(cur)
        flags_out.append((sym, y, layer, fld, v, ref, det))

    for sym, yrs in by_comp.items():
        for y in sorted(yrs):
            d = yrs[y]
            prev = yrs.get(y - 1)
            # L3: YoY sanity / exact repeat / negative equity
            if prev:
                for fld in ("TA", "Sales", "Eq"):
                    a, b = prev.get(fld), d.get(fld)
                    if a and b and a > 0:
                        if b / a > 5 or b / a < 0.2:
                            add_flag(sym, y, "L3", fld, b, a, f"YoY x{b/a:.1f}")
                        elif b == a:
                            add_flag(sym, y, "L3", fld, b, a, "exact repeat")
            if d.get("Eq") is not None and d["Eq"] < 0:
                add_flag(sym, y, "L3", "Eq", d["Eq"], None, "negative equity")
            # L2: year t value in report t+1's prior column
            nxt = yrs.get(y + 1)
            if nxt:
                for fld, key in FLD_MAP.items():
                    v = d.get(fld)
                    if v is None:
                        continue
                    try:
                        pv = json.loads(nxt.get("ev", "{}")).get(key, {}).get("prior")
                    except Exception:
                        pv = None
                    if pv is None or abs(pv) < 10000 or (1900 <= pv <= 2100):
                        continue
                    if approx(v, pv):
                        continue
                    # restatement? report y+1's prior (year y as restated) differs
                    # from report y's own current — but the VLM matched report y's
                    # as-reported value => real restatement, documented, keep
                    pc = None
                    try:
                        pc = json.loads(d.get("ev", "{}")).get(key, {}).get("cur")
                    except Exception:
                        pc = None
                    if pc is not None and approx(v, pc):
                        continue  # documented restatement — keep as-reported
                    add_flag(sym, y, "L2", key, v, pv, "prior-year overlap")

    csv_path = OUT / "financials_annual.csv"
    n_ok = n_rev = n_excl = 0
    l1_bad = {(sym, y) for (sym, y), d in rows.items() if not l1_ok(d)}
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PANEL_HDR)
        w.writeheader()
        for (sym, y) in sorted(rows):
            d = rows[(sym, y)]
            layers = d["flags"].split(";") if d["flags"] else []
            if (sym, y) in l1_bad:
                # L1 fails -> NOT in the panel (plan: review only)
                n_excl += 1
                rev = REVIEW_DIR / f"{sym}_{y}.json"
                evd = {"row": {k: d.get(k) for k in PANEL_HDR},
                       "flags": [{"layer": "L1", "field": "identity",
                                  "detail": "L1 failed after repair pass"}],
                       "evidence": d.get("ev", "{}")}
                REVIEW_DIR.mkdir(exist_ok=True)
                rev.write_text(json.dumps(evd, indent=1))
                continue
            if "L1" in layers:
                d["status"] = "review"  # defensive; repair should have cleared
                n_rev += 1
            elif layers:
                n_rev += 1
            else:
                n_ok += 1
            w.writerow({k: d.get(k, "") for k in PANEL_HDR})

    qa_path = OUT / "qa_report.csv"
    n_res = 0
    with open(qa_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["symbol", "year", "layer", "field", "vlm_value", "ref_value",
                    "detail", "resolution"])
        for sym, y, layer, fld, v, ref, det in sorted(set(flags_out),
            key=lambda x: (str(x[0] or ''), x[1] or 0, str(x[2] or ''),
                           str(x[3] or ''))):
            resolution = ""
            d = rows[(sym, y)]
            try:
                ev = json.loads(d.get("ev", "{}"))
            except Exception:
                ev = {}
            if layer == "L2.5":
                # known parser-summary misalignment: VLM matches the parser's
                # own audited current value -> summary table is the broken one
                pc = ev.get(fld, {}).get("cur")
                if pc is not None and approx(v, pc):
                    resolution = ("parser summary misaligned (known bug); "
                                  "audited value confirmed: VLM==parser")
            elif layer == "L5":
                if ref is not None and (abs(ref) < 10000 or (1900 <= ref <= 2100)):
                    resolution = ("parser value is a year token / index junk; "
                                  "VLM audited value accepted")
                elif approx(v, ref):
                    resolution = "stale flag (values agree within tol)"
                elif fld == "cost_of_sales" and ref is not None and \
                        approx(abs(v), abs(ref)):
                    resolution = ("COGS sign convention (negative-expense "
                                  "format); magnitudes agree")
                elif fld in ("revenue", "cost_of_sales", "gross_profit"):
                    # VLM's full PL trio identity-consistent -> parser read a
                    # notes-page/segment line (HUBC 2017 lesson)
                    s = d.get("Sales")
                    c = d.get("COGS")
                    g = d.get("GP")
                    if s and c and g and approx(s - abs(c), g):
                        resolution = ("VLM PL identity holds (Sales-|COGS|=GP); "
                                      "parser read a note/segment line")
            w.writerow([sym, y, layer, fld, v, ref, det, resolution])
            if resolution:
                n_res += 1
                continue  # resolved -> no review file
            rev = REVIEW_DIR / f"{sym}_{y}.json"
            evd = {}
            if rev.exists():
                try:
                    evd = json.loads(rev.read_text())
                except Exception:
                    evd = {}
            evd.setdefault("flags", []).append({"layer": layer, "field": fld,
                                                "vlm": v, "ref": ref,
                                                "detail": det})
            evd.setdefault("row", {k: rows[(sym, y)].get(k) for k in PANEL_HDR})
            evd.setdefault("evidence", rows[(sym, y)].get("ev", "{}"))
            REVIEW_DIR.mkdir(exist_ok=True)
            rev.write_text(json.dumps(evd, indent=1))
    log(f"finalize: {len(rows)} rows -> CSV ok={n_ok} review={n_rev}; "
        f"qa_report {len(set(flags_out))} flags ({n_res} auto-resolved)")
    return csv_path, qa_path


# ------------------------------------------------------------ validation ---
def run_validation():
    """4 companies, PDFs from /tmp/pk_e2e, compare vs p0_e2e_*_raw.json."""
    log("=== VALIDATION mode (4 companies) ===")
    totals = {"pairs": 0, "match": 0, "flags": 0, "trap_ok": 0, "trap_bad": 0}
    for sym in ["ABOT", "NML", "OGDC", "HUBC"]:
        gt_raw = json.loads((OUT / f"p0_e2e_{sym}_raw.json").read_text())
        gt = {int(k): v for k, v in gt_raw.items()}
        for y in sorted(gt):
            pdf = E2E_DIR / sym / f"{y}.pdf"
            if not pdf.exists():
                log(f"  {sym} {y}: PDF missing — skip")
                continue
            res = process_company_year(sym, y, pdf, gt=gt.get(y))
            if res is None:
                continue
            row, parser_cur, parser_prior, flags = res
            ROWS_DIR.mkdir(exist_ok=True)
            (ROWS_DIR / f"{sym}_{y}.json").write_text(json.dumps(row, indent=1))
            mark(sym, y, "done", ";".join(x[0] for x in flags))  # universe skips these
            for fld, key in (("TA", "total_assets"), ("TL", "total_liabilities"),
                             ("Eq", "total_equity"), ("CA", "current_assets"),
                             ("CL", "current_liabilities"), ("Sales", "revenue"),
                             ("GP", "gross_profit"), ("PAT", "net_income"),
                             ("Cash", "cash"), ("Inv", "inventory")):
                gv = gt.get(y, {}).get(key)
                vv = row.get(fld)
                if gv is None or vv is None:
                    continue
                if abs(gv) >= 10000:  # real statement values are >= 10,000k;
                    # index-table junk (100/1,130/1,060.49) stays below
                    totals["pairs"] += 1
                    if approx(vv, gv):
                        totals["match"] += 1
                    else:
                        log(f"  !! {sym} {y} {fld}: VLM={vv} GT={gv} MISMATCH")
                else:
                    if abs(vv) >= 10000:
                        totals["trap_ok"] += 1
                    else:
                        totals["trap_bad"] += 1
                        log(f"  !! {sym} {y} {fld}: VLM={vv} still trapped (GT={gv})")
            l1 = [f for f in flags if f[0] == "L1"]
            if l1:
                log(f"  !! {sym} {y}: L1 flags {l1}")
            totals["flags"] += len(flags)
            nf = sum(1 for k in PANEL_HDR if row.get(k) not in (None, ""))
            log(f"  {sym} {y}: fields={nf} flags={len(flags)}")
    rate = 100 * totals["match"] / max(totals["pairs"], 1)
    log(f"=== VALIDATION GATE: pairs={totals['pairs']} match={totals['match']} "
        f"({rate:.1f}%) trap_ok={totals['trap_ok']} trap_bad={totals['trap_bad']} "
        f"total_flags={totals['flags']}")
    return rate, totals


# ------------------------------------------------------------ universe ---
def liquid_universe(min_months=48):
    c = collections.Counter()
    with open(OUT / "processed" / "monthly_returns.csv") as f:
        for row in csv.DictReader(f):
            c[row["ticker"]] += 1
    return sorted(t for t, n in c.items() if n >= min_months)


def run_universe(workers=8, symbols=None, min_months=48):
    log(f"=== UNIVERSE mode: workers={workers} ===")
    syms = symbols or liquid_universe(min_months)
    log(f"universe: {len(syms)} liquid symbols")
    st = load_state()
    tasks = []
    for sym in syms:
        ann = company_annuals(sym)
        for rec in ann:
            key = f"{sym}|{rec['year']}"
            if st.get(key, {}).get("status") in ("done", "failed"):
                continue
            tasks.append((sym, rec["year"], rec["id"]))
    log(f"pending company-years: {len(tasks)}")
    if not tasks:
        log("nothing pending")
        return
    TMP_WORK.mkdir(exist_ok=True)
    (TMP_WORK / "png").mkdir(exist_ok=True)
    ROWS_DIR.mkdir(exist_ok=True)

    def work(t):
        sym, year, pdf_id = t
        if _stop.is_set():
            return None
        wd = TMP_WORK / sym
        wd.mkdir(parents=True, exist_ok=True)
        pdf = wd / f"{year}.pdf"
        if not download_pdf(sym, year, pdf_id, pdf):
            log(f"  {sym} {year}: download FAILED -> failed")
            mark(sym, year, "failed", "DOWNLOAD")
            return None
        res = process_company_year(sym, year, pdf)
        try:
            pdf.unlink()  # stream-extract-delete
        except Exception:
            pass
        if res is None:
            mark(sym, year, "failed", "PROCESS")
            return None
        row, _, _, flags = res
        row["status"] = "ok" if not flags else "review"
        (ROWS_DIR / f"{sym}_{year}.json").write_text(json.dumps(row, indent=1))
        mark(sym, year, "done", ";".join(x[0] for x in flags))
        return (sym, year, len(flags))

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(work, t): t for t in tasks}
        for fut in as_completed(futs):
            if _stop.is_set():
                log("stop event set — draining in-flight tasks")
                break
            try:
                r = fut.result()
                if r:
                    done += 1
                    if done % 25 == 0:
                        log(f"progress: {done}/{len(tasks)} done "
                            f"(cost ${_cost['total']:.3f})")
            except Exception as e:
                t = futs[fut]
                log(f"  worker exception {t}: {type(e).__name__}: {e}")
                mark(t[0], t[1], "failed", "EXCEPTION")
    log(f"universe run finished: {done} done, cost ${_cost['total']:.3f}")
    finalize()


# ------------------------------------------------------------ main ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--universe", action="store_true")
    ap.add_argument("--finalize", action="store_true")
    ap.add_argument("--cost", action="store_true")
    ap.add_argument("--symbols", type=str, default="")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--min-months", type=int, default=48)
    a = ap.parse_args()

    if a.cost:
        if COST_FILE.exists():
            print(json.dumps(json.loads(COST_FILE.read_text()), indent=1))
        else:
            print("no cost file yet")
        return
    if a.finalize:
        finalize()
        return
    if a.validate:
        run_validation()
        return
    if a.universe:
        syms = [s.strip() for s in a.symbols.split(",") if s.strip()] or None
        run_universe(workers=a.workers, symbols=syms, min_months=a.min_months)
        return
    ap.print_help()


if __name__ == "__main__":
    main()
