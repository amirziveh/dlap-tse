#!/usr/bin/env python3
"""
Download all articles listed in LITERATURE_LINKS.md to papers/.
v2: tries ALL urls per entry in priority order (arxiv > nber > ssrn > doi > other),
better HTML->PDF extraction for OA landing pages, fixed section detection.
"""
import re, os, csv, json, time, urllib.request, urllib.parse, subprocess, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.error import HTTPError, URLError

BASE = "/home/ubuntu/research/dlap-tse"
LINKS = os.path.join(BASE, "LITERATURE_LINKS.md")
PAPERS = os.path.join(BASE, "papers")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36"}
TIMEOUT = 50

def fetch(url, timeout=TIMEOUT, headers=None):
    h = dict(UA)
    if headers: h.update(headers)
    req = urllib.request.Request(url, headers=h)
    return urllib.request.urlopen(req, timeout=timeout)

def is_pdf(data):
    return data[:5] == b"%PDF-"

def save_pdf(data, path):
    if not is_pdf(data):
        return False, f"not a PDF (first bytes: {data[:8]!r})"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    return True, f"{len(data):,} bytes"

def openalex_oa(doi):
    try:
        url = f"https://api.openalex.org/works/doi:{urllib.parse.quote(doi)}"
        d = json.loads(fetch(url, timeout=25).read().decode())
        oa = d.get("open_access") or {}
        if oa.get("is_oa") and oa.get("oa_url"):
            return oa["oa_url"]
        return None
    except Exception:
        return None

def find_pdf_in_html(html, base_url):
    """Find a pdf link in an HTML page (many patterns)."""
    patterns = [
        r'href="([^"]+\.pdf[^"]*)"',
        r'href="([^"]*download[^"]*)"',
        r'href="([^"]*article[^"]*pdf[^"]*)"',
        r'src="([^"]+\.pdf[^"]*)"',
        r"href='([^']+\.pdf[^']*)'",
    ]
    for pat in patterns:
        for m in re.finditer(pat, html, re.I):
            u = m.group(1)
            if "javascript" in u.lower(): continue
            return urllib.parse.urljoin(base_url, u)
    return None

def download_pdf_from_url(url, path, depth=0):
    """Try to download a PDF from url (or an HTML landing page). Returns (ok, detail, final_url)."""
    if depth > 2:
        return False, "too many redirects", url
    try:
        data, final = fetch(url).read(), fetch(url).geturl()
    except Exception as ex:
        return False, f"fetch error: {ex}", url
    if is_pdf(data):
        ok, det = save_pdf(data, path)
        return ok, det, final
    # HTML: find pdf link
    html = data.decode("utf-8", "ignore")
    pdf_link = find_pdf_in_html(html, final)
    if pdf_link:
        try:
            data2, final2 = fetch(pdf_link).read(), fetch(pdf_link).geturl()
            if is_pdf(data2):
                ok, det = save_pdf(data2, path)
                return ok, det, final2
        except Exception:
            pass
    return False, "no PDF found on page", final

def parse_entries():
    text = open(LINKS, encoding="utf-8").read()
    entries = []
    section = "intl"
    for line in text.split("\n"):
        s = line.strip()
        if s.startswith("## "):
            sec = s.lower()
            if "iranian" in sec and "persian" in sec: section = "iran_fa"
            elif "iranian" in sec: section = "iran_en"
            else: section = "intl"
            continue
        if s.startswith("### "):
            sec = s.lower()
            if "persian" in sec: section = "iran_fa"
            elif "english" in sec or "iranian" in sec: section = "iran_en"
            continue
        m = re.match(r"^(\d+)\.\s+(.+?)\s+[-–—]\s+(.+)$", s)
        if not m:
            continue
        num = int(m.group(1))
        text_part = m.group(2).strip()
        url_part = m.group(3)
        urls = re.findall(r"https?://[^\s·]+", url_part)
        urls = [u.rstrip(".,;") for u in urls]
        # dedupe, keep order
        seen = set(); urls = [u for u in urls if not (u in seen or seen.add(u))]
        entries.append({"num": num, "text": text_part, "urls": urls, "section": section})
    return entries

def slugify(text, maxlen=60):
    s = re.sub(r"[^\w\- ]", "", text)
    s = re.sub(r"\s+", "_", s.strip())[:maxlen]
    return s or "paper"

def url_priority(url):
    d = urllib.parse.urlparse(url).netloc.lower()
    if "arxiv.org" in d: return 0
    if "nber.org" in d: return 1
    if "github.com" in d: return 2
    if "ssrn.com" in d: return 8  # manual
    if "scholar.google" in d or "noormags.ir/view/fa/search" in url: return 9
    if "noormags.ir" in d: return 7
    if "press.princeton" in d or "openassetpricing.com" in d: return 9
    if "doi.org" in d: return 3
    return 4

def process(entry):
    num, text, urls, section = entry["num"], entry["text"], entry["urls"], entry["section"]
    path_tmpl = os.path.join(PAPERS, section, f"{num:03d}_{slugify(text)}.pdf")
    # sort urls by priority
    urls = sorted(urls, key=url_priority)
    for url in urls:
        domain = urllib.parse.urlparse(url).netloc.lower()
        try:
            if "arxiv.org" in domain:
                pdf_url = url.replace("/abs/", "/pdf/") if "/abs/" in url else url
                ok, det, fin = download_pdf_from_url(pdf_url, path_tmpl)
                if ok: return ("downloaded", det, fin)
                return ("failed", f"arxiv: {det}", pdf_url)
            elif "nber.org" in domain:
                m = re.search(r"w(\d{5})", url)
                if m:
                    pdf_url = f"https://www.nber.org/system/files/working_papers/w{m.group(1)}/w{m.group(1)}.pdf"
                else:
                    pdf_url = url
                ok, det, fin = download_pdf_from_url(pdf_url, path_tmpl)
                if ok: return ("downloaded", det, fin)
                return ("failed", f"nber: {det}", pdf_url)
            elif "ssrn.com" in domain:
                return ("manual", "SSRN requires free account login", url)
            elif "scholar.google" in domain or "noormags.ir/view/fa/search" in url:
                return ("manual", "search page, not a document", url)
            elif "noormags.ir" in domain:
                m = re.search(r"/articlepage/(\d+)", url)
                if m:
                    return ("manual", "NoorMags PDF needs paid credit; page: " + url, url)
                return ("manual", "NoorMags search link", url)
            elif "press.princeton" in domain or "openassetpricing.com" in domain:
                return ("manual", "not a PDF (book/data site)", url)
            elif "github.com" in domain:
                repo = url.rstrip("/").split("github.com/")[-1]
                dest = os.path.join(PAPERS, "code", repo.replace("/", "_"))
                subprocess.run(["git", "clone", "--depth", "1", f"https://github.com/{repo}", dest],
                               check=False, capture_output=True, timeout=180)
                if os.path.isdir(dest) and os.listdir(dest):
                    return ("downloaded", "git clone", url)
                return ("failed", "git clone failed", url)
            elif "doi.org" in domain:
                doi = url.split("doi.org/")[-1]
                oa_url = openalex_oa(doi)
                if oa_url:
                    ok, det, fin = download_pdf_from_url(oa_url, path_tmpl)
                    if ok: return ("downloaded", f"OA: {det}", fin)
                    return ("failed", f"OA page no pdf ({oa_url})", oa_url)
                # try direct doi redirect (maybe OA without OpenAlex index)
                ok, det, fin = download_pdf_from_url(url, path_tmpl)
                if ok: return ("downloaded", f"direct: {det}", fin)
                return ("paywalled", "no OA version found", url)
            else:
                ok, det, fin = download_pdf_from_url(url, path_tmpl)
                if ok: return ("downloaded", det, fin)
                return ("failed", f"{domain}: {det}", url)
        except (HTTPError, URLError) as ex:
            continue
    return ("failed", "all URLs failed", "; ".join(urls[:2]) if urls else "none")

def main():
    entries = parse_entries()
    print(f"Parsed {len(entries)} entries")
    results = []
    lock = threading.Lock()
    def worker(e):
        st, det, src = process(e)
        with lock:
            results.append((e["num"], e["section"], e["text"], st, det, src))
            print(f"[{st.upper():10s}] #{e['num']:3d} [{e['section']:7s}] {e['text'][:50]}")
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = [ex.submit(worker, e) for e in entries]
        for f in as_completed(futs):
            pass
    with open(os.path.join(PAPERS, "status.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["num", "section", "title", "status", "detail", "url"])
        for r in sorted(results):
            w.writerow(r)
    from collections import Counter
    c = Counter(r[3] for r in results)
    with open(os.path.join(PAPERS, "download_report.md"), "w", encoding="utf-8") as f:
        f.write("# Download Report\n\n")
        f.write("| status | count |\n|---|---|\n")
        for k, v in c.most_common():
            f.write(f"| {k} | {v} |\n")
        f.write("\n## Manual / failed (need your action)\n\n")
        for r in sorted(r for r in results if r[3] in ("manual", "failed", "paywalled")):
            f.write(f"- **#{r[0]} [{r[1]}]** {r[2]} — {r[4]} ({r[5]})\n")
        f.write("\n## Downloaded\n\n")
        for r in sorted(r for r in results if r[3] == "downloaded"):
            f.write(f"- #{r[0]} [{r[1]}] {r[2]} — {r[5]}\n")
    print("\nDone. Report:", os.path.join(PAPERS, "download_report.md"))
    print(c)

if __name__ == "__main__":
    main()
