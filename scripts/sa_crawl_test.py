#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P0.2 crawl test: Mubasher financial statements via Playwright."""
import asyncio, json, random, sys
from playwright.async_api import async_playwright

ROOT = "/home/ubuntu/research/dlap-tse"

def load_sample(n=10):
    quotes = json.load(open(f"{ROOT}/data_sa/universe_raw.json"))
    random.seed(11)
    return random.sample(quotes, n)

async def grab(page, base, code):
    url = f"{base}/markets/TDWL/stocks/{code}/financial-statements"
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=60000)
        await page.wait_for_timeout(6000)  # let Angular render + API settle
        # statements table header text
        body = await page.evaluate("document.body.innerText")
        if "Balance sheet" not in body and "الميزانية" not in body:
            return {"code": code, "status": "NO_TABLE", "url": url,
                    "snippet": body[:200]}
        # year/period selects
        selects = await page.evaluate("""() => Array.from(document.querySelectorAll('select')).map(s => ({
            name: s.name || s.id || '?',
            options: Array.from(s.options).map(o => o.text.trim())
        }))""")
        # rendered table rows (label + values)
        rows = await page.evaluate("""() => Array.from(document.querySelectorAll('table tr')).map(tr =>
            Array.from(tr.querySelectorAll('td,th')).map(c => c.innerText.trim())
        ).filter(r => r.length > 1)""")
        return {"code": code, "status": "OK", "url": url,
                "selects": selects, "rows": rows}
    except Exception as e:
        return {"code": code, "status": f"ERR {type(e).__name__}: {str(e)[:120]}", "url": url}

async def main():
    sample = load_sample()
    print(f"testing {len(sample)} stocks", flush=True)
    results = {}
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True,
            executable_path="/home/ubuntu/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome")
        ctx = await browser.new_context(
            user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
            locale="en-US")
        page = await ctx.new_page()
        for i, q in enumerate(sample):
            code = q["symbol"].replace(".SR", "")
            name = q.get("longName", "")[:25]
            r = await grab(page, "https://english.mubasher.info", code)
            if r["status"] != "OK":
                r2 = await grab(page, "https://www.mubasher.info", code)
                r2["ar_fallback"] = r["status"]
                r = r2
            r["name"] = name
            results[code] = r
            print(f"[{i+1}/10] {code} {name}: {r['status']}", flush=True)
        await browser.close()
    json.dump(results, open(f"{ROOT}/data_sa/p0_crawl_test.json", "w"), ensure_ascii=False, indent=1)
    ok = sum(1 for r in results.values() if r["status"] == "OK")
    print(f"\nOK: {ok}/10 -> data_sa/p0_crawl_test.json", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
