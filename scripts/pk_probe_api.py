#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PK-P0.1: capture PSX data portal API + universe + price depth."""
import asyncio, json
from playwright.async_api import async_playwright

API_CALLS = []
ROWS = []

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True,
            executable_path="/home/ubuntu/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome")
        ctx = await browser.new_context(
            user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36")
        page = await ctx.new_page()

        async def on_req(req):
            if any(k in req.url for k in ("api", "data", "symbol", "list", "json", "download")) and "psx.com.pk" in req.url:
                API_CALLS.append({"m": req.method, "u": req.url,
                                  "post": req.post_data[:300] if req.post_data else None})

        page.on("request", on_req)
        await page.goto("https://dps.psx.com.pk/historical", wait_until="domcontentloaded", timeout=60000)
        await page.wait_for_timeout(9000)

        # dump the grid rows from the DOM (React grid)
        rows = await page.evaluate("""() => {
            const out = [];
            document.querySelectorAll('[role=row]').forEach(r => {
                const cells = Array.from(r.querySelectorAll('[role=gridcell], [role=columnheader]')).map(c => c.innerText.trim());
                if (cells.length) out.push(cells);
            });
            return out.slice(0, 8);
        }""")
        print("=== grid rows ===")
        for r in rows:
            print(" |", " | ".join(c[:14] for c in r))

        # page HTML embedded data check
        html = await page.content()
        has_data = "TSBL" in html
        print("TSBL in html:", has_data)

        print("\n=== API calls ===")
        seen = set()
        for c in API_CALLS:
            k = c["u"].split("?")[0]
            if k in seen: continue
            seen.add(k)
            print(f"{c['m']:4s} {c['u'][:150]}")
            if c["post"]: print(f"      POST: {c['post'][:200]}")
        await browser.close()

asyncio.run(main())
