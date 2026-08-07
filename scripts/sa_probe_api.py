#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P0.2c: intercept API calls + test annual period on Yamama (3020)."""
import asyncio, json
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True,
            executable_path="/home/ubuntu/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome")
        ctx = await browser.new_context(
            user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36")
        page = await ctx.new_page()
        api_calls = []
        page.on("request", lambda r: api_calls.append((r.method, r.url, dict(r.headers).get('authorization','')[:20])) if 'mubasher.info/api' in r.url else None)
        page.on("response", lambda r: api_calls.append(("RESP", r.status, r.url)) if 'mubasher.info/api' in r.url and r.request.method != 'OPTIONS' else None)
        await page.goto("https://www.mubasher.info/markets/TDWL/stocks/3020/financial-statements",
                        wait_until="domcontentloaded", timeout=60000)
        await page.wait_for_timeout(8000)
        print("=== API calls ===")
        seen = set()
        for m, u, h in api_calls:
            if isinstance(u, int):
                print(f"RESP   {u} {h[:160]}")
                continue
            key = (m, u.split('?')[0])
            if key not in seen:
                seen.add(key)
                print(f"{m:6s} {u[:160]}  auth={h}")
        # select annual period
        try:
            await page.select_option("select", label="الميزانية السنوية")
            await page.wait_for_timeout(6000)
            rows = await page.evaluate("""() => Array.from(document.querySelectorAll('table tr')).map(tr =>
                Array.from(tr.querySelectorAll('td,th')).map(c => c.innerText.trim())
            ).filter(r => r.length > 1)""")
            print(f"\n=== annual view: {len(rows)} rows ===")
            for r in rows:
                print(' | '.join(c[:26] for c in r[:5]))
        except Exception as e:
            print("select annual failed:", e)
        await browser.close()

asyncio.run(main())
