#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P0.2b probe: find year selector + full field list on Al Rajhi (1120)."""
import asyncio, json
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True,
            executable_path="/home/ubuntu/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome")
        ctx = await browser.new_context(
            user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36")
        page = await ctx.new_page()
        await page.goto("https://www.mubasher.info/markets/TDWL/stocks/1120/financial-statements",
                        wait_until="domcontentloaded", timeout=60000)
        await page.wait_for_timeout(7000)
        # 1) find clickable year elements
        yrs = await page.evaluate("""() => {
            const out = [];
            document.querySelectorAll('*').forEach(el => {
                const t = (el.innerText||'').trim();
                if (/^20(1[0-9]|2[0-6])$/.test(t) && el.children.length === 0 && t.length <= 4) {
                    const r = el.getBoundingClientRect();
                    if (r.width > 0 && r.height > 0)
                        out.push({tag: el.tagName, cls: (el.className||'').toString().slice(0,60), text: t, x: r.x, y: r.y});
                }
            });
            return out.slice(0, 20);
        }""")
        print("year elements:", json.dumps(yrs, ensure_ascii=False)[:600])
        # 2) full table rows (ALL fields)
        rows = await page.evaluate("""() => Array.from(document.querySelectorAll('table tr')).map(tr =>
            Array.from(tr.querySelectorAll('td,th')).map(c => c.innerText.trim())
        ).filter(r => r.length > 1)""")
        print(f"\nfull table: {len(rows)} rows")
        for r in rows:
            print(' | '.join(c[:26] for c in r))
        # 3) try clicking a year element if found (e.g. first with 2019)
        await browser.close()

asyncio.run(main())
