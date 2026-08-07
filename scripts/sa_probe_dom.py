#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P0.2f: find where statements data lives in the live DOM."""
import asyncio, json, re
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True,
            executable_path="/home/ubuntu/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome")
        ctx = await browser.new_context(
            user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36")
        page = await ctx.new_page()
        await page.goto("https://www.mubasher.info/markets/TDWL/stocks/3020/financial-statements",
                        wait_until="domcontentloaded", timeout=60000)
        await page.wait_for_timeout(8000)
        # 1) where does the value 7316831.953 live?
        info = await page.evaluate("""() => {
            const probe = '7316831';
            const hits = [];
            document.querySelectorAll('script').forEach((s, i) => {
                const t = s.textContent || '';
                if (t.includes(probe)) {
                    hits.push({script: i, type: s.type || 'js', len: t.length,
                               snippet: t.slice(Math.max(0, t.indexOf(probe)-150), t.indexOf(probe)+150)});
                }
            });
            // also check non-script text nodes
            const bodyText = document.body.innerHTML;
            return {scriptHits: hits, inBodyHtml: bodyText.includes(probe)};
        }""")
        print(json.dumps(info, ensure_ascii=False)[:900])
        # 2) window globals that look like data containers
        w = await page.evaluate("""() => {
            const out = [];
            for (const k in window) {
                try {
                    const v = window[k];
                    if (v && typeof v === 'object' && JSON.stringify(v).length > 5000) out.push(k);
                } catch(e) {}
            }
            return out.slice(0, 15);
        }""")
        print("big window objects:", w)
        await browser.close()

asyncio.run(main())
