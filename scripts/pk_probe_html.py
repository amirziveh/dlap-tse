#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PK-P0.1b: find where the daily table data lives in the rendered page."""
import asyncio, json
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True,
            executable_path="/home/ubuntu/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome")
        ctx = await browser.new_context(
            user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36")
        page = await ctx.new_page()
        await page.goto("https://dps.psx.com.pk/historical", wait_until="domcontentloaded", timeout=60000)
        await page.wait_for_timeout(8000)
        html = await page.content()
        i = html.find("TSBL")
        print("TSBL at:", i, "of", len(html))
        if i > 0:
            print(repr(html[max(0,i-200):i+150]))
        # all script srcs
        srcs = await page.evaluate("Array.from(document.querySelectorAll('script[src]')).map(s => s.src)")
        print("\nscripts:", [s for s in srcs if 'psx' in s or 'static' in s][:6])
        # any inline script with data
        big = await page.evaluate("""() => Array.from(document.querySelectorAll('script:not([src])')).filter(s => s.textContent.length > 2000).map(s => s.textContent.slice(0, 80))""")
        print("inline scripts >2KB:", len(big))
        await browser.close()

asyncio.run(main())
