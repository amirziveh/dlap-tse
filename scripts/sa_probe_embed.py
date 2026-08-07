#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P0.2g: dump full embedded statements script + probe year param."""
import asyncio, json, re
from playwright.async_api import async_playwright

async def grab(page, url, tag):
    await page.goto(url, wait_until="domcontentloaded", timeout=60000)
    await page.wait_for_timeout(6000)
    data = await page.evaluate("""() => {
        for (const s of document.querySelectorAll('script')) {
            const t = s.textContent || '';
            if (t.includes("'values'") && t.includes('label')) {
                // try to extract the array literal
                const m = t.match(/\\[\\s*\\{[^]*?\\}\\]/);
                return m ? m[0] : t.slice(0, 500);
            }
        }
        return null;
    }""")
    return data

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True,
            executable_path="/home/ubuntu/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome")
        ctx = await browser.new_context(
            user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36")
        page = await ctx.new_page()
        for url, tag in [
            ("https://www.mubasher.info/markets/TDWL/stocks/3020/financial-statements", "3020 default"),
            ("https://www.mubasher.info/markets/TDWL/stocks/3020/financial-statements?year=2016", "3020 year=2016"),
        ]:
            raw = await grab(page, url, tag)
            print(f"\n===== {tag} =====")
            if not raw:
                print("no embedded statements found"); continue
            open(f"/tmp/stmt_{tag.split()[0]}_{tag.split()[1].replace('=','')}.txt","w").write(raw)
            # decode unicode escapes and parse the array
            txt = raw.encode().decode('unicode_escape', errors='ignore')
            m = re.search(r'\[\s*\{.*', txt, re.S)
            if m:
                try:
                    arr = json.loads(m.group(0).replace("'", '"'))
                    print(f"parsed: {len(arr)} rows")
                    for row in arr:
                        vals = row.get('values', {})
                        yrs = sorted(vals.keys())
                        print(f"  {row.get('label','')[:40]:42s} years={yrs[0] if yrs else '-'}..{yrs[-1] if yrs else '-'} n={len(yrs)}")
                except Exception as e:
                    print("parse err:", e, "| sample:", txt[:400])
        await browser.close()

asyncio.run(main())
