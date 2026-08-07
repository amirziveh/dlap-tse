#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P0.2h: robust parse of embedded statements."""
import asyncio, json, re, ast
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True,
            executable_path="/home/ubuntu/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome")
        ctx = await browser.new_context(
            user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36")
        page = await ctx.new_page()
        for code in ("3020", "7010"):
            await page.goto(f"https://www.mubasher.info/markets/TDWL/stocks/{code}/financial-statements",
                            wait_until="domcontentloaded", timeout=60000)
            await page.wait_for_timeout(6000)
            raw = await page.evaluate("""() => {
                for (const s of document.querySelectorAll('script')) {
                    const t = s.textContent || '';
                    if (t.includes("'values'") && t.includes('records')) return t;
                }
                return null;
            }""")
            if not raw:
                print(code, "not found"); continue
            # balanced-brace extraction of the first array literal
            start = raw.find("[{")
            depth = 0; i = start; instr = False; esc = False
            while i < len(raw):
                c = raw[i]
                if instr:
                    if esc: esc = False
                    elif c == "\\": esc = True
                    elif c == "'": instr = False
                else:
                    if c == "'": instr = True
                    elif c == "[": depth += 1
                    elif c == "]":
                        depth -= 1
                        if depth == 0: break
                i += 1
            txt = raw[start:i+1].encode().decode('unicode_escape', errors='ignore')
            # JS object literal -> Python: single quotes -> double quotes
            txt = re.sub(r"'", '"', txt)
            try:
                arr = json.loads(txt)
            except Exception as e:
                print(code, "parse err:", e); print("ctx:", txt[600:720]); continue
            print(f"\n===== {code}: {len(arr)} period(s) =====")
            all_labels = set()
            for per in arr:
                print(f"period: {per['label']}")
                for sec in per.get('sections', []):
                    recs = sec.get('records', [])
                    yrs = set()
                    for r in recs:
                        yrs.update(r.get('values', {}).keys())
                        all_labels.add(r['label'])
                    print(f"  section: {sec['label']}  ({len(recs)} records, years {sorted(yrs)})")
            print(f"\nALL record labels ({len(all_labels)}):")
            for l in sorted(all_labels):
                print("   -", l)
        await browser.close()

asyncio.run(main())
