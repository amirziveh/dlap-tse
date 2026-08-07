#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P0.2d: dump initialData payload structure for 3020."""
import asyncio, json
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True,
            executable_path="/home/ubuntu/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome")
        ctx = await browser.new_context(
            user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36")
        page = await ctx.new_page()
        payload = {}
        async def on_response(resp):
            if "initialData" in resp.url and resp.status == 200:
                try:
                    payload['data'] = await resp.json()
                except Exception as e:
                    payload['err'] = str(e)
        page.on("response", on_response)
        await page.goto("https://www.mubasher.info/markets/TDWL/stocks/3020/financial-statements",
                        wait_until="domcontentloaded", timeout=60000)
        await page.wait_for_timeout(8000)
        d = payload.get('data')
        if not d:
            print("no payload:", payload.get('err')); return
        def keys(o, prefix="", depth=0):
            out = []
            if depth > 3: return out
            if isinstance(o, dict):
                for k, v in o.items():
                    out.append(f"{prefix}{k}: {type(v).__name__}" + (f" ({len(v)})" if isinstance(v, (list, dict)) else ""))
                    if isinstance(v, dict): out += keys(v, prefix+"  ", depth+1)
                    elif isinstance(v, list) and v and isinstance(v[0], dict): out += keys(v[0], prefix+"  [0] ", depth+1)
            return out
        print("=== top-level keys ===")
        for line in keys(d)[:40]:
            print(line)
        json.dump(d, open("/home/ubuntu/research/dlap-tse/data_sa/p0_initialData_3020.json","w"), ensure_ascii=False)
        print("\nsaved p0_initialData_3020.json")
        await browser.close()

asyncio.run(main())
