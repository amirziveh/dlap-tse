#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P0.2e: capture ALL /api/ calls incl. gfm.support host."""
import asyncio, json
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True,
            executable_path="/home/ubuntu/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome")
        ctx = await browser.new_context(
            user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36")
        page = await ctx.new_page()
        calls = []
        async def on_request(req):
            if "/api/" in req.url:
                calls.append({"m": req.method, "u": req.url, "h": dict(req.headers)})
        async def on_response(resp):
            if "/api/" in resp.url and resp.status == 200 and "initialData" not in resp.url:
                try:
                    body = await resp.json()
                    calls.append({"m": "BODY", "u": resp.url, "body": body})
                except Exception:
                    pass
        page.on("request", on_request)
        page.on("response", on_response)
        await page.goto("https://www.mubasher.info/markets/TDWL/stocks/3020/financial-statements",
                        wait_until="domcontentloaded", timeout=60000)
        await page.wait_for_timeout(9000)
        seen = set()
        for c in calls:
            u = c["u"]
            key = u.split("?")[0]
            if key in seen and c["m"] != "BODY":
                continue
            seen.add(key)
            if c["m"] == "BODY":
                d = c["body"]
                print(f"BODY {key[:110]}")
                print("   top keys:", list(d.keys())[:12] if isinstance(d, dict) else type(d))
                if isinstance(d, dict):
                    for k, v in d.items():
                        if isinstance(v, list) and v:
                            print(f"   {k}: list[{len(v)}]", "first:", json.dumps(v[0], ensure_ascii=False)[:220])
                        elif isinstance(v, dict):
                            print(f"   {k}: dict", json.dumps(v, ensure_ascii=False)[:220])
                        else:
                            print(f"   {k}: {str(v)[:100]}")
            else:
                print(f"{c['m']:5s} {key[:110]}  hdrs={ {k:v for k,v in c['h'].items() if k.lower() in ('authorization','x-api-key','origin','referer','content-type')} }")
        await browser.close()

asyncio.run(main())
