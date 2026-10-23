#!/usr/bin/env python3
"""Screenshots of the studio page (Playwright + the system Chrome): main view, results, settings."""
import asyncio
import sys
from pathlib import Path

from playwright.async_api import async_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8092/"
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "build/shots")
IMG = sys.argv[3] if len(sys.argv) > 3 else "data/images/cats.jpg"
PROMPT = sys.argv[4] if len(sys.argv) > 4 else "a cat . a remote control . couch ."


async def main():
    OUT.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as p:
        b = await p.chromium.launch(channel="chrome", headless=True, args=["--ignore-certificate-errors"])
        for theme in ("dark", "light"):
            ctx = await b.new_context(viewport={"width": 1440, "height": 900}, device_scale_factor=1,
                                      color_scheme=theme, ignore_https_errors=True)
            pg = await ctx.new_page()
            pg.on("console", lambda m: print("console:", m.text) if m.type in ("error", "warning") else None)
            await pg.goto(URL)
            await pg.wait_for_timeout(800)
            await pg.screenshot(path=str(OUT / f"empty_{theme}.png"))
            await pg.fill("#prompt", PROMPT)
            await pg.set_input_files("#file", IMG)
            await pg.wait_for_selector("#det-list li:not(.empty)", timeout=60000)
            await pg.wait_for_timeout(600)
            await pg.screenshot(path=str(OUT / f"result_{theme}.png"))
            await pg.click("#settings-btn")
            await pg.wait_for_timeout(900)
            await pg.screenshot(path=str(OUT / f"models_{theme}.png"))
            await ctx.close()
        ctx = await b.new_context(viewport={"width": 390, "height": 844}, device_scale_factor=2, color_scheme="dark")
        pg = await ctx.new_page()
        await pg.goto(URL)
        await pg.fill("#prompt", PROMPT)
        await pg.set_input_files("#file", IMG)
        await pg.wait_for_selector("#det-list li:not(.empty)", timeout=60000)
        await pg.wait_for_timeout(600)
        await pg.screenshot(path=str(OUT / "phone_dark.png"), full_page=True)
        await b.close()


asyncio.run(main())
