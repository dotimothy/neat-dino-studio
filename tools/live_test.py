#!/usr/bin/env python3
"""Smoke test of the webcam Live mode with Chrome's fake camera (Playwright).

    .venv/bin/python tools/live_test.py [http://localhost:8092/]
"""
import asyncio
import sys

from playwright.async_api import async_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8092/"


async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(channel="chrome", headless=True, args=[
            "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream", "--ignore-certificate-errors"])
        ctx = await b.new_context(viewport={"width": 1280, "height": 800}, color_scheme="dark", ignore_https_errors=True)
        await ctx.grant_permissions(["camera"])
        pg = await ctx.new_page()
        pg.on("console", lambda m: print("console:", m.text) if m.type == "error" else None)
        await pg.goto(URL)
        await pg.fill("#prompt", "circle . ball .")
        await pg.click("#cam-btn")
        await pg.wait_for_timeout(1500)
        await pg.click("#live-btn")
        await pg.wait_for_timeout(8000)
        print("fps:", await pg.text_content("#fps"))
        print("live button:", await pg.text_content("#live-btn span"))
        await pg.screenshot(path="build/shots/live.png")
        await pg.click("#live-btn")
        await pg.click("#cam-btn")
        await b.close()


asyncio.run(main())
