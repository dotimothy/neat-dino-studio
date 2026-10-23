#!/usr/bin/env python3
"""End-to-end voice check in Chrome: a WAV file plays as the microphone, the script holds the
Speak button, and the prompt must fill in and a detection run.

    .venv/bin/python tools/voice_ui_check.py https://<board>:8092/ speech.wav image.jpg [out_dir]
"""
import asyncio
import sys
from pathlib import Path

from playwright.async_api import async_playwright

URL, WAV, IMG = sys.argv[1:4]
OUT = Path(sys.argv[4] if len(sys.argv) > 4 else "build/shots")


async def main():
    OUT.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as p:
        b = await p.chromium.launch(channel="chrome", headless=True, args=[
            "--ignore-certificate-errors", "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
            f"--use-file-for-fake-audio-capture={Path(WAV).resolve()}%noloop"])
        ctx = await b.new_context(viewport={"width": 1440, "height": 900}, ignore_https_errors=True, color_scheme="dark",
                                  permissions=["microphone"])
        pg = await ctx.new_page()
        pg.on("console", lambda m: print("console:", m.text) if m.type in ("error", "warning") else None)
        await pg.goto(URL)
        await pg.wait_for_timeout(1500)
        await pg.set_input_files("#file", IMG)
        await pg.fill("#prompt", "")
        await pg.wait_for_timeout(500)
        btn = await pg.query_selector("#mic-btn")
        box = await btn.bounding_box()
        await pg.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        await pg.mouse.down()
        await pg.wait_for_timeout(1500)
        await pg.screenshot(path=str(OUT / "voice_listening.png"))
        await pg.wait_for_timeout(2500)
        await pg.mouse.up()
        await pg.wait_for_function("document.querySelector('#prompt').value.length > 0", timeout=30000)
        print("prompt:", await pg.input_value("#prompt"))
        print("voice line:", (await pg.inner_text("#voice-line")).replace("\n", " "))
        await pg.wait_for_selector("#det-list li:not(.empty)", timeout=60000)
        await pg.wait_for_timeout(800)
        print("detections:", await pg.inner_text("#det-count"))
        await pg.screenshot(path=str(OUT / "voice_result.png"))
        await pg.click("#settings-btn")
        await pg.click(".nav-item[data-pane=voice]")
        await pg.wait_for_timeout(500)
        await pg.screenshot(path=str(OUT / "voice_settings.png"))
        await b.close()


asyncio.run(main())
