#!/usr/bin/env python3
"""Try the prompt-rewrite instructions on a running voice sidecar: python3 llm_probe.py [port]."""
import json
import sys
import time
import urllib.request

sys.path.insert(0, __file__.rsplit("/", 2)[0])
from app.voice import SYSTEM, SHOTS  # noqa: E402

port = int(sys.argv[1]) if len(sys.argv) > 1 else 9987
tests = ["can you find the red cars and any dogs", "where are my keys", "show me people wearing hats",
         "is there anything I could sit on", "detect the laptop, the coffee mug and the phone",
         "look for a cat", "Find all the traffic lights and stop signs.", "um, the blue backpack I think"]
for t in tests:
    msgs = [{"role": "system", "content": SYSTEM}]
    for u, a in SHOTS:
        msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    msgs.append({"role": "user", "content": t})
    body = json.dumps({"model": "llm", "max_tokens": 40, "messages": msgs,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    s = time.time()
    r = urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body,
                                                      {"Content-Type": "application/json"}), timeout=60)
    out = json.loads(r.read())["choices"][0]["message"]["content"]
    print(f"{time.time() - s:5.2f} s  {t!r:50} -> {out!r}")
