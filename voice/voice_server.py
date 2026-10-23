#!/usr/bin/env python3
"""Voice sidecar for Neat DINO Studio: Whisper (speech to text) and a small LLM on the MLA.

Runs LLiMa models through pyneat's GenAIServer, an OpenAI-compatible HTTP server:

    POST /v1/audio/transcriptions   multipart: file, model=asr, language
    POST /v1/chat/completions       {"model": "llm", "messages": [...], "max_tokens": ...}

It is a separate process so the web app stays up if a model fails to load, and so that
stopping it (SIGTERM, never SIGKILL) frees its MLA memory. The studio starts and stops it
(app/voice.py); run it by hand with the pyneat environment:

    ~/pyneat/bin/python3 voice/voice_server.py --asr /media/nvme/llima/models/whisper-small-a16w8 \\
        --llm /media/nvme/llima/models/LFM2.5-230M-a16w4 --port 9987
"""
import argparse
import signal
import sys
import threading
import time


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9987)
    ap.add_argument("--asr", help="Whisper model directory (LLiMa)")
    ap.add_argument("--llm", help="LLM model directory (LLiMa)")
    args = ap.parse_args()
    if not (args.asr or args.llm):
        ap.error("give --asr and/or --llm")

    import pyneat as neat

    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())

    opts = neat.genai.GenAIServerOptions()
    opts.host, opts.port = args.host, args.port
    server = neat.genai.GenAIServer(opts)
    if args.asr:
        server.add_model(args.asr, "asr")
    if args.llm:
        server.add_model(args.llm, "llm")
    t0 = time.time()
    print(f"VOICE loading {', '.join(server.model_names())}", flush=True)
    server.start()                                      # loads and warms every model
    print(f"VOICE ready {time.time() - t0:.1f} s on http://{args.host}:{args.port}", flush=True)
    while not stop.wait(0.5):
        pass
    print("VOICE stopping", flush=True)
    server.stop()
    # stop() only ends serving: the models stay allocated in the MLA service, and a process
    # that exits with them loaded leaves that memory allocated (about 15 GB for Whisper + Gemma).
    for name in list(server.model_names()):
        server.remove_model(name)
    del server
    import gc
    gc.collect()
    print("VOICE stopped", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
