"""Voice prompts: Whisper turns speech into text, a small LLM turns the text into a DINO prompt.

Both models run on the MLA through LLiMa, in a sidecar process (voice/voice_server.py,
pyneat's OpenAI-compatible GenAIServer) that this module starts and stops. The sidecar
shares the MLA with dino_rt: each program runs in turn, so a transcription briefly delays
detection and the other way round.

"find the red cars and any dogs" -> Whisper -> LLM -> "red car . dog ."

When no LLM is loaded (or it answers nonsense), `heuristic_prompt` strips the request
words and splits on "and"/commas, which covers the common phrasings.
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections import deque
from pathlib import Path

from app.llima_hub import HubError, LlimaHub

DEFAULT_MODELS = Path(os.environ.get("DINO_LLIMA_MODELS", "/media/nvme/llima/models"))
ASR_CHOICES = ["whisper-small-a16w8"]
# Preferred first. Small instruction-following models: the answer is a few words.
# Gemma 4 E2B gets indirect requests right ("anything I could sit on" -> chair . sofa . ...) in
# 0.15-0.3 s; LFM2.5 230M answers in ~40 ms but misses those.
LLM_CHOICES = ["Gemma-4-E2B-it-TextOnly-GPTQ-a16w4", "LFM2.5-230M-a16w4"]
LLM_TITLES = {"Gemma-4-E2B-it-TextOnly-GPTQ-a16w4": "Gemma 4 E2B", "LFM2.5-230M-a16w4": "LFM2.5 230M"}

SYSTEM = ("You turn a spoken request into the input of an object detector. Reply with only the things "
          "to find, as short noun phrases in lower case, singular, keeping colours and other details, "
          "each followed by \" .\". Never add other words. If nothing is asked for, reply with \"none\".")
SHOTS = [
    ("can you find the red cars and any dogs", "red car . dog ."),
    ("where did I leave my glasses", "glasses ."),
    ("is there anything I could sit on", "chair . sofa . bench . stool ."),
    ("show me the people on bikes and also their helmets", "person . bicycle . helmet ."),
    ("uh, the big brown dog I think", "big brown dog ."),
    ("thanks, that's all", "none"),
]

_LEAD = re.compile(
    r"^(?:(?:hey|ok(?:ay)?|um+|uh+|so|well|please|now|and|also|then)\b[\s,]*)*"
    r"(?:(?:can|could|would|will) you(?: please)?\s+)?"
    r"(?:(?:please\s+)?(?:find|detect|show(?: me)?|look(?:ing)? for|search(?: for)?|spot|locate|highlight|"
    r"point out|get|mark|track|identify|where(?:'s| is| are)|is there|are there|i (?:want|need) to see|"
    r"i'm looking for|i am looking for|give me)\b\s*)?", re.I)
_DROP = re.compile(r"^(?:the|a|an|any|all(?: the)?|every|some|my|your|of|those|these|that|this)\s+", re.I)
_TAIL = re.compile(r"\s*(?:please|for me|in (?:the|this) (?:image|picture|photo|frame|scene)|here|"
                   r"i think|thanks?|thank you)\s*$", re.I)


def heuristic_prompt(text: str) -> str:
    """'Can you find the red cars and any dogs?' -> 'red cars . dogs .' (no LLM)."""
    t = re.sub(r"[?!]", " ", text.strip())
    t = _LEAD.sub("", t)
    parts = re.split(r"\s*(?:,|;|\.|\band also\b|\band\b|\bor\b|\bplus\b|\bas well as\b|&)\s*", t, flags=re.I)
    out = []
    for p in parts:
        p = _TAIL.sub("", _LEAD.sub("", p.strip()))
        while _DROP.match(p):
            p = _DROP.sub("", p)
        p = re.sub(r"[^\w\s'-]", "", p).strip().lower()
        if p and p not in out and len(p) <= 40:
            out.append(p)
    return " . ".join(out) + " ." if out else ""


def clean_llm_answer(answer: str) -> str:
    """Keep only 'a . b .' from what the LLM said; '' when it found nothing usable."""
    a = answer.strip().splitlines()[0] if answer.strip() else ""
    a = re.sub(r"^(?:answer|output|objects?|prompt)\s*:\s*", "", a, flags=re.I).strip().strip('"\'`')
    if not a or a.lower().rstrip(" .") == "none":
        return ""
    phrases = []
    for p in re.split(r"\s*[.,;]\s*", a):
        p = re.sub(r"[^\w\s'-]", "", p).strip().lower()
        if p and p not in phrases and len(p.split()) <= 5:
            phrases.append(p)
    return " . ".join(phrases) + " ." if phrases else ""


def _multipart(fields: dict, file_field: str, filename: str, data: bytes, mime: str) -> tuple[bytes, str]:
    b = uuid.uuid4().hex
    out = bytearray()
    for k, v in fields.items():
        out += f"--{b}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode()
    out += (f"--{b}\r\nContent-Disposition: form-data; name=\"{file_field}\"; filename=\"{filename}\"\r\n"
            f"Content-Type: {mime}\r\n\r\n").encode() + data + b"\r\n"
    out += f"--{b}--\r\n".encode()
    return bytes(out), f"multipart/form-data; boundary={b}"


class VoiceError(RuntimeError):
    pass


class VoiceService:
    """Starts/stops the sidecar and talks to it. Also the studio's PromptAssist (typed text)."""

    def __init__(self, models_dir: Path = DEFAULT_MODELS, python: str | None = None, port: int = 9987,
                 log=None, load_lock=None):
        self.models_dir = Path(models_dir)
        self.port = port
        self.python = python or self._find_python()
        self.script = Path(__file__).resolve().parents[1] / "voice" / "voice_server.py"
        self._log = log or (lambda s: None)
        self.lock = threading.Lock()          # one start/stop at a time
        # Held while the sidecar connects and loads: the MLA dispatcher refuses a client that
        # connects while another one is loading ("Unable to connect to the server from dispatcher").
        self.load_lock = load_lock or threading.Lock()
        self.proc: subprocess.Popen | None = None
        self.state = "off"                    # off | loading | ready | stopping | error
        self.error: str | None = None
        self.asr: str | None = None
        self.llm: str | None = None
        self.load_started = 0.0
        self.load_seconds: float | None = None
        self.log_lines: deque[str] = deque(maxlen=200)
        self.hub = LlimaHub(self.models_dir, log=self._log)   # downloads into the shared LLiMa folder

    @staticmethod
    def _find_python() -> str | None:
        for p in (os.environ.get("DINO_PYNEAT_PYTHON"), str(Path.home() / "pyneat/bin/python3")):
            if p and Path(p).exists():
                return p
        try:
            import pyneat  # noqa: F401
            return sys.executable
        except ImportError:
            return None

    # ---- what can be loaded
    def available(self) -> dict:
        def present(names):
            return [n for n in names if (self.models_dir / n).is_dir()]
        return {"asr": present(ASR_CHOICES), "llm": present(LLM_CHOICES), "python": bool(self.python),
                "models_dir": str(self.models_dir)}

    def status(self) -> dict:
        av = self.available()
        return {"state": self.state, "error": self.error, "asr": self.asr, "llm": self.llm,
                "llm_title": LLM_TITLES.get(self.llm or "", self.llm), "load_seconds": self.load_seconds,
                "elapsed": round(time.time() - self.load_started, 1) if self.state == "loading" else None,
                "can_start": bool(av["asr"] and av["python"]), "choices": av["llm"],
                "titles": {n: LLM_TITLES.get(n, n) for n in av["llm"]},
                "python": av["python"], "models_dir": av["models_dir"], "models": self.hub.models()}

    def download(self, action: str, name: str) -> None:
        try:
            {"download": self.hub.download, "cancel": self.hub.cancel, "discard": self.hub.discard}[action](name)
        except HubError as e:
            raise VoiceError(str(e)) from e

    # ---- PromptAssist
    def describe(self) -> dict:
        ok = self.state == "ready" and self.llm is not None
        return {"available": ok, "name": LLM_TITLES.get(self.llm or "", self.llm),
                "about": "Rewrites a request into things to find, with an LLM on the MLA (LLiMa)."}

    def rewrite(self, text: str) -> str:
        return self.to_prompt(text)["prompt"] or text

    # ---- lifecycle
    def start(self, llm: str | None = None, with_llm: bool = True) -> None:
        with self.lock:
            if self.state in ("loading", "ready"):
                return
            av = self.available()
            if not self.python:
                raise VoiceError("pyneat is not installed (needs ~/pyneat or DINO_PYNEAT_PYTHON)")
            if not av["asr"]:
                raise VoiceError(f"No Whisper model in {self.models_dir} (expected {ASR_CHOICES[0]})")
            self.asr = av["asr"][0]
            self.llm = None
            if with_llm and av["llm"]:
                self.llm = llm if llm in av["llm"] else av["llm"][0]
            cmd = [self.python, str(self.script), "--port", str(self.port), "--asr", str(self.models_dir / self.asr)]
            if self.llm:
                cmd += ["--llm", str(self.models_dir / self.llm)]
            self.state, self.error, self.load_seconds = "loading", None, None
            self.load_started = time.time()
            threading.Thread(target=self._launch, args=(cmd,), name="voice-load", daemon=True).start()

    def _launch(self, cmd: list[str]) -> None:
        for attempt in range(2):
            with self.load_lock:
                if self.state != "loading":           # turned off while waiting for the lock
                    return
                self._log(f"voice: starting {self.asr} + {self.llm or 'no LLM'}")
                self.log_lines.clear()
                settled = threading.Event()
                with self.lock:
                    self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                                 start_new_session=True, text=True, bufsize=1)
                threading.Thread(target=self._watch, args=(self.proc, settled), daemon=True).start()
                settled.wait(600)
            if self.state == "error" and attempt == 0 and any("Unable to connect" in s for s in self.log_lines):
                self._log("voice: the MLA dispatcher was busy; trying again")
                self.state, self.error = "loading", None
                time.sleep(3)
                continue
            return

    def _watch(self, proc: subprocess.Popen, settled: threading.Event) -> None:
        for line in proc.stdout:
            line = line.rstrip()
            self.log_lines.append(line)
            if line.startswith("VOICE ready") and proc is self.proc:
                self.state = "ready"
                self.load_seconds = round(time.time() - self.load_started, 1)
                self._log(f"voice: ready in {self.load_seconds} s")
                settled.set()
        rc = proc.wait()
        if proc is self.proc and self.state != "stopping":
            self.state = "error"
            tail = [s for s in self.log_lines if "rror" in s][-3:] or list(self.log_lines)[-3:]
            self.error = f"The voice models stopped (exit {rc}): " + " | ".join(tail)[-400:]
            self._log("voice: " + self.error)
        elif proc is self.proc:
            self.state = "off"
        settled.set()

    def stop(self, wait: bool = True, timeout: float = 30.0) -> None:
        with self.lock:
            p = self.proc
            if p is None or p.poll() is not None:
                self.state = "off"                    # also cancels a start still waiting for the lock
                return
            self.state = "stopping"
            p.send_signal(signal.SIGTERM)             # never SIGKILL: it would leak MLA memory
        if wait:
            try:
                p.wait(timeout)
            except subprocess.TimeoutExpired:
                self._log("voice: sidecar did not stop in time (left running; check the board)")
                return
            self.state = "off"

    # ---- requests
    def _post(self, path: str, body: bytes, ctype: str, timeout: float = 60.0) -> dict:
        if self.state != "ready":
            raise VoiceError("Voice is not loaded. Turn it on in Settings → Voice.")
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", body, {"Content-Type": ctype})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise VoiceError(f"voice server: {e.code} {e.read()[:300].decode('utf-8', 'replace')}") from e
        except (urllib.error.URLError, OSError) as e:
            raise VoiceError(f"voice server not reachable: {e}") from e

    def transcribe(self, audio: bytes, filename: str = "speech.webm", mime: str = "audio/webm",
                   language: str = "en") -> dict:
        body, ctype = _multipart({"model": "asr", "language": language, "stream": "false"}, "file",
                                 filename, audio, mime)
        t = time.time()
        r = self._post("/v1/audio/transcriptions", body, ctype)
        return {"text": (r.get("text") or "").strip(), "no_speech": float(r.get("no_speech_prob") or 0.0),
                "ms": round((time.time() - t) * 1e3, 1)}

    def to_prompt(self, text: str) -> dict:
        """Request text -> {'prompt': 'a . b .', 'by': 'llm' | 'rules', 'ms': ...}."""
        t = time.time()
        if self.state == "ready" and self.llm:
            msgs = [{"role": "system", "content": SYSTEM}]
            for u, a in SHOTS:
                msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
            msgs.append({"role": "user", "content": text})
            body = json.dumps({"model": "llm", "messages": msgs, "max_tokens": 40, "stream": False,
                               "chat_template_kwargs": {"enable_thinking": False}}).encode()
            try:
                r = self._post("/v1/chat/completions", body, "application/json")
                answer = r["choices"][0]["message"]["content"]
                p = clean_llm_answer(answer)
                if p or answer.strip().lower().rstrip(" .") == "none":
                    return {"prompt": p, "by": "llm", "answer": answer, "ms": round((time.time() - t) * 1e3, 1)}
            except (VoiceError, KeyError, IndexError) as e:
                self._log(f"voice: LLM rewrite failed ({e}); using rules")
        return {"prompt": heuristic_prompt(text), "by": "rules", "ms": round((time.time() - t) * 1e3, 1)}

    def close(self) -> None:
        self.stop(wait=True)
