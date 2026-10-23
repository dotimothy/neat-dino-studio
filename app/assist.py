"""Prompt assist: the hook where an LLM (LLiMa on the board) can rewrite what the user typed.

Grounding DINO wants short noun phrases ("chair . sofa ."); people type requests
("anything I could sit on"). An assist turns the second into the first before the
prompt is tokenized. Nothing ships today: NullAssist passes the text through and
reports itself unavailable, which keeps the UI's Assist control hidden.

To add one later, implement `PromptAssist` (e.g. calling a LLiMa GenAIServer's
OpenAI-compatible /v1/chat/completions on 127.0.0.1:9998, or pyneat.genai in a
worker process) and set `Studio.assist`. Keep in mind that the detector and the
LLM then share the MLA's memory (CMA): load the smallest model that does the job.
"""
from __future__ import annotations

from typing import Protocol


class PromptAssist(Protocol):
    def describe(self) -> dict: ...
    def rewrite(self, text: str) -> str: ...


class NullAssist:
    """No assist: the text is used as typed."""

    def describe(self) -> dict:
        return {"available": False, "name": None,
                "about": "Prompt assist (an LLM via LLiMa) is not part of this build."}

    def rewrite(self, text: str) -> str:
        return text
