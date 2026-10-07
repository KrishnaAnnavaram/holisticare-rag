"""Chat model adapters. Standard library HTTP only. Temperature and the token cap come from settings.

* ``OllamaChat``: ``POST /api/chat`` with ``options.temperature`` and ``options.num_predict``.
* ``OpenAICompatChat``: ``POST /chat/completions`` on any OpenAI-compatible server.
* ``ScriptedChat``: a fake for tests.

The offline default is not a chat model: ``answer.extractive_answer`` builds the answer from
the passages.
"""
from __future__ import annotations

import json
import os
import urllib.request
from typing import Callable, Protocol

Post = Callable[[str, dict, dict], dict]


def _http_post(url: str, body: dict, headers: dict, timeout: float = 60.0) -> dict:  # pragma: no cover - network
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


class ChatModel(Protocol):
    name: str

    def complete(self, messages: list[dict]) -> str: ...


class OllamaChat:
    def __init__(self, model: str, base_url: str | None = None, temperature: float = 0.1, max_tokens: int = 512,
                 timeout_s: float = 60.0, post: Post | None = None):
        self.url = (base_url or "http://localhost:11434").rstrip("/") + "/api/chat"
        self.model, self.temperature, self.max_tokens, self.timeout = model, temperature, max_tokens, timeout_s
        self._post = post or (lambda u, b, h: _http_post(u, b, h, self.timeout))
        self.name = f"ollama:{model}"

    def payload(self, messages: list[dict]) -> dict:
        return {"model": self.model, "messages": messages, "stream": False,
                "options": {"temperature": self.temperature, "num_predict": self.max_tokens}}

    def complete(self, messages):
        return self._post(self.url, self.payload(messages), {})["message"]["content"]


class OpenAICompatChat:
    def __init__(self, model: str, base_url: str | None = None, temperature: float = 0.1, max_tokens: int = 512,
                 timeout_s: float = 60.0, api_key: str | None = None, post: Post | None = None):
        key = api_key or os.environ.get("HOLISTICARE_LLM_API_KEY")
        if not key:
            raise ValueError("set HOLISTICARE_LLM_API_KEY for HOLISTICARE_LLM=openai")
        self._key = key
        self.url = (base_url or "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
        self.model, self.temperature, self.max_tokens, self.timeout = model, temperature, max_tokens, timeout_s
        self._post = post or (lambda u, b, h: _http_post(u, b, h, self.timeout))
        self.name = f"openai:{model}"

    def complete(self, messages):
        body = {"model": self.model, "messages": messages, "temperature": self.temperature, "max_tokens": self.max_tokens}
        out = self._post(self.url, body, {"Authorization": f"Bearer {self._key}"})
        return out["choices"][0]["message"]["content"]


class ScriptedChat:
    def __init__(self, replies: list[str], name: str = "scripted"):
        self.replies, self.name = list(replies), name
        self.calls: list[list[dict]] = []

    def complete(self, messages):
        self.calls.append(messages)
        return self.replies.pop(0) if self.replies else ""
