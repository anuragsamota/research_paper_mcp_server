"""A scripted stand-in for an Ollama server (httpx MockTransport)."""

from __future__ import annotations

import json
from typing import Any, Callable

import httpx

from research_mcp.llm import OllamaLLM

Responder = Callable[[dict[str, Any]], dict[str, Any]]


class FakeOllama:
    def __init__(self, responder: Responder, models: tuple[str, ...] = ("llama3.1:8b", "nomic-embed-text:latest")):
        self.responder = responder
        self.models = models
        self.requests: list[dict[str, Any]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": m} for m in self.models]})
        if request.url.path == "/api/chat":
            body = json.loads(request.content)
            self.requests.append(body)
            if body["model"] not in self.models:
                return httpx.Response(404, json={"error": f"model '{body['model']}' not found"})
            return httpx.Response(200, json={"model": body["model"], "message": self.responder(body), "done": True})
        return httpx.Response(404)

    def llm(self, model: str = "llama3.1:8b") -> OllamaLLM:
        client = httpx.Client(transport=httpx.MockTransport(self.handler))
        return OllamaLLM("http://192.168.1.50:11434", model, client=client)


def echo_citations(body: dict[str, Any]) -> dict[str, Any]:
    """Answer by quoting the first citation tag found in the prompt."""
    prompt = body["messages"][-1]["content"]
    start = prompt.find("[")
    tag = prompt[start : prompt.find("]", start) + 1] if start >= 0 else ""
    return {"role": "assistant", "content": f"## Answer\n\nThe sources say this {tag}."}
