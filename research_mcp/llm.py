"""Ollama LLM client (local machine, a LAN host such as http://192.168.1.50:11434, or Ollama Cloud)."""

from __future__ import annotations

import ipaddress
import json
import logging
from dataclasses import dataclass, field
from typing import Any

import httpx

from .config import Settings

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


def is_local_url(url: str) -> bool:
    """True for localhost, private-range IPs (LAN) and mDNS/LAN hostnames like ``gpu-box.local``."""
    host = httpx.URL(url).host
    if host in ("localhost", "") or host.endswith((".local", ".lan", ".home", ".internal")) or "." not in host:
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return ip.is_private or ip.is_loopback or ip.is_link_local


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass
class ChatReply:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


class OllamaLLM:
    def __init__(
        self,
        url: str,
        model: str,
        *,
        timeout: float = 300.0,
        temperature: float = 0.2,
        num_ctx: int = 8192,
        api_key: str = "",
        client: httpx.Client | None = None,
    ):
        self.url = url.rstrip("/")
        self.model = model
        self.options = {"temperature": temperature, "num_ctx": num_ctx}
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        # A LAN/localhost Ollama must not be routed through an HTTP(S)_PROXY from the environment.
        self._client = client or httpx.Client(timeout=timeout, headers=headers, trust_env=not is_local_url(self.url))

    @classmethod
    def from_settings(cls, settings: Settings, client: httpx.Client | None = None) -> "OllamaLLM":
        return cls(
            settings.llm_url or settings.ollama_url,
            settings.llm_model,
            timeout=settings.llm_timeout,
            temperature=settings.llm_temperature,
            num_ctx=settings.llm_num_ctx,
            api_key=settings.ollama_api_key,
            client=client,
        )

    @property
    def name(self) -> str:
        return f"ollama:{self.model}@{self.url}"

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self._client.post(f"{self.url}{path}", json=payload)
        except httpx.ConnectError as exc:
            raise LLMError(
                f"Cannot reach Ollama at {self.url}. Is it running, and started with OLLAMA_HOST=0.0.0.0 "
                f"so other machines on the LAN can connect? ({exc})"
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"Ollama request failed: {exc}") from exc
        if response.status_code == 404 and "not found" in response.text.lower():
            raise LLMError(f"Model {self.model!r} is not installed on {self.url}: run `ollama pull {self.model}` there")
        if response.status_code >= 400:
            raise LLMError(f"Ollama returned HTTP {response.status_code}: {response.text[:300]}")
        return response.json()

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None, json_mode: bool = False) -> ChatReply:
        payload: dict[str, Any] = {"model": self.model, "messages": messages, "stream": False, "options": self.options}
        if tools:
            payload["tools"] = tools
        if json_mode:
            payload["format"] = "json"
        data = self._post("/api/chat", payload)
        message = data.get("message") or {}
        calls = []
        for call in message.get("tool_calls") or []:
            fn = call.get("function") or {}
            args = fn.get("arguments") or {}
            if isinstance(args, str):  # some models return a JSON string
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            calls.append(ToolCall(fn.get("name", ""), args))
        return ChatReply(message.get("content") or "", calls, data)

    def complete(self, prompt: str, system: str = "") -> str:
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        return self.chat(messages).content.strip()

    def models(self) -> list[str]:
        try:
            response = self._client.get(f"{self.url}/api/tags")
            response.raise_for_status()
        except httpx.ConnectError as exc:
            raise LLMError(
                f"Cannot reach Ollama at {self.url}. Is it running, and started with OLLAMA_HOST=0.0.0.0 "
                f"so other machines on the LAN can connect? ({exc})"
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"Ollama request to {self.url} failed: {exc}") from exc
        return [m.get("name", "") for m in response.json().get("models") or []]

    def health(self) -> dict[str, Any]:
        """Connectivity check used by `llm_status`: reachable? model installed?"""
        try:
            installed = self.models()
        except LLMError as exc:
            return {"reachable": False, "url": self.url, "model": self.model, "error": str(exc)}
        base = self.model.split(":")[0]
        present = any(m == self.model or m.split(":")[0] == base and ":" not in self.model for m in installed)
        status = {"reachable": True, "url": self.url, "model": self.model, "model_installed": present, "installed_models": installed}
        if not present:
            status["hint"] = f"run `ollama pull {self.model}` on the Ollama host"
        return status
