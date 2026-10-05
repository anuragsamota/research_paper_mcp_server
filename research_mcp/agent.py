"""research-agent: a chat agent whose LLM is Ollama (e.g. on your LAN) and whose tools come from this MCP server.

    research-agent                                  # spawns the MCP server over stdio
    research-agent --server-url http://host:8000/mcp --token ...   # or connects to a running one
    research-agent "Summarize what my library says about LoRA"     # one-shot
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import AsyncExitStack
from typing import Any

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .config import Settings
from .llm import LLMError, OllamaLLM

SYSTEM_PROMPT = """\
You are a research assistant with tools for searching, reading and analyzing research papers.
- Discover papers with search_papers; add them with add_paper (use the `add_with` value).
- Before answering questions about papers, call semantic_search and base your answer on the passages.
- Cite claims with the citation tags from tool results, e.g. [Hu2021, p. 4]. Never invent citations.
- Use summarize_paper, compare_papers, find_citations, find_related_papers and literature_review for analysis.
- Be concise. When a tool returns an error, fix the arguments or explain the problem to the user."""

# Generation tools call the server's own LLM; the agent *is* the LLM, so it doesn't need them.
EXCLUDED_TOOLS = {"ask_papers", "synthesize_comparison", "write_literature_review", "llm_status"}


def _simplify_schema(schema: Any) -> Any:
    """Make MCP JSON Schemas friendlier to local models: drop titles, collapse Optional[...] anyOf."""
    if isinstance(schema, dict):
        if "anyOf" in schema:
            options = [o for o in schema["anyOf"] if o.get("type") != "null"]
            merged = {k: v for k, v in schema.items() if k != "anyOf"}
            if len(options) == 1:
                merged.update(options[0])
                return _simplify_schema(merged)
        return {k: _simplify_schema(v) for k, v in schema.items() if k != "title"}
    if isinstance(schema, list):
        return [_simplify_schema(s) for s in schema]
    return schema


def mcp_tools_to_ollama(tools: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": (t.description or "").strip(),
                "parameters": _simplify_schema(t.inputSchema or {"type": "object", "properties": {}}),
            },
        }
        for t in tools
        if t.name not in EXCLUDED_TOOLS
    ]


class ResearchAgent:
    def __init__(self, session: ClientSession, llm: OllamaLLM, *, max_steps: int = 10, max_tool_chars: int = 6000,
                 on_event=None):
        self.session = session
        self.llm = llm
        self.max_steps = max_steps
        self.max_tool_chars = max_tool_chars
        self.on_event = on_event or (lambda kind, data: None)
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.tools: list[dict[str, Any]] = []

    async def start(self) -> None:
        listing = await self.session.list_tools()
        self.tools = mcp_tools_to_ollama(listing.tools)

    def reset(self) -> None:
        self.messages = self.messages[:1]

    async def _call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        if name not in {t["function"]["name"] for t in self.tools}:
            return f"Error: unknown tool {name!r}."
        try:
            result = await self.session.call_tool(name, arguments)
        except Exception as exc:  # transport errors etc.
            return f"Error calling {name}: {exc}"
        structured = result.structuredContent
        if structured and not result.isError:
            # List results arrive as one text block per item; the structured form is a single valid JSON document.
            payload = structured.get("result", structured) if set(structured) == {"result"} else structured
            text = json.dumps(payload, ensure_ascii=False)
        else:
            text = "\n".join(getattr(c, "text", "") for c in result.content)
        if result.isError:
            text = f"Error: {text}"
        if len(text) > self.max_tool_chars:
            text = text[: self.max_tool_chars] + f"\n…[truncated {len(text) - self.max_tool_chars} chars]"
        return text

    async def ask(self, user_message: str) -> str:
        self.messages.append({"role": "user", "content": user_message})
        for _ in range(self.max_steps):
            reply = await anyio.to_thread.run_sync(lambda: self.llm.chat(self.messages, self.tools))
            if not reply.tool_calls:
                self.messages.append({"role": "assistant", "content": reply.content})
                return reply.content
            self.messages.append({
                "role": "assistant",
                "content": reply.content,
                "tool_calls": [{"function": {"name": c.name, "arguments": c.arguments}} for c in reply.tool_calls],
            })
            for call in reply.tool_calls:
                self.on_event("tool_call", {"name": call.name, "arguments": call.arguments})
                output = await self._call_tool(call.name, call.arguments)
                self.on_event("tool_result", {"name": call.name, "chars": len(output), "error": output.startswith("Error")})
                self.messages.append({"role": "tool", "content": output, "tool_name": call.name})
        # Step budget exhausted: ask for a final answer without tools.
        self.messages.append({"role": "user", "content": "Stop using tools now and give your best final answer."})
        reply = await anyio.to_thread.run_sync(lambda: self.llm.chat(self.messages))
        self.messages.append({"role": "assistant", "content": reply.content})
        return reply.content


async def _connect(stack: AsyncExitStack, args: argparse.Namespace) -> ClientSession:
    if args.server_url:
        from mcp.client.streamable_http import streamablehttp_client

        headers = {"Authorization": f"Bearer {args.token}"} if args.token else None
        read, write, _ = await stack.enter_async_context(streamablehttp_client(args.server_url, headers=headers))
    else:
        params = StdioServerParameters(
            command=sys.executable, args=["-m", "research_mcp", "--transport", "stdio"],
            env={**os.environ, "RESEARCH_LOG_LEVEL": os.environ.get("RESEARCH_LOG_LEVEL", "WARNING")},
        )
        read, write = await stack.enter_async_context(stdio_client(params))
    session = await stack.enter_async_context(ClientSession(read, write))
    await session.initialize()
    return session


def _printer(verbose: bool):
    def on_event(kind: str, data: dict[str, Any]) -> None:
        if kind == "tool_call":
            args = json.dumps(data["arguments"], ensure_ascii=False)
            print(f"  → {data['name']}({args if verbose else args[:120]})", file=sys.stderr)
        elif kind == "tool_result" and data["error"]:
            print(f"  ✗ {data['name']} returned an error", file=sys.stderr)
    return on_event


async def _main(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    if args.ollama_url:
        settings.llm_url = args.ollama_url.rstrip("/")
    if args.model:
        settings.llm_model = args.model
    llm = OllamaLLM.from_settings(settings)
    status = llm.health()
    if not status["reachable"]:
        print(status["error"], file=sys.stderr)
        return 2
    if not status["model_installed"]:
        print(f"Model {llm.model!r} not found on {llm.url}. Installed: {', '.join(status['installed_models']) or 'none'}.\n"
              f"Run `ollama pull {llm.model}` on that machine or pass --model.", file=sys.stderr)
        return 2

    async with AsyncExitStack() as stack:
        session = await _connect(stack, args)
        agent = ResearchAgent(session, llm, max_steps=args.max_steps, on_event=_printer(args.verbose))
        await agent.start()
        if args.question:
            print(await agent.ask(" ".join(args.question)))
            return 0
        print(f"Research agent · LLM {llm.name} · {len(agent.tools)} tools. Commands: /reset /tools /quit", file=sys.stderr)
        while True:
            try:
                line = (await anyio.to_thread.run_sync(lambda: input("\nyou> "))).strip()
            except (EOFError, KeyboardInterrupt):
                return 0
            if not line:
                continue
            if line in ("/quit", "/exit"):
                return 0
            if line == "/reset":
                agent.reset()
                print("(conversation cleared)")
                continue
            if line == "/tools":
                print(", ".join(t["function"]["name"] for t in agent.tools))
                continue
            try:
                print("\nassistant> " + await agent.ask(line))
            except LLMError as exc:
                print(f"LLM error: {exc}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Research assistant agent powered by Ollama + the research MCP server")
    parser.add_argument("question", nargs="*", help="Ask one question and exit (omit for interactive chat)")
    parser.add_argument("--ollama-url", default=None, help="Ollama URL, e.g. http://192.168.1.50:11434 (env RESEARCH_LLM_URL / RESEARCH_OLLAMA_URL)")
    parser.add_argument("--model", default=None, help="Ollama model with tool calling, e.g. llama3.1:8b, qwen2.5:14b (env RESEARCH_LLM_MODEL)")
    parser.add_argument("--server-url", default=os.environ.get("RESEARCH_SERVER_URL"), help="Use a running MCP server over HTTP instead of spawning one")
    parser.add_argument("--token", default=os.environ.get("RESEARCH_API_TOKEN"), help="Bearer token for --server-url")
    parser.add_argument("--max-steps", type=int, default=10, help="Max tool-calling rounds per question")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show full tool arguments")
    sys.exit(anyio.run(_main, parser.parse_args()))


if __name__ == "__main__":
    main()
