import json

import httpx
import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from research_mcp import generation, server
from research_mcp.agent import ResearchAgent, mcp_tools_to_ollama
from research_mcp.llm import LLMError, OllamaLLM, is_local_url

from fake_ollama import FakeOllama, echo_citations


@pytest.fixture()
def anyio_backend():
    return "asyncio"


@pytest.mark.parametrize(
    "url,local",
    [
        ("http://192.168.1.50:11434", True),
        ("http://10.0.0.7:11434", True),
        ("http://localhost:11434", True),
        ("http://gpu-box.local:11434", True),
        ("http://gpubox:11434", True),
        ("https://ollama.com", False),
    ],
)
def test_is_local_url(url, local):
    assert is_local_url(url) is local


def test_health_reports_missing_model():
    fake = FakeOllama(echo_citations, models=("mistral:latest",))
    status = fake.llm("llama3.1:8b").health()
    assert status["reachable"] and not status["model_installed"] and "ollama pull" in status["hint"]
    assert FakeOllama(echo_citations, models=("llama3.1:latest",)).llm("llama3.1").health()["model_installed"]


def test_unreachable_server_gives_lan_hint():
    def refuse(request):
        raise httpx.ConnectError("connection refused")

    llm = OllamaLLM("http://192.168.1.50:11434", "llama3.1:8b", client=httpx.Client(transport=httpx.MockTransport(refuse)))
    with pytest.raises(LLMError, match="OLLAMA_HOST=0.0.0.0"):
        llm.complete("hi")
    assert llm.health()["reachable"] is False


def test_ask_is_grounded_and_flags_unknown_citations(loaded):
    fake = FakeOllama(echo_citations)
    result = generation.ask(loaded, fake.llm(), "What dataset is used for molecules?")
    assert result["sources"] and "warning" not in result
    assert result["sources"][0]["citation"] in result["answer"]
    prompt = fake.requests[0]["messages"][-1]["content"]
    assert "QM9" in prompt and fake.requests[0]["messages"][0]["role"] == "system"

    liar = FakeOllama(lambda body: {"role": "assistant", "content": "As shown by [Madeup2020, p. 1]."})
    assert "Madeup2020" in generation.ask(loaded, liar.llm(), "molecules")["warning"]


def test_write_literature_review(loaded):
    fake = FakeOllama(echo_citations)
    result = generation.write_literature_review(loaded, fake.llm(), "efficient fine-tuning of language models", n_themes=2)
    md = result["review_markdown"]
    assert md.startswith("# Literature Review") and "```bibtex" in md
    assert len(fake.requests) == len(result["themes"]) + 2  # one per theme + intro + closing


def test_synthesize_comparison(loaded):
    fake = FakeOllama(echo_citations)
    ids = [p.id for p in loaded.store.list_papers()][:2]
    result = generation.synthesize_comparison(loaded, fake.llm(), ids, ["method"])
    assert result["comparison"] and "Evidence by aspect" in fake.requests[0]["messages"][-1]["content"]


def test_tool_schema_conversion():
    class T:
        name = "semantic_search"
        description = "search"
        inputSchema = {"type": "object", "title": "x", "properties": {
            "paper_ids": {"anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "null"}], "default": None, "title": "Paper Ids"}}}

    class Excluded(T):
        name = "ask_papers"

    tools = mcp_tools_to_ollama([T(), Excluded()])
    assert len(tools) == 1
    assert tools[0]["function"]["parameters"]["properties"]["paper_ids"] == {"type": "array", "items": {"type": "string"}, "default": None}


@pytest.mark.anyio
async def test_agent_calls_mcp_tools_then_answers(loaded):
    def responder(body):
        if body["messages"][-1]["role"] == "user":
            return {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "semantic_search", "arguments": {"query": "QM9 molecules", "top_k": 2}}}]}
        tool_output = json.loads(body["messages"][-1]["content"])
        return {"role": "assistant", "content": f"They use QM9 {tool_output[0]['citation']}."}

    fake = FakeOllama(responder)
    server.set_library(loaded)
    events = []
    try:
        async with create_connected_server_and_client_session(server.mcp._mcp_server) as session:
            agent = ResearchAgent(session, fake.llm(), on_event=lambda k, d: events.append((k, d)))
            await agent.start()
            assert "semantic_search" in {t["function"]["name"] for t in agent.tools}
            assert "ask_papers" not in {t["function"]["name"] for t in agent.tools}
            answer = await agent.ask("Which dataset do the molecule papers use?")
    finally:
        server.set_library(None)
    assert answer.startswith("They use QM9 [White2021")
    assert events[0] == ("tool_call", {"name": "semantic_search", "arguments": {"query": "QM9 molecules", "top_k": 2}})
    roles = [m["role"] for m in agent.messages]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    assert fake.requests[0]["tools"]


@pytest.mark.anyio
async def test_server_llm_tools(loaded):
    fake = FakeOllama(echo_citations)
    server.set_library(loaded)
    server.set_llm(fake.llm())
    try:
        async with create_connected_server_and_client_session(server.mcp._mcp_server) as client:
            status = await client.call_tool("llm_status", {})
            assert '"reachable": true' in status.content[0].text
            res = await client.call_tool("ask_papers", {"question": "What is low-rank adaptation?"})
            assert not res.isError and "The sources say" in res.content[0].text
    finally:
        server.set_library(None)
        server.set_llm(None)


def test_dotenv_configures_lan_ollama(tmp_path, monkeypatch):
    from research_mcp.config import Settings

    env = tmp_path / ".env"
    env.write_text("# LAN box\nRESEARCH_OLLAMA_URL=http://192.168.1.50:11434\nexport RESEARCH_LLM_MODEL='qwen2.5:14b'\nRESEARCH_DATA_DIR=" + str(tmp_path) + "\n")
    for key in ("RESEARCH_OLLAMA_URL", "RESEARCH_LLM_MODEL", "RESEARCH_LLM_URL", "RESEARCH_DATA_DIR"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("RESEARCH_ENV_FILE", str(env))
    settings = Settings.from_env()
    llm = OllamaLLM.from_settings(settings)
    assert llm.url == "http://192.168.1.50:11434" and llm.model == "qwen2.5:14b"
    for key in ("RESEARCH_OLLAMA_URL", "RESEARCH_LLM_MODEL", "RESEARCH_DATA_DIR"):
        monkeypatch.delenv(key, raising=False)
