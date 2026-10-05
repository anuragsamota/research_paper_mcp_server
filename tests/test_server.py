import json

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from research_mcp import server


@pytest.fixture()
def anyio_backend():
    return "asyncio"


@pytest.fixture()
def mcp_library(loaded):
    server.set_library(loaded)
    yield loaded
    server.set_library(None)


@pytest.mark.anyio
async def test_tools_resources_prompts(mcp_library):
    async with create_connected_server_and_client_session(server.mcp._mcp_server) as client:
        tools = {t.name for t in (await client.list_tools()).tools}
        assert {"search_papers", "add_paper", "semantic_search", "compare_papers", "literature_review",
                "find_citations", "summarize_paper"} <= tools

        result = await client.call_tool("semantic_search", {"query": "QM9 molecules", "top_k": 2})
        assert not result.isError
        hits = result.structuredContent["result"]
        assert "Message Passing" in hits[0]["title"]

        pid = hits[0]["paper_id"]
        resource = await client.read_resource(f"papers://{pid}/fulltext")
        assert "## Method" in resource.contents[0].text
        listing = json.loads((await client.read_resource("papers://library")).contents[0].text)
        assert len(listing) == 3

        bad = await client.call_tool("summarize_paper", {"paper_id": "nope"})
        assert bad.isError and "No paper" in bad.content[0].text

        prompts = {p.name for p in (await client.list_prompts()).prompts}
        assert {"review_literature", "critique_paper", "compare_approaches", "answer_question"} <= prompts
        prompt = await client.get_prompt("review_literature", {"topic": "efficient fine-tuning"})
        assert "literature_review" in prompt.messages[0].content.text


def test_http_token_auth(mcp_library, monkeypatch):
    from starlette.testclient import TestClient

    from research_mcp.http_app import create_app

    monkeypatch.setenv("RESEARCH_API_TOKEN", "s3cret")
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
    headers = {"accept": "application/json, text/event-stream", "content-type": "application/json"}
    app = create_app()
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.post("/mcp", json=body, headers=headers).status_code == 401
        ok = client.post("/mcp", json=body, headers={**headers, "authorization": "Bearer s3cret"})
        assert ok.status_code == 200
        assert any(t["name"] == "semantic_search" for t in ok.json()["result"]["tools"])
