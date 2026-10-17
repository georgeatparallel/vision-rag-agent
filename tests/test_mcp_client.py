"""Exercise the real MCP HTTP transport and client dispatch without network calls."""
import asyncio
from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import httpx

import config
import mcp_client


SEARCH_SCHEMA = {
    "type": "object",
    "properties": {"objective": {"type": "string"},
                   "search_queries": {"type": "array", "items": {"type": "string"}}},
    "required": ["objective", "search_queries"],
}
FETCH_SCHEMA = {
    "type": "object", "properties": {"urls": {"type": "array", "items": {"type": "string"}}},
    "required": ["urls"],
}
SEARCH_ARGS = {"objective": "Find Python venv documentation",
               "search_queries": ["Python venv documentation"]}
FETCH_ARGS = {"urls": ["https://docs.python.org/3/library/venv.html"]}
PAGE_TEXT = "venv creates virtual environments. Source: https://docs.python.org/3/library/venv.html"


class MCPClientTests(unittest.TestCase):
    def setUp(self):
        self.requests = []
        self.error = None
        self.block = False
        self.entered = None
        self.tools = [
            {"name": "web_search", "description": "Search current sources", "inputSchema": SEARCH_SCHEMA},
            {"name": "web_fetch", "description": "Fetch page content", "inputSchema": FETCH_SCHEMA},
        ]
        # Use the real SDK, HTTP client and JSON-RPC protocol. Replace only the
        # network boundary, retaining and asserting every destination/header/body.
        real_client = httpx.AsyncClient

        def client(**kwargs):
            return real_client(transport=httpx.MockTransport(self.handle), **kwargs)

        self.provider = patch.object(config, "WEB_TOOLS_PROVIDER", "parallel")
        self.http = patch.object(mcp_client.httpx, "AsyncClient", side_effect=client)
        self.provider.start()
        self.http.start()
        self.addCleanup(self.provider.stop)
        self.addCleanup(self.http.stop)

    async def handle(self, request):
        self.assertEqual(str(request.url), "https://search.parallel.ai/mcp")
        self.assertEqual(request.headers["user-agent"],
                         "vision-rag-agent/1.0 (+https://github.com/JayKimProject/vision-rag-agent)")
        self.assertNotIn("authorization", request.headers)
        payload = json.loads(request.content)
        self.requests.append(payload)
        method = payload["method"]
        if method == "notifications/initialized":
            return httpx.Response(202)
        if method == "initialize":
            result = {"protocolVersion": payload["params"]["protocolVersion"],
                      "capabilities": {"tools": {}}, "serverInfo": {"name": "fixture", "version": "1"}}
        elif method == "tools/list":
            result = {"tools": self.tools}
        elif method == "tools/call":
            if self.block:
                if self.entered:
                    self.entered.set()
                await asyncio.Event().wait()
            if self.error == "http":
                return httpx.Response(503, text="Service unavailable")
            result = {"content": [{"type": "text", "text": self.error or PAGE_TEXT}],
                      "isError": bool(self.error)}
        else:
            self.fail(f"Unexpected MCP method: {method}")
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": payload["id"], "result": result})

    def test_live_schemas_and_routing_keep_wikipedia_local(self):
        schemas = mcp_client.list_mcp_tools()  # Includes actual stdio subprocess discovery.
        by_name = {t["name"]: t for t in schemas}
        self.assertEqual(set(by_name), {"web_search", "fetch_url", "wikipedia"})
        self.assertEqual(by_name["web_search"]["input_schema"], SEARCH_SCHEMA)
        self.assertEqual(by_name["fetch_url"]["input_schema"], FETCH_SCHEMA)
        for name, arguments, remote in [("web_search", SEARCH_ARGS, "web_search"),
                                         ("fetch_url", FETCH_ARGS, "web_fetch")]:
            self.assertEqual(mcp_client.call_mcp_tool(name, arguments), PAGE_TEXT)
            call = [r for r in self.requests if r["method"] == "tools/call"][-1]
            self.assertEqual(call["params"]["name"], remote)
            self.assertEqual(call["params"]["arguments"], arguments)
        with patch.object(mcp_client, "stdio_client", side_effect=RuntimeError("local dispatch")):
            with self.assertRaisesRegex(RuntimeError, "local dispatch"):
                mcp_client.call_mcp_tool("wikipedia", {"topic": "Python"})

    def test_local_default_never_contacts_parallel(self):
        with patch.object(config, "WEB_TOOLS_PROVIDER", "local"):
            tools = {t["name"]: t for t in mcp_client.list_mcp_tools()}
            self.assertIn("query", tools["web_search"]["input_schema"]["properties"])
            self.assertIn("url", tools["fetch_url"]["input_schema"]["properties"])
            with patch.object(mcp_client, "stdio_client", side_effect=RuntimeError("local dispatch")):
                with self.assertRaisesRegex(RuntimeError, "local dispatch"):
                    mcp_client.call_mcp_tool("web_search", {"query": "Python"})
        self.assertEqual(self.requests, [])

    def test_invalid_provider_and_missing_tools_fail_discovery(self):
        with patch.object(config, "WEB_TOOLS_PROVIDER", "typo"):
            with self.assertRaisesRegex(ValueError, "WEB_TOOLS_PROVIDER"):
                mcp_client.list_mcp_tools()
        self.tools = self.tools[:1]
        with self.assertRaisesRegex(RuntimeError, "did not advertise"):
            mcp_client.list_mcp_tools()

    def test_tool_and_transport_errors_reach_the_caller_without_fallback(self):
        self.error = "Free usage quota exceeded"
        self.assertEqual(mcp_client.call_mcp_tool("web_search", SEARCH_ARGS), self.error)
        self.error = "http"
        self.assertIn("Parallel MCP request failed", mcp_client.call_mcp_tool("fetch_url", FETCH_ARGS))
        calls = [r for r in self.requests if r["method"] == "tools/call"]
        self.assertEqual([c["params"]["name"] for c in calls], ["web_search", "web_fetch"])

    def test_timeout_is_bounded(self):
        self.block = True
        with patch.object(mcp_client, "_PARALLEL_TIMEOUT", 0.1):
            result = mcp_client.call_mcp_tool("web_search", SEARCH_ARGS)
        self.assertEqual(json.loads(result), {"error": "Parallel MCP request timed out"})
        self.assertTrue(any(r["method"] == "tools/call" for r in self.requests))

    def test_cancellation_propagates(self):
        async def run():
            self.block = True
            self.entered = asyncio.Event()
            task = asyncio.create_task(mcp_client._call_tool_async("web_search", SEARCH_ARGS))
            await asyncio.wait_for(self.entered.wait(), 2)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        asyncio.run(run())

    def test_agent_dispatch_consumes_remote_result(self):
        # The agent's MCP dispatch does not require loading embedding models or
        # making a paid model request. Stub those unrelated imports only.
        tools = ModuleType("tools")
        tools.RAG_TOOL_SCHEMAS = []
        tools.execute_rag_tool = lambda *a: self.fail("unexpected RAG dispatch")
        anthropic = ModuleType("anthropic")
        anthropic.Anthropic = lambda **kw: SimpleNamespace()
        spec = importlib.util.spec_from_file_location("agent_under_test", Path(__file__).parents[1] / "agent.py")
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"tools": tools, "anthropic": anthropic}):
            spec.loader.exec_module(module)
            agent = module.MultimodalRAGAgent()
            self.assertEqual(agent._execute("web_search", SEARCH_ARGS), [{"type": "text", "text": PAGE_TEXT}])
            self.assertEqual(agent._execute("fetch_url", FETCH_ARGS), [{"type": "text", "text": PAGE_TEXT}])
            self.error = "No sources available"
            self.assertEqual(agent._execute("web_search", SEARCH_ARGS), [{"type": "text", "text": self.error}])
            self.error = None
            turns = []

            def stream(**kwargs):
                turns.append(kwargs)
                if len(turns) == 1:
                    blocks = [SimpleNamespace(type="tool_use", id="search", name="web_search", input=SEARCH_ARGS),
                              SimpleNamespace(type="tool_use", id="fetch", name="fetch_url", input=FETCH_ARGS)]
                    final = SimpleNamespace(stop_reason="tool_use", content=blocks)
                    tokens = []
                else:
                    results = kwargs["messages"][-1]["content"]
                    self.assertEqual([r["tool_use_id"] for r in results], ["search", "fetch"])
                    self.assertTrue(all(r["content"] == [{"type": "text", "text": PAGE_TEXT}] for r in results))
                    final = SimpleNamespace(stop_reason="end_turn", content=[])
                    tokens = ["A virtual environment isolates Python packages."]
                return nullcontext(SimpleNamespace(text_stream=iter(tokens), get_final_message=lambda: final))

            agent._client.messages = SimpleNamespace(stream=stream)
            self.assertEqual(agent.query("How do I create a Python environment?"),
                             "A virtual environment isolates Python packages.")
            self.assertEqual(len(turns), 2)


if __name__ == "__main__":
    unittest.main()
