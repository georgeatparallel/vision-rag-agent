"""
MCP client exposing web tools synchronously to the agentic loop.

The bundled stdio server is the default. WEB_TOOLS_PROVIDER=parallel selects
Parallel's HTTP search/fetch tools, with Wikipedia still served locally.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client

import config

_PARALLEL_URL = "https://search.parallel.ai/mcp"
_USER_AGENT = "vision-rag-agent/1.0 (+https://github.com/JayKimProject/vision-rag-agent)"
_PARALLEL_TIMEOUT = 60
_PARALLEL_TOOLS = {"web_search": "web_search", "fetch_url": "web_fetch"}

_SERVER_SCRIPT = str(Path(__file__).parent / "rag_mcp_server.py")

_SERVER_PARAMS = StdioServerParameters(
    command=sys.executable,
    args=[_SERVER_SCRIPT],
)


def _provider() -> str:
    provider = config.WEB_TOOLS_PROVIDER
    if provider not in ("local", "parallel"):
        raise ValueError("WEB_TOOLS_PROVIDER must be 'local' or 'parallel'")
    return provider


async def _parallel_request(name: str | None = None, arguments: dict | None = None):
    async def request():
        async with httpx.AsyncClient(
            headers={"User-Agent": _USER_AGENT},
            timeout=httpx.Timeout(30, read=60),
        ) as client:
            async with streamable_http_client(_PARALLEL_URL, http_client=client) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    if name is None:
                        return await session.list_tools()
                    return await session.call_tool(name, arguments or {})

    return await asyncio.wait_for(request(), timeout=_PARALLEL_TIMEOUT)


async def _list_tools_async() -> list[dict]:
    provider = _provider()
    async with stdio_client(_SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools_result = await session.list_tools()
    tools = tools_result.tools
    if provider == "parallel":
        remote = await _parallel_request()
        by_name = {t.name: t for t in remote.tools}
        if not set(_PARALLEL_TOOLS.values()).issubset(by_name):
            raise RuntimeError("Parallel MCP did not advertise web_search and web_fetch")
        # Keep public tool names aligned with the agent's existing system prompt,
        # but use the provider's live schemas rather than guessing its arguments.
        tools = [t for t in tools if t.name not in _PARALLEL_TOOLS]
        tools += [by_name[remote_name].model_copy(update={"name": local_name})
                  for local_name, remote_name in _PARALLEL_TOOLS.items()]
    return [
        {"name": t.name, "description": t.description or "", "input_schema": t.inputSchema}
        for t in tools
    ]


async def _call_tool_async(name: str, arguments: dict) -> str:
    if _provider() == "parallel" and name in _PARALLEL_TOOLS:
        try:
            result = await _parallel_request(_PARALLEL_TOOLS[name], arguments)
            return "\n".join(block.text for block in result.content if hasattr(block, "text"))
        except asyncio.TimeoutError:
            return json.dumps({"error": "Parallel MCP request timed out"})
        except Exception as exc:
            return json.dumps({"error": f"Parallel MCP request failed: {exc}"})
    async with stdio_client(_SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(name, arguments)
            return "\n".join(
                block.text for block in result.content if hasattr(block, "text")
            )


def list_mcp_tools() -> list[dict]:
    """Return tool schemas from the MCP server (for passing to Claude)."""
    return asyncio.run(_list_tools_async())


def call_mcp_tool(name: str, arguments: dict) -> str:
    """Call a tool on the MCP server and return the text result."""
    return asyncio.run(_call_tool_async(name, arguments))
