"""
Async MCP stdio client that spawns rag_mcp_server.py as a subprocess
and exposes its tools synchronously to the agentic loop.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

_SERVER_SCRIPT = str(Path(__file__).parent / "rag_mcp_server.py")

_SERVER_PARAMS = StdioServerParameters(
    command=sys.executable,
    args=[_SERVER_SCRIPT],
)


async def _list_tools_async() -> list[dict]:
    async with stdio_client(_SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools_result = await session.list_tools()
            return [
                {
                    "name": t.name,
                    "description": t.description or "",
                    "input_schema": t.inputSchema,
                }
                for t in tools_result.tools
            ]


async def _call_tool_async(name: str, arguments: dict) -> str:
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
