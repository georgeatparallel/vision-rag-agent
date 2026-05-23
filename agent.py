"""
Agentic multimodal RAG using Claude tool_use.

Claude drives the loop:
  1. Calls search_knowledge_base (ChromaDB text + image)
  2. Calls web_search / fetch_url / wikipedia (MCP tools) when KB is insufficient
  3. May call ingest_file to persist fetched content locally
  4. Repeats until confident, then streams the final answer token by token

MCP tools (fetch_url, web_search, wikipedia) run in rag_mcp_server.py via
stdio transport and are loaded dynamically at startup.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Iterator

import anthropic

import config
from tools import RAG_TOOL_SCHEMAS, execute_rag_tool
from mcp_client import list_mcp_tools, call_mcp_tool


# ── Media helpers ──────────────────────────────────────────────────────────

_MEDIA_TYPES = {
    "jpg": "image/jpeg", "jpeg": "image/jpeg",
    "png": "image/png", "gif": "image/gif",
    "webp": "image/webp", "bmp": "image/png",
}


def _image_block(path: str) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    media_type = _MEDIA_TYPES.get(p.suffix.lower().lstrip("."), "image/jpeg")
    data = base64.standard_b64encode(p.read_bytes()).decode()
    return {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}}


def _build_tool_result_content(tool_name: str, result: str) -> list[dict]:
    """For KB search results, attach the actual image bytes alongside the JSON."""
    content: list[dict] = [{"type": "text", "text": result}]
    if tool_name != "search_knowledge_base":
        return content
    try:
        hits = json.loads(result)
        seen: set[str] = set()
        for hit in hits:
            img_path = hit.get("image_path")
            if img_path and img_path not in seen:
                seen.add(img_path)
                block = _image_block(img_path)
                if block:
                    content.append({"type": "text", "text": f"[Image: {Path(img_path).name}]"})
                    content.append(block)
    except (json.JSONDecodeError, TypeError):
        pass
    return content


# ── System prompt ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a multimodal RAG assistant with a local knowledge base (text + images) \
and live web tools.

On every question:
1. Call `search_knowledge_base` first (search_type="both") to check the local KB.
2. If KB results are weak or missing, call `web_search` to find leads, then \
`fetch_url` to read pages or `wikipedia` for encyclopedic topics.
3. You may call tools multiple times and refine your queries.
4. Optionally call `ingest_file` to persist useful fetched content for future queries.
5. Once you have sufficient context, answer thoroughly and cite sources by file name or URL.

When search results include image captions or raw images, describe what you see \
and explain how it relates to the question.
If no reliable answer is found after reasonable effort, say so honestly."""


# ── Agent ──────────────────────────────────────────────────────────────────

class MultimodalRAGAgent:
    def __init__(self, max_iterations: int = 12):
        self.max_iterations = max_iterations
        self._client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

        print("[agent] loading MCP tools from rag_mcp_server...")
        mcp_schemas = list_mcp_tools()
        self._mcp_tool_names: set[str] = {t["name"] for t in mcp_schemas}
        self._all_tools: list[dict] = RAG_TOOL_SCHEMAS + mcp_schemas
        names = [t["name"] for t in self._all_tools]
        print(f"[agent] tools ready: {names}")

    def _execute(self, name: str, inputs: dict) -> list[dict]:
        """Run a tool and return Claude-compatible content blocks."""
        if name in self._mcp_tool_names:
            result = call_mcp_tool(name, inputs)
            return [{"type": "text", "text": result}]
        result = execute_rag_tool(name, inputs)
        return _build_tool_result_content(name, str(result))

    def query_stream(self, question: str, verbose: bool = False) -> Iterator[str]:
        """
        Agentic loop with true token streaming for the final answer.

        Tool-use turns run non-streaming (collect results, feed back).
        The final answer turn uses the streaming API — yields text tokens
        as they arrive from Claude.
        """
        messages: list[dict] = [{"role": "user", "content": question}]

        for iteration in range(self.max_iterations):
            with self._client.messages.stream(
                model=config.CLAUDE_MODEL,
                max_tokens=2048,
                system=SYSTEM_PROMPT,
                tools=self._all_tools,
                messages=messages,
            ) as stream:
                # stream.text_stream yields text tokens in real time.
                # For tool-use turns it yields nothing (or a text preamble).
                for token in stream.text_stream:
                    yield token

                final_msg = stream.get_final_message()

            if verbose:
                print(f"\n[agent] step {iteration + 1} | stop={final_msg.stop_reason}")

            # Final answer — streaming already delivered it above
            if final_msg.stop_reason == "end_turn":
                return

            if final_msg.stop_reason != "tool_use":
                break

            # Execute all tool calls
            tool_results: list[dict] = []
            for block in final_msg.content:
                if block.type != "tool_use":
                    continue

                if verbose:
                    args_preview = json.dumps(block.input)[:100]
                    print(f"  → {block.name}({args_preview})")

                result_content = self._execute(block.name, block.input)

                if verbose:
                    text_preview = next(
                        (c["text"][:120] for c in result_content if c.get("type") == "text"),
                        "",
                    ).replace("\n", " ")
                    print(f"     ↳ {text_preview}")

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_content,
                })

            messages = messages + [
                {"role": "assistant", "content": final_msg.content},
                {"role": "user", "content": tool_results},
            ]

        yield "\n\n[Agent reached max iterations without a final answer.]"

    def query(self, question: str, verbose: bool = False) -> str:
        return "".join(self.query_stream(question, verbose))
