"""
RAG tool schemas and implementations for the agentic loop.

Tools handled here (direct Python, no MCP):
  - search_knowledge_base
  - list_sources
  - ingest_file

MCP tools (fetch_url, web_search, wikipedia) are loaded at runtime
from the MCP server via mcp_client.list_mcp_tools().
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import chromadb

import config
from embeddings import embed_texts, embed_texts_clip


# ── RAG tool schemas ───────────────────────────────────────────────────────

RAG_TOOL_SCHEMAS: list[dict] = [
    {
        "name": "search_knowledge_base",
        "description": (
            "Search the local vector database for relevant text and images. "
            "Use search_type='text' for text/PDF content, 'image' for visual CLIP search, "
            "or 'both' to cover all sources. Always call this first before answering."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"},
                "search_type": {
                    "type": "string",
                    "enum": ["text", "image", "both"],
                    "description": "Which collection to search (default: both)",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results (1-20, default: 5)",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "list_sources",
        "description": "List all files currently indexed in the knowledge base.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "ingest_file",
        "description": (
            "Ingest a local file (text, PDF, or image) into the knowledge base "
            "so future queries can retrieve it."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute or relative path to the file",
                }
            },
            "required": ["file_path"],
        },
    },
]


# ── Helpers ────────────────────────────────────────────────────────────────

def _get_collections() -> tuple[chromadb.Collection, chromadb.Collection]:
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    text_col = client.get_or_create_collection(
        name=config.TEXT_COLLECTION, metadata={"hnsw:space": "cosine"}
    )
    image_col = client.get_or_create_collection(
        name=config.IMAGE_COLLECTION, metadata={"hnsw:space": "cosine"}
    )
    return text_col, image_col


# ── Tool implementations ───────────────────────────────────────────────────

def tool_search_knowledge_base(
    query: str, search_type: str = "both", top_k: int = 5
) -> str:
    top_k = max(1, min(20, top_k))
    text_col, image_col = _get_collections()
    results: list[dict] = []
    seen_image_paths: set[str] = set()

    if search_type in ("text", "both") and text_col.count() > 0:
        emb = embed_texts([query])[0]
        hits = text_col.query(
            query_embeddings=[emb],
            n_results=min(top_k, text_col.count()),
            include=["documents", "metadatas", "distances"],
        )
        for doc, meta, dist in zip(
            hits["documents"][0], hits["metadatas"][0], hits["distances"][0]
        ):
            image_path = meta.get("image_path")
            results.append(
                {
                    "source": Path(meta.get("source", "")).name,
                    "type": meta.get("type", "text"),
                    "distance": round(dist, 4),
                    "content": doc[:600],
                    "image_path": image_path,
                }
            )
            if image_path:
                seen_image_paths.add(image_path)

    if search_type in ("image", "both") and image_col.count() > 0:
        clip_emb = embed_texts_clip([query])[0]
        hits = image_col.query(
            query_embeddings=[clip_emb],
            n_results=min(top_k, image_col.count()),
            include=["documents", "metadatas", "distances"],
        )
        for doc, meta, dist in zip(
            hits["documents"][0], hits["metadatas"][0], hits["distances"][0]
        ):
            image_path = meta.get("image_path", meta.get("source", ""))
            if image_path in seen_image_paths:
                continue
            results.append(
                {
                    "source": Path(meta.get("source", "")).name,
                    "type": "image_clip",
                    "distance": round(dist, 4),
                    "content": doc[:400],
                    "image_path": image_path,
                }
            )
            seen_image_paths.add(image_path)

    if not results:
        return "No results found in the knowledge base."

    results.sort(key=lambda r: r["distance"])
    return json.dumps(results[:top_k], indent=2)


def tool_list_sources() -> str:
    text_col, image_col = _get_collections()
    sources: dict[str, dict] = {}

    for col in (text_col, image_col):
        if col.count() == 0:
            continue
        all_docs = col.get(include=["metadatas"])
        for meta in all_docs["metadatas"]:
            src = meta.get("source", "unknown")
            key = Path(src).name
            if key not in sources:
                sources[key] = {"file": key, "path": src, "types": set(), "chunks": 0}
            sources[key]["types"].add(meta.get("type", "?"))
            sources[key]["chunks"] += 1

    if not sources:
        return "Knowledge base is empty."

    out = [
        {"file": v["file"], "types": sorted(v["types"]), "chunks": v["chunks"]}
        for v in sources.values()
    ]
    return json.dumps(out, indent=2)


def tool_ingest_file(file_path: str) -> str:
    from ingest import ingest_single_file
    try:
        ingest_single_file(file_path)
        return f"Successfully ingested: {file_path}"
    except FileNotFoundError:
        return f"File not found: {file_path}"
    except Exception as e:
        return f"Ingest error: {e}"


# ── Dispatch ───────────────────────────────────────────────────────────────

def execute_rag_tool(name: str, inputs: dict) -> Any:
    """Execute a RAG tool by name. Returns a string result."""
    if name == "search_knowledge_base":
        return tool_search_knowledge_base(**inputs)
    elif name == "list_sources":
        return tool_list_sources()
    elif name == "ingest_file":
        return tool_ingest_file(**inputs)
    return f"Unknown RAG tool: {name}"
