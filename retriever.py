"""
Retrieval layer: searches both ChromaDB collections and merges results.

Text query → sentence-transformer embedding → text_docs search
Text query → CLIP text embedding           → image_docs search (cross-modal)

Returns a unified list of RetrievedDoc objects sorted by relevance distance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import chromadb

import config
from embeddings import embed_texts, embed_texts_clip


@dataclass
class RetrievedDoc:
    id: str
    text: str                    # chunk text or image caption
    source: str                  # file path string
    doc_type: str                # "text" | "image_caption" | "image"
    distance: float
    image_path: str | None = None


def _get_collections() -> tuple[chromadb.Collection, chromadb.Collection]:
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    text_col = client.get_or_create_collection(
        name=config.TEXT_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    image_col = client.get_or_create_collection(
        name=config.IMAGE_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    return text_col, image_col


def retrieve(query: str, top_k: int = config.TOP_K) -> list[RetrievedDoc]:
    """
    Search text_docs with sentence-transformer and image_docs with CLIP.
    Returns merged, deduplicated results sorted by distance (ascending = more relevant).
    """
    text_col, image_col = _get_collections()
    results: list[RetrievedDoc] = []
    seen_image_paths: set[str] = set()

    # ── Text collection search ─────────────────────────────────────────────
    if text_col.count() > 0:
        text_embedding = embed_texts([query])[0]
        text_hits = text_col.query(
            query_embeddings=[text_embedding],
            n_results=min(top_k, text_col.count()),
            include=["documents", "metadatas", "distances"],
        )
        for doc, meta, dist in zip(
            text_hits["documents"][0],
            text_hits["metadatas"][0],
            text_hits["distances"][0],
        ):
            image_path = meta.get("image_path")
            results.append(RetrievedDoc(
                id=meta.get("source", "") + str(meta.get("chunk", "")),
                text=doc,
                source=meta.get("source", ""),
                doc_type=meta.get("type", "text"),
                distance=dist,
                image_path=image_path,
            ))
            if image_path:
                seen_image_paths.add(image_path)

    # ── Image collection search (CLIP cross-modal) ─────────────────────────
    if image_col.count() > 0:
        clip_embedding = embed_texts_clip([query])[0]
        image_hits = image_col.query(
            query_embeddings=[clip_embedding],
            n_results=min(top_k, image_col.count()),
            include=["documents", "metadatas", "distances"],
        )
        for doc, meta, dist in zip(
            image_hits["documents"][0],
            image_hits["metadatas"][0],
            image_hits["distances"][0],
        ):
            image_path = meta.get("image_path", meta.get("source", ""))
            # skip if this image was already returned via caption search
            if image_path in seen_image_paths:
                continue
            results.append(RetrievedDoc(
                id=meta.get("source", ""),
                text=doc,
                source=meta.get("source", ""),
                doc_type="image",
                distance=dist,
                image_path=image_path,
            ))
            seen_image_paths.add(image_path)

    # sort by distance (lower = more similar in cosine space)
    results.sort(key=lambda r: r.distance)
    return results[:top_k]
