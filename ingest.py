"""
Ingest text files, PDFs, and images into ChromaDB.

Text pipeline:
  file → chunks → sentence-transformer embeddings → text_docs collection

Image pipeline:
  image → Claude vision caption → text_docs collection (caption)
         → CLIP image embedding  → image_docs collection (for visual search)
"""
from __future__ import annotations

import base64
import hashlib
import uuid
from pathlib import Path
from typing import Generator

import chromadb
from tqdm import tqdm

import config
from embeddings import embed_texts, embed_image


# ── ChromaDB setup ─────────────────────────────────────────────────────────

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


# ── Text chunking ──────────────────────────────────────────────────────────

def _chunk_text(text: str, size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i : i + size])
        if chunk:
            chunks.append(chunk)
        i += size - overlap
    return chunks


def _file_id(path: Path) -> str:
    return hashlib.md5(str(path.resolve()).encode()).hexdigest()


# ── Text / PDF ingestion ───────────────────────────────────────────────────

def _read_text_file(path: Path) -> str:
    return path.read_text(errors="replace")


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def ingest_text_file(path: Path, text_col: chromadb.Collection) -> int:
    suffix = path.suffix.lower()
    if suffix in config.TEXT_EXTENSIONS:
        raw = _read_text_file(path)
    elif suffix in config.PDF_EXTENSIONS:
        raw = _read_pdf(path)
    else:
        return 0

    chunks = _chunk_text(raw)
    if not chunks:
        return 0

    file_id = _file_id(path)
    ids = [f"{file_id}_{i}" for i in range(len(chunks))]
    embeddings = embed_texts(chunks)
    metadatas = [{"source": str(path), "type": "text", "chunk": i} for i in range(len(chunks))]

    # upsert so re-ingesting is idempotent
    text_col.upsert(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
    return len(chunks)


# ── Image ingestion ────────────────────────────────────────────────────────

def _caption_image_with_claude(path: Path) -> str:
    """Generate a rich text description of an image using Claude vision."""
    import anthropic

    with open(path, "rb") as f:
        img_data = base64.standard_b64encode(f.read()).decode()

    suffix = path.suffix.lower().lstrip(".")
    media_type_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
                      "gif": "image/gif", "webp": "image/webp", "bmp": "image/png"}
    media_type = media_type_map.get(suffix, "image/jpeg")

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    response = client.messages.create(
        model=config.CLAUDE_MODEL,
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": media_type, "data": img_data},
                },
                {
                    "type": "text",
                    "text": (
                        "Describe this image in detail. Include: main subjects, actions, "
                        "colors, text visible, spatial layout, and any notable context. "
                        "Be thorough — this description will be used for search indexing."
                    ),
                },
            ],
        }],
    )
    return response.content[0].text


def ingest_image_file(path: Path, text_col: chromadb.Collection, image_col: chromadb.Collection) -> int:
    file_id = _file_id(path)

    print(f"  captioning {path.name} with Claude...")
    caption = _caption_image_with_claude(path)

    # store caption in text collection for text-based retrieval
    text_col.upsert(
        ids=[f"{file_id}_caption"],
        embeddings=embed_texts([caption]),
        documents=[caption],
        metadatas=[{"source": str(path), "type": "image_caption", "image_path": str(path)}],
    )

    # store CLIP image embedding in image collection for visual retrieval
    image_embedding = embed_image(str(path))
    image_col.upsert(
        ids=[file_id],
        embeddings=[image_embedding],
        documents=[caption],
        metadatas=[{"source": str(path), "type": "image", "image_path": str(path)}],
    )

    return 1


# ── Directory ingestion ────────────────────────────────────────────────────

def ingest_directory(directory: str | Path) -> None:
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    text_col, image_col = _get_collections()

    all_files = list(directory.rglob("*"))
    text_files = [f for f in all_files if f.suffix.lower() in config.TEXT_EXTENSIONS | config.PDF_EXTENSIONS]
    image_files = [f for f in all_files if f.suffix.lower() in config.IMAGE_EXTENSIONS]

    total_chunks = 0
    total_images = 0

    if text_files:
        print(f"\nIngesting {len(text_files)} text/PDF file(s)...")
        for path in tqdm(text_files, unit="file"):
            n = ingest_text_file(path, text_col)
            total_chunks += n
            tqdm.write(f"  {path.name}: {n} chunks")

    if image_files:
        print(f"\nIngesting {len(image_files)} image file(s)...")
        for path in tqdm(image_files, unit="img"):
            n = ingest_image_file(path, text_col, image_col)
            total_images += n

    print(f"\nDone. {total_chunks} text chunks + {total_images} images ingested.")
    print(f"Text collection: {text_col.count()} docs | Image collection: {image_col.count()} docs")


def ingest_single_file(file_path: str | Path) -> None:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    text_col, image_col = _get_collections()

    suffix = path.suffix.lower()
    if suffix in config.IMAGE_EXTENSIONS:
        n = ingest_image_file(path, text_col, image_col)
        print(f"Ingested image: {path.name} ({n} entry)")
    elif suffix in config.TEXT_EXTENSIONS | config.PDF_EXTENSIONS:
        n = ingest_text_file(path, text_col)
        print(f"Ingested text: {path.name} ({n} chunks)")
    else:
        print(f"Unsupported file type: {suffix}")
