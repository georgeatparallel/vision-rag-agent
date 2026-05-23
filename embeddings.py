"""
Two embedding spaces:
  - sentence-transformers (384-dim) for text chunks → text_docs collection
  - CLIP (512-dim) for images + cross-modal text queries → image_docs collection
"""
from __future__ import annotations
from functools import lru_cache


@lru_cache(maxsize=1)
def _text_model():
    from sentence_transformers import SentenceTransformer
    from config import TEXT_EMBED_MODEL
    print(f"[embeddings] loading sentence-transformers: {TEXT_EMBED_MODEL}")
    return SentenceTransformer(TEXT_EMBED_MODEL)


@lru_cache(maxsize=1)
def _clip():
    import warnings
    from config import CLIP_MODEL_ID
    print(f"[embeddings] loading CLIP: {CLIP_MODEL_ID}")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*slow image processor.*")
        warnings.filterwarnings("ignore", message=".*use_fast.*")
        from transformers import CLIPProcessor, CLIPModel
        model = CLIPModel.from_pretrained(CLIP_MODEL_ID)
        processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    model.eval()
    return model, processor


# ── Text embeddings (sentence-transformers) ────────────────────────────────

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed text chunks for the text_docs collection."""
    model = _text_model()
    return model.encode(texts, show_progress_bar=False, normalize_embeddings=True).tolist()


# ── Image embeddings (CLIP) ────────────────────────────────────────────────

def embed_image(image_path: str) -> list[float]:
    """Embed an image file for the image_docs collection."""
    import torch
    from PIL import Image

    model, processor = _clip()
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        features = model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
    return features[0].cpu().numpy().tolist()


def embed_texts_clip(texts: list[str]) -> list[list[float]]:
    """Embed text queries using CLIP for cross-modal image retrieval."""
    import torch

    model, processor = _clip()
    inputs = processor(
        text=texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77,
    )
    with torch.no_grad():
        features = model.get_text_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy().tolist()
