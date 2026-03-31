"""ChromaDB persistent store with sentence-transformers embeddings."""

from __future__ import annotations

import re

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils import embedding_functions

from .chunking import chunk_text
from .config import Settings


def _slug(s: str, max_len: int = 80) -> str:
    s = re.sub(r"[^\w\-]+", "_", s, flags=re.UNICODE)
    return s[:max_len].strip("_") or "doc"


def build_embedding_function(settings: Settings):
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=settings.embedding_model,
    )


def get_collection(settings: Settings, *, reset: bool = False):
    """Return Chroma collection; optionally delete and recreate."""
    settings.chroma_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(settings.chroma_path),
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    ef = build_embedding_function(settings)
    name = settings.collection_name

    if reset:
        try:
            client.delete_collection(name)
        except Exception:
            pass

    return client.get_or_create_collection(name=name, embedding_function=ef)


def build_chunks_from_text(
    text: str,
    settings: Settings,
    *,
    source: str,
    file_name: str,
) -> tuple[list[str], list[dict], list[str]]:
    """Split text into chunks; return (documents, metadatas, ids)."""
    parts = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
    documents: list[str] = []
    metadatas: list[dict] = []
    ids: list[str] = []
    base = _slug(file_name)
    for i, doc in enumerate(parts):
        documents.append(doc)
        metadatas.append(
            {
                "source": source,
                "file": file_name,
                "chunk_index": i,
            }
        )
        ids.append(f"{source}_{base}_{i}")
    return documents, metadatas, ids
