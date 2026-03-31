"""Load settings from environment (.env supported)."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ChromaDB + posthog can log noisy telemetry errors depending on library versions.
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)


def _parse_filters(raw: str | None) -> list[str]:
    if not raw or not raw.strip():
        return []
    return [p.strip() for p in raw.split(",") if p.strip()]


@dataclass(frozen=True)
class Settings:
    knowledge_dir: Path
    chroma_path: Path
    collection_name: str
    embedding_model: str
    ollama_base_url: str
    ollama_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    whisper_model: str
    whisper_device: str
    whisper_compute_type: str
    ingest_file_filters: list[str]


def get_settings() -> Settings:
    root = Path(__file__).resolve().parent.parent
    kd = os.getenv("KNOWLEDGE_DIR", "knowledge")
    knowledge_dir = Path(kd) if Path(kd).is_absolute() else root / kd

    cp = os.getenv("CHROMA_PATH", "chroma_db")
    chroma_path = Path(cp) if Path(cp).is_absolute() else root / cp

    return Settings(
        knowledge_dir=knowledge_dir,
        chroma_path=chroma_path,
        collection_name=os.getenv("COLLECTION_NAME", "rag_knowledge"),
        embedding_model=os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip(
            "/"
        ),
        ollama_model=os.getenv("OLLAMA_MODEL", "llama3.2"),
        chunk_size=int(os.getenv("CHUNK_SIZE", "900")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "120")),
        top_k=int(os.getenv("TOP_K", "6")),
        whisper_model=os.getenv("WHISPER_MODEL", "small"),
        whisper_device=os.getenv("WHISPER_DEVICE", "cpu"),
        whisper_compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
        ingest_file_filters=_parse_filters(os.getenv("INGEST_FILE_FILTERS")),
    )
