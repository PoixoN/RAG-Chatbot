"""Orchestrate PPTX + video ingestion into Chroma."""

from __future__ import annotations

from pathlib import Path

from .config import Settings
from .ingest_audio import transcribe_video
from .ingest_pptx import extract_pptx_text
from .vectorstore import build_chunks_from_text, get_collection


def _match_filters(name: str, filters: list[str]) -> bool:
    if not filters:
        return True
    return any(f in name for f in filters)


def _list_files(root: Path, suffix: str, filters: list[str]) -> list[Path]:
    if not root.is_dir():
        return []
    out: list[Path] = []
    for p in sorted(root.glob(f"*{suffix}")):
        if p.is_file() and _match_filters(p.name, filters):
            out.append(p)
    return out


def run_full_ingest(settings: Settings) -> dict[str, int]:
    """
    Ingest all matching PPTX and MP4 under knowledge_dir.
    Resets the collection first.
    Returns counts: pptx_files, video_files, total_chunks.
    """
    knowledge = settings.knowledge_dir
    filters = settings.ingest_file_filters

    pptx_paths = _list_files(knowledge, ".pptx", filters)
    mp4_paths = _list_files(knowledge, ".mp4", filters)

    all_docs: list[str] = []
    all_meta: list[dict] = []
    all_ids: list[str] = []

    for path in pptx_paths:
        text = extract_pptx_text(path)
        if not text.strip():
            continue
        d, m, i = build_chunks_from_text(
            text,
            settings,
            source="pptx",
            file_name=path.name,
        )
        all_docs.extend(d)
        all_meta.extend(m)
        all_ids.extend(i)

    for path in mp4_paths:
        print(f"Transcribing {path.name} ...", flush=True)
        text = transcribe_video(path, settings)
        if not text.strip():
            continue
        d, m, i = build_chunks_from_text(
            text,
            settings,
            source="video",
            file_name=path.name,
        )
        all_docs.extend(d)
        all_meta.extend(m)
        all_ids.extend(i)

    if not all_docs:
        get_collection(settings, reset=True)
        return {
            "pptx_files": len(pptx_paths),
            "video_files": len(mp4_paths),
            "total_chunks": 0,
        }

    col = get_collection(settings, reset=True)
    # Chroma add in batches to avoid huge payloads
    batch = 128
    for start in range(0, len(all_docs), batch):
        end = min(start + batch, len(all_docs))
        col.add(
            ids=all_ids[start:end],
            documents=all_docs[start:end],
            metadatas=all_meta[start:end],
        )

    return {
        "pptx_files": len(pptx_paths),
        "video_files": len(mp4_paths),
        "total_chunks": len(all_docs),
    }
