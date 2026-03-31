"""Retrieve context from Chroma and generate an answer with Ollama."""

from __future__ import annotations

import json
from typing import Any

import httpx

from .config import Settings
from .vectorstore import get_collection


def retrieve_context(question: str, settings: Settings) -> tuple[str, list[dict[str, Any]]]:
    """Return formatted context string and raw retrieval metadata."""
    col = get_collection(settings, reset=False)
    res = col.query(query_texts=[question], n_results=settings.top_k)
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0] if res.get("distances") else None

    blocks: list[str] = []
    raw: list[dict[str, Any]] = []
    for i, doc in enumerate(docs):
        meta = metas[i] if i < len(metas) else {}
        d = dists[i] if dists is not None and i < len(dists) else None
        label = i + 1
        blocks.append(f"[{label}] (source={meta.get('source', '?')}, file={meta.get('file', '?')})\n{doc}")
        raw.append({"rank": label, "chunk": doc, "metadata": meta, "distance": d})

    return "\n\n".join(blocks), raw


SYSTEM_PROMPT = """You are a teaching assistant for an AI Academy course. Answer using ONLY the provided context excerpts. If the context does not contain enough information, say so clearly and suggest what topic might be missing. Be concise and structured when appropriate."""


def answer_question(
    question: str,
    settings: Settings,
    *,
    client: httpx.Client | None = None,
) -> tuple[str, str]:
    """Return (answer_text, prompt_used_for_debug)."""
    context, _ = retrieve_context(question, settings)
    user_content = f"""Context from course materials:

{context}

Question: {question}
"""

    url = f"{settings.ollama_base_url}/api/chat"
    payload = {
        "model": settings.ollama_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "stream": False,
    }

    own_client = client is None
    c = client or httpx.Client(timeout=300.0)
    try:
        r = c.post(url, json=payload)
        if r.status_code == 404:
            raise RuntimeError(
                f"Ollama returned 404 for model {settings.ollama_model!r}. "
                "Run `ollama list` and set OLLAMA_MODEL in .env to an installed name."
            )
        r.raise_for_status()
        data = r.json()
        msg = data.get("message") or {}
        text = msg.get("content") or data.get("response") or json.dumps(data)
        return text.strip(), user_content
    finally:
        if own_client:
            c.close()
