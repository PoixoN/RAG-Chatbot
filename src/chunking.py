"""Split long text into overlapping chunks (paragraph-aware, then fixed windows)."""

from __future__ import annotations


def _windows(text: str, chunk_size: int, overlap: int) -> list[str]:
    if not text.strip():
        return []
    chunk_size = max(chunk_size, 1)
    overlap = min(max(overlap, 0), chunk_size - 1) if chunk_size > 1 else 0
    step = max(chunk_size - overlap, 1)
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        piece = text[i : i + chunk_size]
        if piece.strip():
            out.append(piece)
        if i + chunk_size >= n:
            break
        i += step
    return out


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split on blank lines first, then window each segment."""
    if not text or not text.strip():
        return []

    blocks = [b.strip() for b in text.split("\n\n\n") if b.strip()]
    if not blocks:
        blocks = [text.strip()]

    chunks: list[str] = []
    for block in blocks:
        if len(block) <= chunk_size:
            chunks.append(block)
        else:
            # sub-split on double newlines
            sub = [s.strip() for s in block.split("\n\n") if s.strip()]
            if len(sub) <= 1:
                chunks.extend(_windows(block, chunk_size, overlap))
            else:
                buf = ""
                for s in sub:
                    if len(buf) + len(s) + 2 <= chunk_size:
                        buf = buf + ("\n\n" if buf else "") + s
                    else:
                        if buf:
                            if len(buf) <= chunk_size:
                                chunks.append(buf)
                            else:
                                chunks.extend(_windows(buf, chunk_size, overlap))
                        buf = s
                if buf:
                    if len(buf) <= chunk_size:
                        chunks.append(buf)
                    else:
                        chunks.extend(_windows(buf, chunk_size, overlap))

    # merge tiny fragments into neighbors
    merged: list[str] = []
    for c in chunks:
        c = c.strip()
        if not c:
            continue
        if merged and len(merged[-1]) + len(c) + 1 < chunk_size // 4:
            merged[-1] = merged[-1] + "\n\n" + c
        else:
            merged.append(c)
    return merged
