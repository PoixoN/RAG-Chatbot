"""CLI: ingest, ask, test."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx

from .config import get_settings
from .ingest_pipeline import run_full_ingest
from .rag import answer_question

TEST_QUESTIONS = [
    "What are the production 'Do's' for RAG?",
    "What is the difference between standard retrieval and the ColPali approach?",
    "Why is hybrid search better than vector-only search?",
]


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def cmd_ingest() -> None:
    settings = get_settings()
    if not settings.knowledge_dir.is_dir():
        raise SystemExit(f"Knowledge directory not found: {settings.knowledge_dir}")
    print(f"Knowledge dir: {settings.knowledge_dir}", file=sys.stderr)
    if settings.ingest_file_filters:
        print(f"Filters: {settings.ingest_file_filters}", file=sys.stderr)
    stats = run_full_ingest(settings)
    print(
        f"Done. PPTX files: {stats['pptx_files']}, videos: {stats['video_files']}, "
        f"chunks: {stats['total_chunks']}"
    )


def cmd_ask(question: str) -> None:
    settings = get_settings()
    try:
        with httpx.Client(timeout=300.0) as client:
            answer, _ = answer_question(question, settings, client=client)
    except RuntimeError as e:
        raise SystemExit(str(e)) from e
    except httpx.HTTPError as e:
        raise SystemExit(f"HTTP error (is Ollama running?): {e}") from e
    print(answer)


def cmd_test(log_path: Path | None) -> None:
    settings = get_settings()
    root = _project_root()
    out = log_path or (root / "logs" / "answers.log")
    out.parent.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [f"\n--- test run {ts} ---\n"]

    with httpx.Client(timeout=600.0) as client:
        for q in TEST_QUESTIONS:
            print(f"Q: {q}", file=sys.stderr)
            try:
                answer, _ = answer_question(q, settings, client=client)
            except (httpx.HTTPError, RuntimeError) as e:
                print(f"Error: {e}", file=sys.stderr)
                answer = f"[error] {e}"
            print(answer)
            print("", file=sys.stderr)
            lines.append(f"Question: {q}\nAnswer:\n{answer}\n\n")

    text = "".join(lines)
    with open(out, "a", encoding="utf-8") as f:
        f.write(text)
    print(f"Appended to {out}", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Local RAG chatbot (PPTX + video → Chroma → Ollama)",
    )
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("ingest", help="Load PPTX + transcribe MP4, chunk, embed, store in Chroma")

    pa = sub.add_parser("ask", help="Ask one question")
    pa.add_argument("question", help="Question text")

    pt = sub.add_parser("test", help="Run three assignment questions; append to log file")
    pt.add_argument(
        "--log",
        type=Path,
        default=None,
        help="Log file (default: logs/answers.log)",
    )
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "ingest":
        cmd_ingest()
    elif args.command == "ask":
        cmd_ask(args.question)
    elif args.command == "test":
        cmd_test(args.log)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
