# Local RAG chatbot (PPTX + video)

End-to-end RAG pipeline: extract text from PowerPoint decks, transcribe lecture videos with **faster-whisper**, chunk and embed with **sentence-transformers**, store in **ChromaDB**, answer with **Ollama** (no cloud APIs).

## Requirements

- **Python 3.10–3.12** (3.14+ may fail to install `torch` / `tokenizers` wheels yet—use 3.12 if you see build errors).
- **ffmpeg** on `PATH` (for extracting audio from `.mp4`).
- **Ollama** installed and running; at least one chat model pulled (`ollama pull …`).

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env`:

- **`OLLAMA_MODEL`**: must match a name from `ollama list` (for example `llama3.2` or another model you have installed).
- **`INGEST_FILE_FILTERS`**: optional comma-separated substrings to limit which files under `knowledge/` are ingested. Example for a quick run (slides only): `Databases for GenAI.pptx` (matches that deck, excludes videos that do not contain that substring). For the full course corpus, leave it empty.

Place your `.pptx` and `.mp4` files under `knowledge/` (large files are gitignored by default).

## Usage

Build or refresh the vector index (this **resets** the Chroma collection):

```bash
python -m src.cli ingest
```

Ask a single question:

```bash
python -m src.cli ask "Why is hybrid search better than vector-only search?"
```

Run the three assignment test questions and append results to `logs/answers.log`:

```bash
python -m src.cli test
```

## Docker (optional)

The `Dockerfile` builds an image with Python 3.12 and ffmpeg. Ollama is expected on the host; set `OLLAMA_BASE_URL` to `http://host.docker.internal:11434` on macOS/Windows when running the container.

```bash
docker build -t rag-chatbot .
docker run --rm -e OLLAMA_BASE_URL=http://host.docker.internal:11434 -e OLLAMA_MODEL=your-model \
  -v "$(pwd)/knowledge:/app/knowledge" -v "$(pwd)/chroma_db:/app/chroma_db" rag-chatbot \
  python -m src.cli ingest
```

## GitHub submission

Include code and `requirements.txt`. Do not commit multi-gigabyte media: keep `knowledge/*.pptx` and `knowledge/*.mp4` ignored, and document that graders copy files locally and run `ingest`. Commit a sample `logs/answers.log` from `python -m src.cli test` if required.
