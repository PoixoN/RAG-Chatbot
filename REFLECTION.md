## Reflection

The most challenging part of this project was balancing **end-to-end correctness** with **practical runtime**: extracting clean text from slides (including tables and grouped shapes), then handling **large video files** where full transcription on CPU can take a long time. Choosing a chunking window and overlap that preserve slide boundaries without creating tiny or giant chunks also required a few iterations. Integrating Chroma with a local embedding model was straightforward compared to tuning ingest scope (full corpus vs a filtered subset for faster feedback).

From this work I reinforced how **retrieval quality** depends on the entire pipeline—transcription errors, missed slide text, and chunk boundaries all surface in the final answers. I also got clearer experience wiring a **fully local** stack (Whisper-class STT, sentence embeddings, Ollama) so the project can run without API keys, at the cost of caring about disk space, RAM, and model choice for the generator.
