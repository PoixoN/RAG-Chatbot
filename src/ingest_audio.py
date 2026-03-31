"""Transcribe MP4/video using ffmpeg + faster-whisper."""

from __future__ import annotations

import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

from faster_whisper import WhisperModel

from .config import Settings


def _check_ffmpeg() -> None:
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            check=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        raise RuntimeError(
            "ffmpeg is required on PATH to extract audio from video. "
            "Install: https://ffmpeg.org/download.html"
        ) from e


def extract_audio_wav(video_path: Path, wav_path: Path) -> None:
    """Extract mono 16 kHz WAV for Whisper."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(wav_path),
    ]
    subprocess.run(cmd, capture_output=True, check=True)


def transcribe_video(
    video_path: Path,
    settings: Settings,
    *,
    on_progress: Callable[[int], None] | None = None,
) -> str:
    """Return full transcript text for one video file."""
    _check_ffmpeg()

    model = WhisperModel(
        settings.whisper_model,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
    )

    with tempfile.TemporaryDirectory(prefix="rag_audio_") as tmp:
        wav = Path(tmp) / "audio.wav"
        extract_audio_wav(video_path, wav)
        segments, _info = model.transcribe(
            str(wav),
            beam_size=5,
            vad_filter=True,
        )
        lines: list[str] = []
        for i, seg in enumerate(segments):
            lines.append(seg.text.strip())
            if on_progress and (i + 1) % 50 == 0:
                on_progress(i + 1)
        return "\n".join(lines)
