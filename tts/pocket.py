"""Pocket TTS (Kyutai 100M) through the audio.cpp CPU runtime.

A GGUF-converted autoregressive model with 26 built-in English voice embeddings,
driven through the prebuilt ``audiocpp_cli`` binary rather than a Python
runtime. Two consequences worth knowing before choosing it:

* **It is the smallest download but not the fastest here.** q8 weights are
  128MB against Inflect's 37.7MB, yet measured RTF on the i5-6200U is **1.05**
  against Inflect's 0.34 and Kokoro's 0.85 -- an autoregressive model decodes
  one step at a time and cannot use the two cores the way a big non-recurrent
  batch can. It also needs ~900MB RSS, more than either alternative.
* **It is the only engine with real voice choice.** 26 English voices
  (``alba``, ``anna``, ``charles``, ...) ship as ~6MB safetensors embeddings,
  and voice cloning from a reference wav is available.

The model is reached through ``audio-cpp/audio.cpp-gguf``, whose conversion is
public. Upstream ``kyutai/pocket-tts`` is a gated repo, so the weights are not
vendored here: ``./download_models.sh alt`` fetches the GGUF and the embeddings.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

from .base import Engine, Voice, boundary_pause, chunk_text, resolve_path

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# resolve_path on all three: a relative POCKET_DIR/AUDIO_CPP_CLI/POCKET_MODEL in
# .env means "inside the project", not "relative to wherever uvicorn was started".
ALT_DIR = resolve_path(os.getenv("POCKET_DIR", os.path.join(_ROOT, "models", "alt", "pocket")))
AUDIO_CPP_CLI = resolve_path(os.getenv("AUDIO_CPP_CLI", os.path.join(_ROOT, "models", "alt", "audiocpp_cli")))
MODEL = resolve_path(os.getenv("POCKET_MODEL", os.path.join(ALT_DIR, "pocket-tts-english-q8_0.gguf")))
EMBEDDINGS = os.path.join(ALT_DIR, "embeddings")

# One CLI invocation per chunk *would* reload the model each time (~2s), so the
# chunks go in as a batch file and audio.cpp runs them in a single session.
# Progress comes from the per-line wavs it writes, not from parsing its logs.
CHUNK_STEPS = int(os.getenv("POCKET_CHUNK_STEPS", "4"))
TIMEOUT = float(os.getenv("POCKET_TIMEOUT", "900"))


class PocketEngine(Engine):
    id = "pocket"
    label = "Pocket TTS 100M (GGUF)"
    note = "26 built-in English voices; smallest download, slowest of the three here."
    sample_rate = 24000

    def unavailable_reason(self) -> str:
        if not os.path.exists(MODEL):
            return f"Pocket TTS model missing ({MODEL}); run ./download_models.sh alt"
        if not os.path.isfile(AUDIO_CPP_CLI):
            return f"audio.cpp CLI missing ({AUDIO_CPP_CLI}); run ./download_models.sh alt"
        if not os.path.isdir(EMBEDDINGS):
            return f"Pocket voice embeddings missing ({EMBEDDINGS}); run ./download_models.sh alt"
        return ""

    def voices(self) -> list[Voice]:
        if not os.path.isdir(EMBEDDINGS):
            return [Voice(id="alba")]
        ids = sorted(
            os.path.splitext(f)[0]
            for f in os.listdir(EMBEDDINGS)
            if f.endswith(".safetensors")
        )
        return [Voice(id=v) for v in ids] or [Voice(id="alba")]

    def default_voice(self) -> str:
        # No environment variable here either -- see the note in tts/kokoro.py.
        # "alba" is the voice audio.cpp's own model spec names as the default,
        # and it is the one guaranteed to be present in every download.
        listed = [v.id for v in self.voices()]
        return "alba" if "alba" in listed else (listed[0] if listed else "alba")

    def _run(self, chunks: list[str], voice: str, workdir: str) -> list[str]:
        batch_file = os.path.join(workdir, "chunks.txt")
        with open(batch_file, "w", encoding="utf-8") as fp:
            fp.write("\n".join(chunks) + "\n")
        out_dir = os.path.join(workdir, "out")
        os.makedirs(out_dir, exist_ok=True)
        merged = os.path.join(workdir, "merged.wav")
        cmd = [
            AUDIO_CPP_CLI,
            "--task", "tts",
            "--family", "pocket_tts",
            "--model", MODEL,
            "--backend", "cpu",
            # Same thread budget as the ORT engines: the physical core count,
            # not ORT's all-logical-threads default (see tts/kokoro.py).
            "--threads", str(os.getenv("KOKORO_INTRA_OP_THREADS", "2")),
            "--voice-id", voice,
            "--batch-text-file", batch_file,
            "--out-dir", out_dir,
            "--batch-merge-audio", "concat",
            "--out", merged,
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=TIMEOUT, check=False
        )
        if result.returncode != 0 or not os.path.exists(merged):
            raise RuntimeError(
                f"audio.cpp failed ({result.returncode}): "
                f"{(result.stderr or result.stdout or '').strip()[-500:]}"
            )
        return [merged]

    def _synthesize(self, text: str, voice: str, on_progress) -> np.ndarray:
        chunks = chunk_text(text, CHUNK_STEPS)
        total_chars = max(1, sum(len(c) for c in chunks))
        pieces: list[np.ndarray] = []
        with tempfile.TemporaryDirectory(prefix="pocket-tts-") as workdir:
            paths = self._run(chunks, voice, workdir)
            done = 0
            for i, path in enumerate(paths):
                samples, sr = sf.read(path, dtype="float32", always_2d=False)
                if samples.ndim > 1:
                    samples = samples.mean(axis=1)
                if i:
                    pause = boundary_pause(chunks[min(i, len(chunks)) - 1])
                    pieces.append(
                        np.zeros(round(sr * pause), dtype=np.float32)
                    )
                pieces.append(samples.astype(np.float32))
                done += len(chunks[i]) if i < len(chunks) else 0
                if on_progress:
                    on_progress(min(done / total_chars, 0.99) * 100.0)
        return np.clip(np.concatenate(pieces), -1.0, 1.0).astype(np.float32)