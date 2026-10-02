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
import time

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
# chunks go in as a batch file and audio.cpp runs them in one session, writing
# each chunk's wav as it finishes. Progress is read back from those files (see
# _run) because that is a real completion signal: stdout is block-buffered when
# it is not a tty, so its timing lines can arrive all at once at the end.
#
# 12 rather than 4 because Pocket is the slowest engine here (RTF ~0.9), so a
# 4-chunk story leaves the progress bar frozen for most of a minute between
# ticks. Measured at 4 / 8 / 12 / 16 steps: 32.5 / 31.3 / 28.9 / 30.1s -- more
# chunks are not slower, because each line's cost is dominated by the audio it
# produces, and shorter lines mean shorter attention contexts.
CHUNK_STEPS = int(os.getenv("POCKET_CHUNK_STEPS", "12"))
TIMEOUT = float(os.getenv("POCKET_TIMEOUT", "900"))
# How often to look at the output directory while the CLI runs. Fast enough to
# feel live, slow enough to be free.
POLL_SECONDS = 0.4


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

    def _run(self, chunks: list[str], voice: str, workdir: str, on_chunk=None) -> list[str]:
        """Run the batch, reporting how many chunks are done. Returns their paths."""
        batch_file = os.path.join(workdir, "chunks.txt")
        with open(batch_file, "w", encoding="utf-8") as fp:
            fp.write("\n".join(chunks) + "\n")
        out_dir = os.path.join(workdir, "out")
        os.makedirs(out_dir, exist_ok=True)
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
        ]
        log_path = os.path.join(workdir, "audiocpp.log")
        # stdout/stderr go to a file rather than a pipe: the CLI emits its
        # progress lines on stdout, but a pipe would leave us either guessing
        # when each line finished or deadlocking on a full pipe buffer.
        #
        # No start_new_session: the CLI stays in our process group so Ctrl+C
        # reaches it. Detaching it made the binary immune to the interrupt that
        # stops the server, which left it burning both cores for minutes with
        # nobody waiting for the result.
        with open(log_path, "w", encoding="utf-8") as log:
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            try:
                deadline = time.time() + TIMEOUT
                completed = 0
                timed_out = False
                while proc.poll() is None:
                    done = self._completed_chunks(out_dir)
                    if done > completed:
                        completed = done
                        if on_chunk:
                            on_chunk(done)
                    if time.time() > deadline:
                        timed_out = True
                        break
                    time.sleep(POLL_SECONDS)
            except BaseException:
                # A cancelled or failed synthesis must not leave the model
                # running: it would hold the generation lock's cores until it
                # finished on its own.
                proc.kill()
                raise
            finally:
                # Also covers the timeout: without killing first, `wait()` on a
                # still-running child blocks for the rest of the generation.
                if proc.poll() is None:
                    proc.kill()
                proc.wait()

        # Final sweep: the last chunk(s) can land between the last poll and exit.
        done = self._completed_chunks(out_dir)
        if done > completed and on_chunk:
            on_chunk(done)

        if timed_out:
            raise RuntimeError(f"audio.cpp timed out after {TIMEOUT:.0f}s")
        # No merged file is requested: the chunks are joined here so the pause
        # between them can follow punctuation, like the other engines. So the
        # success test is "the CLI exited clean and produced chunk files".
        if proc.returncode != 0 or done == 0:
            tail = ""
            try:
                with open(log_path, "r", encoding="utf-8", errors="replace") as log:
                    tail = log.read()[-500:]
            except OSError:
                pass
            raise RuntimeError(
                f"audio.cpp failed ({proc.returncode}): {tail.strip()}"
            )
        return [self._chunk_path(out_dir, i) for i in range(1, done + 1)]

    @staticmethod
    def _chunk_path(out_dir: str, index: int) -> str:
        return os.path.join(out_dir, f"line_{index}.wav")

    @classmethod
    def _completed_chunks(cls, out_dir: str) -> int:
        """How many chunk wavs exist so far (highest index, in case of a gap).

        Counted as the highest N present rather than the number of files so a
        half-written `line_2.wav` that has not been renamed cannot make progress
        jump past work that is still running.
        """
        highest = 0
        try:
            names = os.listdir(out_dir)
        except OSError:
            return 0
        for name in names:
            if not (name.startswith("line_") and name.endswith(".wav")):
                continue
            try:
                index = int(name[len("line_"): -len(".wav")])
            except ValueError:
                continue
            if index > highest and os.path.exists(cls._chunk_path(out_dir, index)):
                highest = index
        return highest

    def _synthesize(self, text: str, voice: str, on_progress) -> np.ndarray:
        chunks = chunk_text(text, CHUNK_STEPS)
        total_chars = max(1, sum(len(c) for c in chunks))
        pieces: list[np.ndarray] = []
        with tempfile.TemporaryDirectory(prefix="pocket-tts-") as workdir:
            # Report progress from the CLI's own completion signal, mapped back
            # onto characters so the percentage matches the other engines (a
            # bare "chunk 3 of 4" would mean something different on a text with
            # unevenly sized paragraphs).
            def report(done: int) -> None:
                if not on_progress:
                    return
                chars = sum(len(c) for c in chunks[:done])
                on_progress(min(chars / total_chars, 0.99) * 100.0)

            paths = self._run(chunks, voice, workdir, on_chunk=report)
            for i, path in enumerate(paths):
                samples, sr = sf.read(path, dtype="float32", always_2d=False)
                if samples.ndim > 1:
                    samples = samples.mean(axis=1)
                if i:
                    # audio.cpp's --batch-merge-audio concat joins the chunks
                    # with no gap at all, which runs two sentences together.
                    # Every other engine pauses on punctuation here, so the
                    # joins are built here instead of using the merged file.
                    pause = boundary_pause(chunks[min(i, len(chunks)) - 1])
                    pieces.append(np.zeros(round(sr * pause), dtype=np.float32))
                pieces.append(samples.astype(np.float32))
        return np.clip(np.concatenate(pieces), -1.0, 1.0).astype(np.float32)