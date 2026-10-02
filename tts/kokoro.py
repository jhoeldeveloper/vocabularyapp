"""Kokoro-82M through ONNX Runtime (kokoro-onnx).

Replaces the old PyTorch ``kokoro`` pipeline. ONNX Runtime is much lighter than
the CUDA-enabled torch build the old code pulled in, which is what was maxing
out the CPU on low-power hardware.

Inference runs entirely through ``onnxruntime`` (never PyTorch). The default
weights are the **fp32** ``model.onnx`` export: on the target CPU (Intel
i5-6200U, Skylake — no VNNI/AVX-512) int8 quantization is *slower* than fp32
(ORT falls back to dequantizing every layer), so fp32 is both faster and
higher quality here. The quantized ``q8f16`` export is kept available as a
low-RAM option via the ``KOKORO_MODEL`` env var.

**A full-blast synthesis is expected to show ~100% CPU on this machine**, and
that is not a bug: measured with a 230-phoneme paragraph at ``intra=2``,
fp32 runs at RTF 0.85 (13.3s of audio in 11.2s of wall time) while consuming
1.88 cores — it is a real-time neural vocoder on two physical cores, so both
of them are busy for the whole synthesis and idle the rest of the time. The
knobs, all measured on the i5-6200U (wall / cpu seconds for the same text):

======================  ==============  ==============
``intra_op`` threads    wall            cpu
======================  ==============  ==============
1                      16.2s / 17.3s   1 core  (RTF 1.22)
**2 (default)**         **11.2s**       **1.88 cores (RTF 0.85)**
4                      13.3s           2.96 cores (RTF 1.00)
======================  ==============  ==============

Weight variants at ``intra=2``: fp32 11.2s, fp16 13.5s, q8f16 31.9s — int8
and fp16 both *lose* here, because without VNNI the CPU has no fast int8 or
native fp16 path and ORT converts on the fly. Setting
``KOKORO_INTRA_OP_THREADS=1`` is therefore the only supported way to halve the
CPU, and it costs ~1.45x the wall time; it is a trade, not an optimization.
If the machine feels hot the fix is that trade, not a different model.

Kept as the default engine because it is the most expressive of the three on
English prose, not because it is the fastest: :mod:`tts.inflect` measured RTF
0.34 on the same text and the same two cores (see :mod:`tts`).
"""
from __future__ import annotations

import os
import sys
import threading
import time

import numpy as np
import onnxruntime as rt
from kokoro_onnx import Kokoro

from .base import Engine, Voice, resolve_path

# ---------------------------------------------------------------------------
# Configuration (all overridable via environment for easy A/B testing)
# ---------------------------------------------------------------------------
# Directory that holds the ONNX model + voices bundle (large, gitignored).
# Every path here goes through resolve_path(), so a relative value in .env is
# resolved against the project root rather than whatever directory the server
# happened to be started from.
MODELS_DIR = resolve_path(
    os.getenv(
        "KOKORO_MODELS_DIR",
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models",
            "onnx",
        ),
    )
)

# fp32 weights (model.onnx) by default, which is why KOKORO_MODEL does not need
# to be in .env at all: with no override this resolves to the file
# download_models.sh puts there. On the i5-6200U (no VNNI) fp32 is both faster
# and higher quality than the int8-quantized exports; point this at
# model_q8f16.onnx (or run MODEL_VARIANT=q8f16 ./download_models.sh) for the
# low-RAM A/B.
KOKORO_MODEL = resolve_path(
    os.getenv("KOKORO_MODEL", os.path.join(MODELS_DIR, "model.onnx"))
)

# Single bundled voices file (one-per-voice `.bin` won't work with kokoro-onnx).
KOKORO_VOICES = resolve_path(
    os.getenv("KOKORO_VOICES", os.path.join(MODELS_DIR, "voices-v1.0.bin"))
)

# CPU is the only provider we want on this low-power machine.
ONNX_PROVIDER = os.getenv("ONNX_PROVIDER", "CPUExecutionProvider")

# ---------------------------------------------------------------------------
# ONNX Runtime thread tuning for the Intel Core i5-6200U.
#
# The i5-6200U has 2 *physical* cores / 4 logical threads (hyperthreading),
# no AVX-512, AVX2 only. ORT's default is to use ALL logical threads, which
# saturates all 4 and makes the two real cores fight over shared resources
# (hyperthreading contention) -> higher latency, not lower. Pinning to the
# physical core count (2) keeps one thread per real core and actually finishes
# faster here. Inter-op threads are kept at 1 because the Kokoro graph is a
# single linear sequence of ops: there is nothing to parallelize across ops,
# and extra inter-op threads would only add scheduling overhead.
#
# These are the defaults for every ORT-backed engine (Kokoro and Inflect), so
# they live here and are imported rather than re-read per engine.
ORT_INTRA_OP_NUM_THREADS = int(os.getenv("KOKORO_INTRA_OP_THREADS", "2"))
ORT_INTER_OP_NUM_THREADS = int(os.getenv("KOKORO_INTER_OP_THREADS", "1"))

# How many `kokoro.create()` calls one synthesis is split into. Each call
# costs a fixed ~0.6s on top of the per-phoneme work, so a story split into 30
# paragraphs pays 18s of pure call overhead; the library already batches
# internally (510 phonemes per inference) and inserts the sentence pauses, so
# handing it several paragraphs at once is strictly less work. Measured on the
# i5-6200U, 6 paragraphs of ~90 words: 6 calls 24.8s / 45.1s cpu, 2 calls
# 23.2s / 42.3s cpu, 1 call 23.6s / 42.7s cpu. Fewer calls than ~4 buys
# nothing more, and each call is a progress tick the UI can show, so the
# default trades the last few percent of CPU for a progress bar that moves.
KOKORO_PROGRESS_STEPS = int(os.getenv("KOKORO_PROGRESS_STEPS", "4"))

SAMPLE_RATE = 24000

# Map the leading letter of a voice id to the language code kokoro-onnx uses
# for grapheme-to-phoneme. Default is American English.
_LANG_BY_VOICE_PREFIX = {
    "a": "en-us",
    "b": "en-gb",
    "j": "ja",
    "z": "zh",
    "f": "fr-fr",
    "i": "it",
    "p": "pt-br",
    "s": "es",
    "h": "hi",
    "k": "ko",
    "n": "nl",
    "r": "ru",
}

_kokoro: Kokoro | None = None
_kokoro_lock = threading.Lock()


def ort_session_options() -> rt.SessionOptions:
    """Session options shared by every ORT-backed engine on this machine."""
    so = rt.SessionOptions()
    # Cap threads to the physical core count (see note above).
    so.intra_op_num_threads = ORT_INTRA_OP_NUM_THREADS
    so.inter_op_num_threads = ORT_INTER_OP_NUM_THREADS
    so.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
    # Stated explicitly rather than left to the default: measured on the
    # i5-6200U (fp32, 2 threads, same paragraph) at ORT_ENABLE_ALL 12.8s,
    # ORT_ENABLE_BASIC 15.7s, ORT_DISABLE_ALL 13.3s. ENABLE_ALL is also the
    # default, so this is a comment that can be trusted, not a behaviour change.
    so.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
    return so


def _providers() -> list[str] | None:
    if ONNX_PROVIDER in rt.get_available_providers():
        return [ONNX_PROVIDER]
    return None


def _ensure_kokoro() -> Kokoro:
    global _kokoro
    if _kokoro is None:
        with _kokoro_lock:
            if _kokoro is None:
                if not os.path.exists(KOKORO_MODEL):
                    raise FileNotFoundError(
                        f"Kokoro ONNX model not found at {KOKORO_MODEL}. "
                        "Run ./download_models.sh (or set KOKORO_MODEL)."
                    )
                if not os.path.exists(KOKORO_VOICES):
                    raise FileNotFoundError(
                        f"Kokoro voices file not found at {KOKORO_VOICES}. "
                        "Run ./download_models.sh (or set KOKORO_VOICES)."
                    )
                providers = _providers()
                print(
                    f"[TTS:kokoro] Loading model={KOKORO_MODEL} "
                    f"voices={KOKORO_VOICES} provider={providers or 'default'} "
                    f"intra_threads={ORT_INTRA_OP_NUM_THREADS} "
                    f"inter_threads={ORT_INTER_OP_NUM_THREADS}",
                    file=sys.stderr,
                    flush=True,
                )
                session = rt.InferenceSession(
                    KOKORO_MODEL, ort_session_options(), providers=providers
                )
                _kokoro = Kokoro.from_session(session, KOKORO_VOICES)
    return _kokoro


def _lang_for_voice(voice: str) -> str:
    prefix = (voice or "af_heart")[:1]
    return _LANG_BY_VOICE_PREFIX.get(prefix, "en-us")


def _split_paragraphs(text: str) -> list[str]:
    """Split on blank lines only (safe: never breaks a sentence mid-word)."""
    parts = [p.strip() for p in text.split("\n") if p.strip()]
    return parts or [text]


def _chunk_paragraphs(paragraphs: list[str], steps: int) -> list[list[str]]:
    """Group paragraphs into about `steps` chunks of roughly equal length.

    Balanced by characters rather than by count, so a story whose first
    paragraph is three times the length of the rest does not finish half its
    work in the first tick. Only whole paragraphs are ever grouped: the
    progress percentage is computed from paragraph characters, so a chunk that
    did not exist in the input would make the reported position a lie.
    """
    if steps <= 1 or len(paragraphs) <= 1:
        return [paragraphs]
    total = sum(len(p) for p in paragraphs)
    chunks: list[list[str]] = []
    current: list[str] = []
    acc = 0
    for para in paragraphs:
        current.append(para)
        acc += len(para)
        # Close the chunk once it has carried its share of the text, but only
        # while there are enough paragraphs left to fill the remaining steps.
        wanted = (len(chunks) + 1) / steps
        if len(chunks) < steps - 1 and acc >= total * wanted:
            chunks.append(current)
            current = []
    if current:
        chunks.append(current)
    return chunks


class KokoroEngine(Engine):
    id = "kokoro"
    label = "Kokoro 82M (ONNX)"
    note = "Most expressive of the three; slowest on this CPU (RTF ~0.9)."
    sample_rate = SAMPLE_RATE

    def unavailable_reason(self) -> str:
        if not os.path.exists(KOKORO_MODEL):
            return f"Kokoro model missing ({KOKORO_MODEL}); run ./download_models.sh"
        if not os.path.exists(KOKORO_VOICES):
            return f"Kokoro voices missing ({KOKORO_VOICES}); run ./download_models.sh"
        return ""

    def voices(self) -> list[Voice]:
        # The voices bundle is a .npz loaded by kokoro-onnx; listing it needs
        # the model loaded, which is a 325MB read. The well-known ids are
        # listed instead so the picker is instant and works before warm-up.
        known = [
            ("af_heart", "en-us"), ("af_bella", "en-us"), ("af_nicole", "en-us"),
            ("af_sarah", "en-us"), ("af_sky", "en-us"), ("am_adam", "en-us"),
            ("am_michael", "en-us"), ("bf_emma", "en-gb"), ("bf_isabella", "en-gb"),
            ("bm_george", "en-gb"), ("bm_lewis", "en-gb"),
        ]
        return [Voice(id=v, lang=lang) for v, lang in known]

    def default_voice(self) -> str:
        # No environment variable, deliberately. A voice is chosen in the UI and
        # stored in the settings table; an env fallback used to exist and was a
        # landmine rather than a convenience -- `TTS_VOICE=en-US-JennyNeural`
        # (a leftover edge-tts name) survived in .env for months and would have
        # raised "Voice not found" on the first synthesis. One source of truth
        # beats a configurable one that can disagree with it.
        return "af_heart"

    def _synthesize(self, text: str, voice: str, on_progress) -> np.ndarray:
        kokoro = _ensure_kokoro()
        lang = _lang_for_voice(voice)
        paragraphs = _split_paragraphs(text)
        chunks = _chunk_paragraphs(paragraphs, KOKORO_PROGRESS_STEPS)
        total_chars = max(1, sum(len(p) for p in paragraphs))

        print(
            f"[TTS:kokoro] Synthesizing (paras={len(paragraphs)} "
            f"calls={len(chunks)}) voice={voice} lang={lang}",
            file=sys.stderr,
            flush=True,
        )

        segments: list[np.ndarray] = []
        processed_chars = 0
        start = time.time()
        for i, chunk in enumerate(chunks):
            # One create() per chunk: it phonemizes, batches at 510 phonemes and
            # inserts the sentence pauses for everything handed to it.
            samples, _ = kokoro.create(
                "\n".join(chunk), voice=voice, speed=1.0, lang=lang
            )
            segments.append(samples)
            processed_chars += sum(len(p) for p in chunk)
            pct = processed_chars / total_chars
            if on_progress:
                on_progress(pct * 100.0)
            print(
                f"[TTS:kokoro]   chunk {i + 1}/{len(chunks)} ready "
                f"({len(chunk)} para) | {pct * 100:5.1f}% | "
                f"{int(time.time() - start)}s elapsed",
                file=sys.stderr,
                flush=True,
            )
        return np.concatenate(segments, axis=0).astype(np.float32)