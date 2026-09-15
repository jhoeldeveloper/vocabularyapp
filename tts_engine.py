"""Local TTS via Kokoro-82M running on ONNX Runtime (kokoro-onnx).

Replaces the old PyTorch ``kokoro`` pipeline. ONNX Runtime is much lighter than
the CUDA-enabled torch build the old code pulled in, which is what was maxing
out the CPU on low-power hardware.

Inference runs entirely through ``onnxruntime`` (never PyTorch). The default
weights are the **fp32** ``model.onnx`` export: on the target CPU (Intel
i5-6200U, Skylake — no VNNI/AVX-512) int8 quantization is *slower* than fp32
(ORT falls back to dequantizing every layer), so fp32 is both faster and
higher quality here. The quantized ``q8f16`` export is kept available as a
low-RAM option via the ``KOKORO_MODEL`` env var.
"""
import io
import os
import sys
import time
import threading
import numpy as np
import soundfile as sf
import onnxruntime as rt
from kokoro_onnx import Kokoro

# ---------------------------------------------------------------------------
# Configuration (all overridable via environment for easy A/B testing)
# ---------------------------------------------------------------------------
# Directory that holds the ONNX model + voices bundle (large, gitignored).
MODELS_DIR = os.getenv(
    "KOKORO_MODELS_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "onnx"),
)

# fp32 weights (model.onnx) by default. On the i5-6200U (no VNNI) fp32 is both
# faster and higher quality than the int8-quantized exports. For a low-RAM A/B,
# point this at model_q8f16.onnx (or run MODEL_VARIANT=q8f16 ./download_models.sh).
KOKORO_MODEL = os.getenv("KOKORO_MODEL", os.path.join(MODELS_DIR, "model.onnx"))

# Single bundled voices file (one per-voice `.bin` won't work with kokoro-onnx).
KOKORO_VOICES = os.getenv(
    "KOKORO_VOICES", os.path.join(MODELS_DIR, "voices-v1.0.bin")
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
ORT_INTRA_OP_NUM_THREADS = int(os.getenv("KOKORO_INTRA_OP_THREADS", "2"))
ORT_INTER_OP_NUM_THREADS = int(os.getenv("KOKORO_INTER_OP_THREADS", "1"))
ORT_EXECUTION_MODE = rt.ExecutionMode.ORT_SEQUENTIAL

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

_kokoro = None
_kokoro_lock = threading.Lock()
_gen_lock = threading.Lock()  # kokoro-onnx generation is not thread-safe -> serialize


def _build_session_options() -> rt.SessionOptions:
    so = rt.SessionOptions()
    # Cap threads to the physical core count (see note above).
    so.intra_op_num_threads = ORT_INTRA_OP_NUM_THREADS
    so.inter_op_num_threads = ORT_INTER_OP_NUM_THREADS
    so.execution_mode = ORT_EXECUTION_MODE
    return so


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
                providers = (
                    [ONNX_PROVIDER]
                    if ONNX_PROVIDER in rt.get_available_providers()
                    else None
                )
                print(
                    f"[TTS] Loading Kokoro ONNX model={KOKORO_MODEL} "
                    f"voices={KOKORO_VOICES} provider={providers or 'default'} "
                    f"intra_threads={ORT_INTRA_OP_NUM_THREADS} "
                    f"inter_threads={ORT_INTER_OP_NUM_THREADS}",
                    file=sys.stderr,
                    flush=True,
                )
                so = _build_session_options()
                session = rt.InferenceSession(KOKORO_MODEL, so, providers=providers)
                _kokoro = Kokoro.from_session(session, KOKORO_VOICES)
    return _kokoro


def _lang_for_voice(voice: str) -> str:
    prefix = (voice or "af_heart")[:1]
    return _LANG_BY_VOICE_PREFIX.get(prefix, "en-us")


def _split_paragraphs(text: str) -> list[str]:
    """Split on blank lines only (safe: never breaks a sentence mid-word)."""
    parts = [p.strip() for p in text.split("\n") if p.strip()]
    return parts or [text]


def synth_wav(text: str, voice: str, on_progress=None) -> bytes:
    """Synthesize `text` with Kokoro and return WAV file bytes.

    `on_progress(pct: float)` is called after each generated paragraph (0-100),
    preserving the old per-segment progress contract.
    """
    if not text or not text.strip():
        raise ValueError("empty text")

    kokoro = _ensure_kokoro()
    lang = _lang_for_voice(voice)
    paragraphs = _split_paragraphs(text)
    total_chars = max(1, sum(len(p) for p in paragraphs))

    print(
        f"[TTS] Synthesizing text (paras={len(paragraphs)}) with voice={voice} "
        f"lang={lang}",
        file=sys.stderr,
        flush=True,
    )
    if on_progress:
        on_progress(0.0)

    segments: list[np.ndarray] = []
    processed_chars = 0
    start = time.time()
    with _gen_lock:
        for i, para in enumerate(paragraphs):
            samples, sr = kokoro.create(para, voice=voice, speed=1.0, lang=lang)
            segments.append(samples)
            processed_chars += len(para)
            pct = processed_chars / total_chars * 100.0
            if on_progress:
                on_progress(pct)
            print(
                f"[TTS]   paragraph {i + 1}/{len(paragraphs)} ready | "
                f"{pct:5.1f}% | {int(time.time() - start)}s elapsed",
                file=sys.stderr,
                flush=True,
            )

    if not segments:
        raise RuntimeError("Kokoro produced no audio")
    audio = np.concatenate(segments, axis=0)
    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="WAV")
    total = time.time() - start
    print(
        f"[TTS] Done -> {len(audio)} samples "
        f"({len(audio)/SAMPLE_RATE:.2f}s) in {total:.1f}s",
        file=sys.stderr,
        flush=True,
    )
    if on_progress:
        on_progress(100.0)
    return buf.getvalue()
