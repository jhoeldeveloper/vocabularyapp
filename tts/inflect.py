"""Inflect-Micro-v2 through ONNX Runtime — the fast one.

Two fp32 ONNX graphs (37.7MB total) replace the 325MB Kokoro session, and on the
i5-6200U they are **~2.5x quicker on the same two cores**:

===================  ======  =======  =====
engine               RTF    cores    audio model
===================  ======  =======  =====
kokoro (fp32)        0.85   1.88     325MB
**inflect (fp32)**   **0.34**  **1.88**  **37.7MB**
pocket (q8 GGUF)     1.05   ~1.9     128MB + voices
===================  ======  =======  =====

Measured on the same paragraph through :class:`Engine.synth`; ``cores`` is
CPU-seconds per wall-second, so 1.88 means both physical cores, which is what
Kokoro does too. Inflect wins on work-per-second, not on cores spent: it is a
VITS model with a 100-step-ish flow and a small decoder, where Kokoro runs a
full transformer over every phoneme in a 510-token batch.

Quality is the trade: Inflect is a compact single-speaker voice, flatter and
less expressive than Kokoro's cast, and it has exactly one voice. Nothing
quantized is released for it (the upstream export is verified fp32 only), which
matters here only because there is no VNNI to make int8 worth having.

Model files: ``./download_models.sh alt`` (Apache-2.0, from
``owensong/Inflect-Micro-v2-ONNX``). The text frontend is upstream Python that
uses eSpeak-ng, vendored under ``tts/_vendor/inflect`` with its LICENSE.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import onnxruntime as rt

from .base import Engine, Voice, boundary_pause, chunk_text, resolve_path
from .kokoro import ort_session_options

VENDOR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_vendor", "inflect")
MODEL_DIR = resolve_path(
    os.getenv(
        "INFLECT_MODEL_DIR",
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models",
            "alt",
            "inflect",
        ),
    )
)
SAMPLE_RATE = 24000

# How many synthesis calls one story is split into, and therefore how often the
# progress bar moves. A VITS call has a fixed setup cost, so this is a balance,
# not a free dial -- but measured at 4 / 8 / 12 / 16 steps the wall time is flat
# (12.4 / 11.8 / 12.7 / 12.8s on the same paragraph), because a call's cost is
# dominated by the audio it produces rather than by its own overhead. 8 is the
# point where the ticks stop getting closer together (the sentence splitter
# bottoms out at ~5 chunks for a typical paragraph), so going higher only
# fragments the text.
PROGRESS_STEPS = int(os.getenv("INFLECT_PROGRESS_STEPS", "8"))

# Upstream's defaults, kept so the output matches their published samples.
SPEED = float(os.getenv("INFLECT_SPEED", "1.0"))
VARIATION = float(os.getenv("INFLECT_VARIATION", "0.667"))
SEED = int(os.getenv("INFLECT_SEED", "0"))


def _load_vendor():
    """Import the vendored frontend and symbol table.

    The symbol table is loaded by path rather than as ``runtime.text.symbols``
    because that package is literally named ``text``: importing it puts a
    directory named ``text`` on ``sys.path`` ahead of the standard library one
    and shadows stdlib ``text`` for the whole process.
    """
    if VENDOR_DIR not in sys.path:
        sys.path.insert(0, VENDOR_DIR)
    from inflect_vits_frontend import run_vits_frontend  # noqa: E402

    spec = importlib.util.spec_from_file_location(
        "_inflect_symbols", os.path.join(VENDOR_DIR, "runtime", "text", "symbols.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return run_vits_frontend, {s: i for i, s in enumerate(module.symbols)}


def _edge_fade(waveform: np.ndarray, milliseconds: float = 5.0) -> np.ndarray:
    """5ms fade at both ends.

    VITS output starts and ends on a non-zero sample often enough that the
    click is audible on a story that starts straight after a play/pause.
    """
    frames = min(round(SAMPLE_RATE * milliseconds / 1000.0), waveform.size // 2)
    if frames <= 0:
        return waveform
    out = waveform.copy()
    ramp = np.linspace(0.0, 1.0, frames, endpoint=True, dtype=np.float32)
    out[:frames] *= ramp
    out[-frames:] *= ramp[::-1]
    return out


class InflectEngine(Engine):
    id = "inflect"
    label = "Inflect Micro v2 (ONNX)"
    note = "Fastest and smallest here (RTF ~0.34, 37.7MB); single flat voice."
    sample_rate = SAMPLE_RATE

    def __init__(self) -> None:
        super().__init__()
        self._duration = None
        self._decode = None
        self._frontend = None
        self._symbol_to_id = None

    def unavailable_reason(self) -> str:
        for name in ("duration.onnx", "decode.onnx"):
            if not os.path.exists(os.path.join(MODEL_DIR, "onnx", name)):
                return (
                    f"Inflect model missing ({name} under {MODEL_DIR}); "
                    "run ./download_models.sh alt"
                )
        return ""

    def voices(self) -> list[Voice]:
        return [Voice(id="default", label="Inflect Micro v2 (only voice)")]

    def _ensure(self):
        if self._decode is None:
            reason = self.unavailable_reason()
            if reason:
                raise FileNotFoundError(reason)
            providers = (
                ["CPUExecutionProvider"]
                if "CPUExecutionProvider" in rt.get_available_providers()
                else None
            )
            onnx_dir = os.path.join(MODEL_DIR, "onnx")
            print(
                f"[TTS:inflect] Loading {onnx_dir} "
                f"intra_threads={rt.SessionOptions().intra_op_num_threads or 'default'}",
                file=sys.stderr,
                flush=True,
            )
            so = ort_session_options()
            self._duration = rt.InferenceSession(
                os.path.join(onnx_dir, "duration.onnx"), so, providers=providers
            )
            self._decode = rt.InferenceSession(
                os.path.join(onnx_dir, "decode.onnx"), so, providers=providers
            )
            self._frontend, self._symbol_to_id = _load_vendor()
        return self._frontend, self._symbol_to_id

    def _synthesize_chunk(self, text: str, seed: int) -> np.ndarray:
        frontend, symbol_to_id = self._ensure()
        phoneme_text = frontend(text).phoneme_text
        sequence = [symbol_to_id[s] for s in phoneme_text if s in symbol_to_id]
        if not sequence:
            raise ValueError(f"No phonemes of {text!r} are in the model vocabulary")
        # VITS wants the symbol sequence interleaved with blank tokens.
        tokens = np.zeros(len(sequence) * 2 + 1, dtype=np.int64)
        tokens[1::2] = sequence

        m_p_exp, logs_p_exp, y_mask = self._duration.run(
            ["m_p_exp", "logs_p_exp", "y_mask"],
            {
                "tokens": tokens[None, :],
                "lengths": np.asarray([tokens.shape[0]], dtype=np.int64),
                "length_scale": np.asarray(1.0 / SPEED, dtype=np.float32),
            },
        )
        rng = np.random.default_rng(seed)
        latent_noise = rng.standard_normal(m_p_exp.shape, dtype=np.float32)
        waveform = self._decode.run(
            ["waveform"],
            {
                "m_p_exp": m_p_exp,
                "logs_p_exp": logs_p_exp,
                "y_mask": y_mask,
                "zp_noise": latent_noise,
                "noise_scale": np.asarray(VARIATION, dtype=np.float32),
            },
        )[0]
        return _edge_fade(np.asarray(waveform, dtype=np.float32).reshape(-1))

    def _synthesize(self, text: str, voice: str, on_progress) -> np.ndarray:
        chunks = chunk_text(text, PROGRESS_STEPS)
        total_chars = max(1, sum(len(c) for c in chunks))
        pieces: list[np.ndarray] = []
        processed = 0
        for i, chunk in enumerate(chunks):
            if i:
                # A pause at the join, sized by how the previous chunk ended,
                # so a paragraph break does not run two sentences together.
                pause = boundary_pause(chunks[i - 1])
                pieces.append(np.zeros(round(SAMPLE_RATE * pause), dtype=np.float32))
            pieces.append(self._synthesize_chunk(chunk, SEED + i))
            processed += len(chunk)
            if on_progress:
                on_progress(processed / total_chars * 100.0)
        return np.clip(np.concatenate(pieces), -1.0, 1.0).astype(np.float32)