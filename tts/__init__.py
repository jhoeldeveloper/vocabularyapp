"""One facade over the local TTS engines.

``from tts import synth_wav`` is the whole API the app uses. Which model runs is
a *runtime* choice, not an import-time one: :func:`configure` switches engine
and voice, and the settings are persisted by ``main.py`` so every client sees
the same pick.

The three engines, measured on the target i5-6200U (2 physical cores) on one
paragraph of story prose, all fp32/ORT or q8/GGUF:

=================  ======  =======  =======  ==================================
engine             RTF     cores    size     voices
=================  ======  =======  =======  ==================================
kokoro (default)   0.85    1.88     325 MB   11+ (af_heart, bf_emma, ...)
inflect            0.34    1.88      37.7 MB 1 (flat)
pocket             1.05    1.9      128 MB   26 (alba, anna, charles, ...)
=================  ======  =======  =======  ==================================

So **Inflect is the fast one and Kokoro the expressive one**, and they are not
close on speed. What none of them can do is use less CPU *and* be faster: cores
spent is ~1.9 in every row, because these are real neural vocoders and the cores
are the resource. Lower CPU is ``KOKORO_INTRA_OP_THREADS=1`` (shared by both ORT
engines, ~1.45x the wall time for Kokoro); for audio.cpp the same knob is
``--threads``, which :mod:`tts.pocket` passes through.
"""
from __future__ import annotations

import io
import os
import sys

import numpy as np
import soundfile as sf

from .base import Engine, EngineStatus, EngineUnavailable, Voice

SAMPLE_RATE = 24000

_ENGINES: dict[str, Engine] = {}
_current_engine = ""
_current_voice = ""

# Imported eagerly: all three are cheap to import (no model is loaded at import)
# and a lazy import would hide a broken optional dependency until the first
# synthesis, which is the worst moment to find out.
from .inflect import InflectEngine  # noqa: E402
from .kokoro import KokoroEngine  # noqa: E402
from .pocket import PocketEngine  # noqa: E402


def _registry() -> dict[str, Engine]:
    if not _ENGINES:
        _ENGINES.update(
            {
                KokoroEngine.id: KokoroEngine(),
                InflectEngine.id: InflectEngine(),
                PocketEngine.id: PocketEngine(),
            }
        )
    return _ENGINES


def engine_ids() -> list[str]:
    return list(_registry())


def get_engine(engine_id: str = "") -> Engine:
    registry = _registry()
    wanted = engine_id or _current_engine or os.getenv("TTS_ENGINE") or "kokoro"
    if wanted not in registry:
        raise KeyError(f"Unknown TTS engine {wanted!r}; have {sorted(registry)}")
    return registry[wanted]


def status() -> list[EngineStatus]:
    """Every engine with its availability and voices, for the UI picker."""
    out = []
    for engine in _registry().values():
        out.append(engine.status())
    return out


def selected() -> dict[str, str]:
    engine = get_engine()
    return {
        "engine": engine.id,
        "voice": current_voice() or engine.default_voice(),
    }


def current_voice() -> str:
    return _current_voice


def configure(engine: str = "", voice: str = "") -> dict[str, str]:
    """Select the engine and/or voice. Both are optional; empty means keep.

    Returns the effective selection so the caller can persist it. Raises
    ``KeyError`` for an unknown engine -- the API layer turns that into a 400
    rather than silently falling back, because a picker that quietly ignores
    the chosen model is worse than one that complains.
    """
    global _current_engine, _current_voice
    target = get_engine(engine) if engine else get_engine()
    if engine:
        _current_engine = target.id
    if voice:
        listed = {v.id for v in target.voices()}
        if listed and voice not in listed:
            raise ValueError(
                f"Unknown voice {voice!r} for {target.id}; have {sorted(listed)}"
            )
        _current_voice = voice
    return selected()


def voice_list(engine_id: str = "") -> list[Voice]:
    return get_engine(engine_id).voices()


def synth_wav(text: str, voice: str = "", on_progress=None) -> bytes:
    """Synthesize `text` with the selected engine and return WAV bytes.

    This is the function ``main.py`` calls; the signature is unchanged from the
    old single-engine ``tts_engine.synth_wav`` so the HTTP and WebSocket layers
    did not have to learn about engines at all.
    """
    samples = synth_samples(text, voice=voice, on_progress=on_progress)
    buf = io.BytesIO()
    sf.write(buf, samples, SAMPLE_RATE, format="WAV")
    return buf.getvalue()


def synth_samples(text: str, voice: str = "", on_progress=None) -> np.ndarray:
    """Synthesize and return float32 mono samples (used by the sample preview)."""
    engine = get_engine()
    return engine.synth(text, voice=voice, on_progress=on_progress)


def warm_up(text: str = "warm up", voice: str = "") -> None:
    """Load the selected engine and synthesize a couple of words.

    Called at startup so the first real story does not pay model load in front
    of the user. A failure is reported, never raised: the model can be missing
    on a fresh checkout and that must not stop the web app from booting.
    """
    try:
        engine = get_engine()
        reason = engine.unavailable_reason()
        if reason:
            print(f"[TTS] warm-up skipped: {reason}", file=sys.stderr, flush=True)
            return
        engine.synth(text, voice=voice)
        print(f"[TTS] warm-up complete ({engine.id})", file=sys.stderr, flush=True)
    except Exception as error:  # noqa: BLE001 - warm-up must never break boot
        print(
            f"[TTS] warm-up skipped: {error}",
            file=sys.stderr,
            flush=True,
        )


__all__ = [
    "Engine",
    "EngineStatus",
    "EngineUnavailable",
    "Voice",
    "configure",
    "engine_ids",
    "get_engine",
    "selected",
    "status",
    "synth_samples",
    "synth_wav",
    "voice_list",
    "warm_up",
]