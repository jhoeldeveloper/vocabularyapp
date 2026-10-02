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

# The engine/voice choice is made **per context**, not once globally. "Meanings"
# is the short word audio played by the per-cell Listen buttons; "Stories" is
# the long narration a story is read aloud with. They are genuinely different
# jobs: a word is one term where clarity beats expressiveness, a story is
# minutes of prose where the opposite holds -- and a user who wants crisp
# definitions should not have to give up a narrative voice for it. Forcing one
# global pick made the wrong choice for one of the two, always.
#
# Both selections live in the settings table as `tts_<context>_engine` /
# `tts_<context>_voice`.
CONTEXTS = ("meanings", "stories")
DEFAULT_CONTEXT = "stories"

_ENGINES: dict[str, Engine] = {}
_selection: dict[str, dict[str, str]] = {
    ctx: {"engine": "", "voice": ""} for ctx in CONTEXTS
}

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


def _check_context(context: str) -> str:
    context = context or DEFAULT_CONTEXT
    if context not in CONTEXTS:
        raise KeyError(f"Unknown TTS context {context!r}; have {list(CONTEXTS)}")
    return context


def get_engine(context: str = "", engine_id: str = "") -> Engine:
    """The engine for one context, or a specific one by id.

    `context` wins over the process default; `engine_id` is for the few places
    that need to inspect another engine (the picker listing every engine's
    voices) without changing anything.
    """
    registry = _registry()
    if engine_id:
        if engine_id not in registry:
            raise KeyError(f"Unknown TTS engine {engine_id!r}; have {sorted(registry)}")
        return registry[engine_id]
    wanted = _selection[_check_context(context)]["engine"] or os.getenv(
        "TTS_ENGINE"
    ) or "kokoro"
    if wanted not in registry:
        raise KeyError(f"Unknown TTS engine {wanted!r}; have {sorted(registry)}")
    return registry[wanted]


def status() -> list[EngineStatus]:
    """Every engine with its availability and voices, for the UI picker."""
    return [engine.status() for engine in _registry().values()]


def selected(context: str = "") -> dict[str, str]:
    """The effective selection for a context, with defaults filled in."""
    ctx = _check_context(context)
    engine = get_engine(ctx)
    voice = _selection[ctx]["voice"]
    listed = [v.id for v in engine.voices()]
    if not voice or voice not in listed:
        # The stored voice belongs to the *other* context's engine (or to
        # nothing, after a model swap): the engine's own default is what will
        # actually be spoken, so that is what gets reported.
        voice = engine.default_voice()
    return {"engine": engine.id, "voice": voice}


def current_voice(context: str = "") -> str:
    return _selection[_check_context(context)]["voice"]


def configure(
    engine: str = "", voice: str = "", context: str = DEFAULT_CONTEXT
) -> dict[str, str]:
    """Select the engine and/or voice for one context. Empty means keep.

    Returns the effective selection so the caller can persist it. Raises
    ``KeyError``/``ValueError`` for an unknown engine or a voice that engine
    does not have -- the API layer turns those into 400s rather than silently
    falling back, because a picker that quietly ignores the chosen model is
    worse than one that complains.
    """
    ctx = _check_context(context)
    target = get_engine(ctx, engine_id=engine) if engine else get_engine(ctx)
    if engine:
        _selection[ctx]["engine"] = target.id
    if voice:
        listed = [v.id for v in target.voices()]
        if listed and voice not in listed:
            raise ValueError(
                f"Unknown voice {voice!r} for {target.id}; have {sorted(listed)}"
            )
        _selection[ctx]["voice"] = voice
    return selected(ctx)


def set_selection(context: str, engine: str = "", voice: str = "") -> dict[str, str]:
    """Apply a selection resolved elsewhere (e.g. restored from settings).

    Unlike :func:`configure` this never validates and never raises: it is the
    startup path, where a stored value that no longer makes sense (an engine
    whose models were deleted, a voice that has since been renamed) must fall
    back to a usable default instead of stopping the app from booting.
    """
    ctx = _check_context(context)
    registry = _registry()
    if engine and engine in registry:
        _selection[ctx]["engine"] = engine
    if voice and voice in [v.id for v in get_engine(ctx).voices()]:
        _selection[ctx]["voice"] = voice
    return selected(ctx)


def voice_list(context: str = "", engine_id: str = "") -> list[Voice]:
    return get_engine(context, engine_id=engine_id).voices()


def synth_wav(
    text: str,
    voice: str = "",
    on_progress=None,
    context: str = DEFAULT_CONTEXT,
) -> bytes:
    """Synthesize `text` with the engine chosen for `context`; return WAV bytes.

    This is the function ``main.py`` calls; the signature only grew `context`,
    so the HTTP and WebSocket layers pass it through and nothing else changed.
    """
    samples = synth_samples(text, voice=voice, on_progress=on_progress, context=context)
    buf = io.BytesIO()
    sf.write(buf, samples, SAMPLE_RATE, format="WAV")
    return buf.getvalue()


def synth_samples(
    text: str,
    voice: str = "",
    on_progress=None,
    context: str = DEFAULT_CONTEXT,
) -> np.ndarray:
    """Synthesize and return float32 mono samples (used by the sample preview)."""
    engine = get_engine(context)
    return engine.synth(text, voice=voice, on_progress=on_progress)


def warm_up(text: str = "warm up", voice: str = "", context: str = DEFAULT_CONTEXT) -> None:
    """Load the selected engine for one context and synthesize a few words.

    Called at startup so the first real request does not pay model load in front
    of the user. A failure is reported, never raised: the model can be missing
    on a fresh checkout and that must not stop the web app from booting.
    """
    try:
        engine = get_engine(context)
        reason = engine.unavailable_reason()
        if reason:
            print(f"[TTS:{context}] warm-up skipped: {reason}", file=sys.stderr, flush=True)
            return
        engine.synth(text, voice=voice)
        print(
            f"[TTS:{context}] warm-up complete ({engine.id})",
            file=sys.stderr,
            flush=True,
        )
    except Exception as error:  # noqa: BLE001 - warm-up must never break boot
        print(
            f"[TTS:{context}] warm-up skipped: {error}",
            file=sys.stderr,
            flush=True,
        )


__all__ = [
    "CONTEXTS",
    "DEFAULT_CONTEXT",
    "Engine",
    "EngineStatus",
    "EngineUnavailable",
    "Voice",
    "configure",
    "current_voice",
    "engine_ids",
    "get_engine",
    "selected",
    "set_selection",
    "status",
    "synth_samples",
    "synth_wav",
    "voice_list",
    "warm_up",
]