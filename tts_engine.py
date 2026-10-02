"""Compatibility shim: the TTS entry point now lives in the ``tts`` package.

``from tts import synth_wav`` is the real API. This module is kept because
external scripts and older imports reach for ``tts_engine``, and because the
engine/voice selection is a *runtime* choice (``tts.configure``) rather than
which module you imported.
"""

from tts import (  # noqa: F401
    configure,
    engine_ids,
    get_engine,
    selected,
    status,
    synth_samples,
    synth_wav,
    voice_list,
    warm_up,
)

__all__ = [
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
