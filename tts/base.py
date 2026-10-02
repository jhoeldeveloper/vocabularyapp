"""Shared plumbing for the local TTS engines.

Every engine is a separate class behind one tiny interface (:class:`Engine`),
so :mod:`tts` can offer "pick a model, pick a voice" without the rest of the app
knowing which model is loaded. What is deliberately *not* here: any engine
itself, and any WAV writing -- that lives in the engine modules and
:mod:`tts.wav`.

Two rules the engines share, both learned the hard way on the i5-6200U:

* **A full-blast synthesis saturates both physical cores and that is correct.**
  See the numbers in :mod:`tts.kokoro`. The engines below differ in *how much*
  work they do, not in whether they use the cores they are given.
* **Generation is serialized.** Only one synthesis runs at a time, so a queue of
  story requests cannot start four ONNX sessions' worth of work on two cores and
  make every one of them slower. The lock is process-wide and shared by all
  engines, so switching engine mid-flight cannot overlap either.
"""
from __future__ import annotations

import contextlib
import os
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

# The project root -- the directory holding main.py, i.e. this package's
# parent. Every model path is resolved against it, so a checkout can live
# anywhere and a relative path in .env means what it looks like it means.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_path(path: str) -> str:
    """Resolve a possibly-relative model path against the project root.

    **Never against the current working directory.** That is the footgun this
    exists to remove: `KOKORO_MODEL=models/onnx/model.onnx` works when uvicorn
    happens to be started from the project directory and fails with a bare
    "model not found" when it is started from anywhere else -- and the failure
    lands at warm-up, minutes after the typo, on a fresh checkout.
    """
    if not path:
        return path
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


@dataclass(frozen=True)
class Voice:
    """One selectable voice of an engine."""

    id: str
    label: str = ""
    lang: str = "en-us"

    def __post_init__(self) -> None:
        if not self.label:
            object.__setattr__(self, "label", self.id)


@dataclass
class EngineStatus:
    """What the UI needs to draw one engine in a picker."""

    id: str
    label: str
    available: bool
    reason: str = ""
    note: str = ""
    default_voice: str = ""
    voices: list[Voice] = field(default_factory=list)
    sample_rate: int = 24000


class EngineUnavailable(RuntimeError):
    """Raised when an engine is selected but its model files are not there."""


# One synthesis at a time. Held across the whole engine call, not per chunk:
# interleaving two syntheses on two cores is slower than running them in turn,
# and it keeps the "both cores busy" behaviour measurable.
_GEN_LOCK = threading.Lock()


def generation_lock() -> threading.Lock:
    return _GEN_LOCK


@contextlib.contextmanager
def low_priority(delta: int = 5):
    """Run the body at a lower CPU priority, then put it back.

    Synthesis legitimately saturates both physical cores (that is what makes it
    near-realtime), but the web server shares them: an unpinned worker starves
    the UI for the whole 20-60s of a story. Nice is per-thread on Linux, so this
    only touches the thread that holds the lock.

    The *applied* delta is measured rather than assumed: some sandboxes and
    cgroup policies clamp how far a thread may be reniced, so asking for +5 can
    move it +3. Restoring by the requested delta would then leave the thread at
    a higher priority than it started, and ``os.nice()`` returns the new value
    rather than the delta, so the two have to be reconciled rather than passed
    back and forth.
    """
    try:
        before = os.nice(0)
        applied = os.nice(delta) - before
    except (AttributeError, OSError):  # pragma: no cover - platform dependent
        yield
        return
    try:
        yield
    finally:
        try:
            os.nice(-applied)
        except OSError:  # pragma: no cover
            pass


def split_sentences(text: str, limit: int = 280) -> list[str]:
    """Split into sentence-ish chunks no longer than `limit` characters.

    Shared by the engines that do not ship their own chunker. Prefers to break
    at punctuation, then at a space, and never mid-word -- a cut word is read
    aloud as a typo.
    """
    import re

    normalized = " ".join(text.split())
    if not normalized:
        return []
    sentences = [
        part.strip()
        for part in re.split(r"(?<=[.!?;:])\s+", normalized)
        if part.strip()
    ]
    chunks: list[str] = []
    for sentence in sentences or [normalized]:
        while len(sentence) > limit:
            window = sentence[: limit + 1]
            punctuation = max(window.rfind(mark) for mark in (",", ";", ":"))
            cut = (
                punctuation + 1
                if punctuation >= limit // 2
                else sentence.rfind(" ", 0, limit + 1)
            )
            if cut < limit // 2:
                cut = limit
            chunks.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if sentence:
            chunks.append(sentence)
    return chunks


def boundary_pause(chunk: str) -> float:
    """Seconds of silence to insert after `chunk`, by how it ends."""
    ending = chunk.rstrip()[-1:] if chunk.strip() else ""
    return {
        "?": 0.28,
        "!": 0.24,
        ".": 0.22,
        ";": 0.16,
        ":": 0.13,
        ",": 0.09,
    }.get(ending, 0.08)


def chunk_text(text: str, steps: int) -> list[str]:
    """Group sentences into about `steps` character-balanced groups.

    Each group becomes one engine call. Calls are not free (fixed setup cost
    inside every engine) and each one is a progress tick the UI can show, so the
    job is to use few enough calls to pay setup once and enough of them for the
    progress bar to move.
    """
    sentences = split_sentences(text)
    if not sentences:
        return [text]
    if steps <= 1 or len(sentences) <= 1:
        return [" ".join(sentences)]
    total = sum(len(s) for s in sentences)
    chunks: list[str] = []
    current: list[str] = []
    acc = 0
    for sentence in sentences:
        current.append(sentence)
        acc += len(sentence)
        wanted = (len(chunks) + 1) / steps
        if len(chunks) < steps - 1 and acc >= total * wanted:
            chunks.append(" ".join(current))
            current = []
    if current:
        chunks.append(" ".join(current))
    return chunks


class Engine(ABC):
    """One local TTS model. Subclasses load lazily and synthesize."""

    id: str = ""
    label: str = ""
    note: str = ""
    sample_rate: int = 24000

    def __init__(self) -> None:
        self._loaded = False

    # -- introspection -------------------------------------------------
    def unavailable_reason(self) -> str:
        """Empty string when the engine can run, otherwise why it cannot."""
        return ""

    @abstractmethod
    def voices(self) -> list[Voice]:
        ...

    def default_voice(self) -> str:
        listed = self.voices()
        return listed[0].id if listed else ""

    def status(self) -> EngineStatus:
        reason = self.unavailable_reason()
        return EngineStatus(
            id=self.id,
            label=self.label,
            available=not reason,
            reason=reason,
            note=self.note,
            default_voice=self.default_voice(),
            voices=self.voices() if not reason else [],
            sample_rate=self.sample_rate,
        )

    # -- synthesis -----------------------------------------------------
    @abstractmethod
    def _synthesize(self, text: str, voice: str, on_progress) -> np.ndarray:
        """Return float32 mono samples at :attr:`sample_rate`."""

    def synth(self, text: str, voice: str = "", on_progress=None) -> np.ndarray:
        """Synthesize `text`, reporting progress as a fraction of the characters.

        The lock lives here rather than in each engine so a backend cannot
        forget it, and so the priority drop covers the whole job.
        """
        if not text or not text.strip():
            raise ValueError("empty text")
        reason = self.unavailable_reason()
        if reason:
            raise EngineUnavailable(reason)
        voice = voice or self.default_voice()
        if on_progress:
            on_progress(0.0)
        started = __import__("time").time()
        with generation_lock(), low_priority():
            samples = self._synthesize(text, voice, on_progress)
        if not isinstance(samples, np.ndarray) or samples.size == 0:
            raise RuntimeError(f"{self.id} produced no audio")
        elapsed = __import__("time").time() - started
        duration = len(samples) / self.sample_rate
        print(
            f"[TTS:{self.id}] Done -> {len(samples)} samples "
            f"({duration:.2f}s) in {elapsed:.1f}s "
            f"(RTF {elapsed / max(duration, 1e-6):.2f})",
            flush=True,
        )
        if on_progress:
            on_progress(100.0)
        return samples