# Plan: Replace edge-tts with Kokoro 82M (WAV output)

Status: Approved by user. Ready to implement.

## Decisions (confirmed)
- **Full replacement** of edge-tts (no TTS_PROVIDER toggle).
- Output **WAV** (24 kHz), no ffmpeg dependency.
- Voice read from `TTS_VOICE` env, default changed to a Kokoro code (`af_heart`).

## Files to change

### 1. `requirements.txt`
Remove `edge-tts`; add `kokoro` and `soundfile`. Result:
```
fastapi
pydantic
pydantic-settings
python-dotenv
google-generativeai
uvicorn[standard]
kokoro
soundfile
groq
requests
```
Note: `kokoro` pulls `torch`/`onnxruntime`. First run downloads model weights (~300 MB) from HuggingFace.

### 2. NEW FILE `tts_engine.py`
Encapsulates Kokoro (mirrors `gemini_agent.py`/`groq_agent.py` style).
```python
"""Local TTS via Kokoro-82M. Replaces edge-tts."""
import io
import threading
import numpy as np
import soundfile as sf
from kokoro import KPipeline

# Cache one KPipeline per language code (voice[0]: a=American, b=British, ...)
_PIPELINES: dict[str, "KPipeline"] = {}
_PIPE_LOCK = threading.Lock()
_GEN_LOCK = threading.Lock()  # Kokoro generation is not thread-safe -> serialize

SAMPLE_RATE = 24000


def _pipeline_for(voice: str) -> "KPipeline":
    lang_code = (voice or "af_heart")[0]
    if lang_code not in ("a", "b", "j", "z", "f", "i", "p", "s", "h", "k", "n", "r"):
        lang_code = "a"
    with _PIPE_LOCK:
        if lang_code not in _PIPELINES:
            _PIPELINES[lang_code] = KPipeline(lang_code=lang_code)
        return _PIPELINES[lang_code]


def synth_wav(text: str, voice: str) -> bytes:
    """Synthesize `text` with Kokoro and return WAV file bytes."""
    if not text or not text.strip():
        raise ValueError("empty text")
    pipeline = _pipeline_for(voice)
    segments = []
    with _GEN_LOCK:
        for _gs, _ps, audio in pipeline(text, voice=voice, speed=1.0):
            segments.append(audio)
    if not segments:
        raise RuntimeError("Kokoro produced no audio")
    audio = np.concatenate(segments, axis=0)
    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="WAV")
    return buf.getvalue()
```

### 3. `main.py` edits
- **Line 8**: remove `import edge_tts`; add `import asyncio` (if not already present) and `from tts_engine import synth_wav`.
- **Line 20**: change default:
  ```python
  TTS_VOICE = os.getenv("TTS_VOICE", "af_heart")
  ```
- **`/api/tts` (lines 639-646)**: replace body with:
  ```python
  @app.post("/api/tts")
  async def text_to_speech(text: str = Body(..., embed=True)):
      wav = await run_in_threadpool(synth_wav, text, TTS_VOICE)
      return Response(wav, media_type="audio/wav")
  ```
- **`/api/tts/save` (lines 649-680)**: replace edge-tts block:
  ```python
  @app.post("/api/tts/save")
  async def text_to_speech_save(
      text: str = Body(..., embed=True),
      story_id: Optional[int] = Body(None, embed=True),
  ):
      """Generate audio with Kokoro, persist it to disk, and return the filename."""
      wav = await run_in_threadpool(synth_wav, text, TTS_VOICE)

      os.makedirs("audio", exist_ok=True)
      filename = f"{uuid.uuid4().hex}.wav"
      with open(os.path.join("audio", filename), "wb") as f:
          f.write(wav)

      if story_id is not None:
          def db_operation():
              conn = sqlite3.connect(DATABASE_URL)
              cursor = conn.cursor()
              cursor.execute(
                  "UPDATE stories SET audio_path = ?, updatedAt = CURRENT_TIMESTAMP WHERE id = ?",
                  (filename, story_id),
              )
              conn.commit()
              conn.close()
          await run_in_threadpool(db_operation)
          manager.broadcast_from_sync("update_stories")

      return {"filename": filename}
  ```
- Keep `asyncio.Lock` optional: Kokoro is serialized inside the engine via `_GEN_LOCK`, so no extra lock needed in the route. `run_in_threadpool` already keeps the event loop free.

### 4. Frontend `static/words.html`
**No code changes required.** `playTTS` (line 880) consumes the response blob and `new Audio(url)` plays WAV natively; `ttsCache` is format-agnostic. Saved story audio is served via `/audio/<filename>` and plays WAV. The `.wav` extension is stored in DB, so `delete_story` cleanup (line 624) works unchanged.
Optional: update stray "mp3" comments for accuracy (non-essential).

## Verification
1. `pip install -r requirements.txt` (downloads torch/onnxruntime + model on first use).
2. Start app: `./setup_n_run.sh` or `uvicorn main:app --reload`.
3. Click a speaker button -> word/meaning/example audio plays (first call slow while model loads ~1-2s).
4. Generate a story; confirm a `.wav` is written to `audio/` and plays back.
5. Set `TTS_VOICE=am_michael` (or `bf_emma`) -> voice changes.

## Risks / notes
- **Cold-start latency**: first `/api/tts` loads the model. If snappier first use is desired, warm the pipeline in a FastAPI `startup` event (e.g. call `_pipeline_for(TTS_VOICE)` once). Out of scope unless requested.
- **Disk size**: WAV larger than MP3 for saved stories (acceptable per WAV choice).
- Long story text is segmented by Kokoro and concatenated in `synth_wav`.
