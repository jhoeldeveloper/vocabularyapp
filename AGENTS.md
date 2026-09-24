# Agent Guide: Vocabulary App

FastAPI vocabulary builder: AI definitions/sentences/synonyms/stories, SQLite, real-time editable grid SPA, local Kokoro TTS.

## Setup & commands
- **Python 3.12 only** (`.python-version`; kokoro-onnx/misaki pin `<3.13`). System Python here may be 3.14 — do not use it.
- **Run:** `./setup_n_run.sh` — installs uv + best-effort espeak-ng, downloads Kokoro models, builds **`venv/`** (3.12), starts uvicorn on `:8000`.
- **Manual:** `uv run uvicorn main:app --reload`. Gotcha: the script uses `venv/`, but bare `uv run` prefers/creates `.venv/` — both exist in this checkout; activate explicitly if paths matter.
- **Deps:** `uv pip install -r requirements.txt` (venv active).
- **Models:** `./download_models.sh` → `models/onnx/` (gitignored). `MODEL_VARIANT=q8f16|fp16|fp32` (default **fp32** — faster and higher quality on the target CPU; int8 is slower without VNNI).
- **Env (`.env`, gitignored):** `AI_PROVIDER` = `gemini` (code default) | `groq` | `openrouter`, plus the matching `*_API_KEY`. Also `TTS_VOICE`; optional `KOKORO_MODEL` / `KOKORO_VOICES`.
- **No tests, linter, typechecker, CI, or pre-commit are configured** — do not hunt for them.

## Architecture (easy to get wrong)
- `AI_PROVIDER` is read once at import in `main.py` — changing it requires a restart.
- **Stories always go through OpenRouter** (`import openrouter_agent` is unconditional), even when `AI_PROVIDER` is gemini/groq. `/api/story*`, `/api/openrouter/*`, and story generation need `OPENROUTER_API_KEY`.
- Adding a word: 3 parallel lookups via `asyncio.gather` + threadpool. OpenRouter dedupes concurrent identical lookups and uses a **single batched prompt** returning `{meaning, sentences, synonyms}` JSON; retries once with `openrouter/free` on 429/5xx/404. `add_word` itself retries the whole gather twice (5s apart), then **fails closed** (no DB write) if any field still starts with `Error fetching`.
- Migrations run automatically in `main.py:init_db()` at import (dictionary table rebuild + `stories`/`settings` columns). Stories left `generating` from a dead process are marked `failed` on startup.
- Story generation and story TTS run as `asyncio.create_task` background jobs — HTTP returns `status: generating` immediately; completion/failure/progress arrive over WebSocket `/ws` (`update_words`, `update_stories`, `story_ready:*`, `story_error:*`, `story_audio_*`).
- Model picks (`meanings_model`, `story_model`, `story_provider`) live in the SQLite `settings` table via `GET/POST /api/config`, shared across clients. localStorage is only theme, volume, and sort.

## Frontend (`static/words.html` — entire SPA, vanilla JS)
- Word details are raw Markdown in the DB, rendered with `marked` (CDN). Double-click a cell to edit raw Markdown; saves on blur/Enter. HTML paste converts via Turndown.
- OpenRouter-only UI: meanings-model icon button next to **Add**; story-model autocomplete next to **Generate Story**. Pointless when `AI_PROVIDER` is gemini/groq.

## TTS (`tts_engine.py`)
- Kokoro-82M ONNX, CPU-only (`CPUExecutionProvider`). Voice from `TTS_VOICE` (default `af_heart`) — must be a **Kokoro voice code** (`af_heart`, `bf_emma`, …), not Azure/edge-tts names (stale values may linger in `.env`).
- Needs system `espeak-ng` (phonemizer backend; best-effort install in the setup script). ORT threads pinned to 2 / 1 (physical cores of target i5-6200U); override with `KOKORO_INTRA_OP_THREADS` / `KOKORO_INTER_OP_THREADS`.
- Weights: `models/onnx/model.onnx` (fp32 default) + `voices-v1.0.bin`.

## Key files
- `main.py` — API routes, SQLite, WebSocket, background story/TTS jobs, migrations
- `static/words.html` — all frontend logic
- `gemini_agent.py` / `groq_agent.py` / `openrouter_agent.py` — same lookup interface: `sync_get_meanings_of`, `sync_get_sentences_with`, `sync_get_synonyms_of`, `is_ready`. OpenRouter adds `sync_generate_story`, model/provider catalog, cancel.
- `tts_engine.py` — Kokoro ONNX synthesis
- DB tables: `dictionary` (words), `stories`, `story_words`, `settings`
