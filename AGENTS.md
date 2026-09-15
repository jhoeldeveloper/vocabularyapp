# Agent Guide: Vocabulary App

## Overview
A FastAPI-based vocabulary builder that uses AI (Gemini/Groq) to generate definitions, sentences, and synonyms. Features a real-time editable grid frontend.

## Architecture & Stack
- **Backend:** FastAPI (Python 3.12 — required because Kokoro/torch don't yet support 3.13+)
- **Frontend:** Single-page application (vanilla JS/HTML/CSS) in `static/words.html`.
- **Database:** SQLite (`dictionary.db`).
- **AI Integration:** `gemini_agent.py` (default), `groq_agent.py`, and `openrouter_agent.py`. OpenRouter is also used for story generation (so a single `OPENROUTER_API_KEY` covers both flows).
- **TTS:** `kokoro-onnx` (Kokoro-82M running on ONNX Runtime) via `tts_engine.py`; replaces the old `edge-tts` cloud TTS and the heavier CUDA-enabled PyTorch `kokoro` package (which was maxing out the CPU). Weights are the **fp32** `model.onnx` export by default (on the i5-6200U's Skylake cores, int8 quantization is actually slower than fp32, so fp32 wins on both speed and quality); `q8f16` is kept available as a low-RAM option via `KOKORO_MODEL`. Model files are fetched by `download_models.sh` into `models/onnx/` (`model.onnx` + `voices-v1.0.bin`) and are gitignored.

> **Python version note:** This project targets Python 3.12 (`uv` fetches it automatically — no system Python 3.12 needed). kokoro-onnx supports Python 3.10–3.13, so 3.12 is just a safe, well-tested pin. The old PyTorch `kokoro` build broke on 3.13/3.14 (spacy/`blis`), but the ONNX Runtime stack avoids that.

## Developer Commands
- **Start App:** `./setup_n_run.sh` (uses `uv`: creates a Python 3.12 venv and installs deps) or `uv run uvicorn main:app --reload`.
- **Dependencies:** `uv pip install -r requirements.txt` (with the venv active).
- **Python version:** Pinned via `.python-version` (3.12); `uv` fetches it automatically — no system Python 3.12 needed.
- **Environment:** Requires `.env` with `GEMINI_API_KEY` (or `GROQ_API_KEY` / `OPENROUTER_API_KEY`, depending on `AI_PROVIDER`).
- **OpenRouter models:** both meanings and stories default to `openrouter/free` (OpenRouter's free-tier router; provider from `AI_PROVIDER` backend, model pickers persisted in DB via `GET/POST /api/config`; 429/5xx/404-unavailable errors are retried once with `openrouter/free` and logged to `stderr`). The meanings call is a single batched prompt that returns `{meaning, sentences, synonyms}` as JSON.
- **Migrations:** Handled automatically in `main.py:init_db()`. It detects column changes and rebuilds the table if necessary.

## Key Conventions & Quirks
- **AI Selection:** Toggle between `gemini` (default), `groq`, and `openrouter` via `AI_PROVIDER` env var (provider from backend). When on `openrouter`, the meanings model is picked from a small icon button (`<i class="fa-solid fa-sliders"></i>`) next to **Add** that opens a modal; the story model uses the autocomplete next to **Generate Story**. Both model choices are persisted in DB via `GET/POST /api/config`; fallback `openrouter/free`.
- **Frontend Editing:** Word details are rendered Markdown. **Double-click** a cell to edit the raw Markdown; it saves on blur or Enter.
- **Real-time:** Uses WebSockets (`/ws`) to broadcast updates. If one client adds/edits a word, others update automatically.
- **AI Flow:** Adding a word triggers three parallel AI calls (`gather`) for meaning, sentences, and synonyms.
- **TTS Voice:** Configured via `TTS_VOICE` env var (default: `af_heart`). Uses Kokoro voice codes (e.g. `af_heart`, `am_michael`, `bf_emma`) — see `tts_engine.py`.
- **TTS Engine:** `kokoro-onnx` runs the `q8f16` ONNX export on ONNX Runtime (`CPUExecutionProvider`). ONNX Runtime threads are pinned to **2** (`KOKORO_INTRA_OP_THREADS`, with `KOKORO_INTER_OP_THREADS=1` and `ORT_SEQUENTIAL` execution mode) — the i5-6200U's *physical* core count, not its 4 logical threads, to avoid hyperthreading contention. Override the execution provider with `ONNX_PROVIDER` and the model/voices files with `KOKORO_MODEL` / `KOKORO_VOICES` (e.g. set `KOKORO_MODEL` to `model.onnx` fp32 for a quality A/B test — do **not** use the `int8` variant by default). Phonemization uses `misaki`, which needs the `espeak-ng` system library (installed best-effort by `setup_n_run.sh`).

## Important Files
- `main.py`: The "brain" — contains API routes, DB logic, and WebSocket management.
- `static/words.html`: Contains all frontend logic, styles, and templates.
- `dictionary.db`: SQLite database file (ignored by git but central to local dev).
