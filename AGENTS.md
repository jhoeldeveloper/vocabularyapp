# Agent Guide: Vocabulary App

## Overview
A FastAPI-based vocabulary builder that uses AI (Gemini/Groq) to generate definitions, sentences, and synonyms. Features a real-time editable grid frontend.

## Architecture & Stack
- **Backend:** FastAPI (Python 3.14+)
- **Frontend:** Single-page application (vanilla JS/HTML/CSS) in `static/words.html`.
- **Database:** SQLite (`dictionary.db`).
- **AI Integration:** `gemini_agent.py` (default) and `groq_agent.py`.
- **TTS:** `edge-tts` for word and sentence pronunciation.

## Developer Commands
- **Start App:** `./setup_n_run.sh` (handles venv and deps) or `uvicorn main:app --reload`.
- **Dependencies:** `pip install -r requirements.txt`.
- **Environment:** Requires `.env` with `GEMINI_API_KEY` (or `GROQ_API_KEY`).
- **Migrations:** Handled automatically in `main.py:init_db()`. It detects column changes and rebuilds the table if necessary.

## Key Conventions & Quirks
- **AI Selection:** Toggle between `gemini` and `groq` via `AI_PROVIDER` env var.
- **Frontend Editing:** Word details are rendered Markdown. **Double-click** a cell to edit the raw Markdown; it saves on blur or Enter.
- **Real-time:** Uses WebSockets (`/ws`) to broadcast updates. If one client adds/edits a word, others update automatically.
- **AI Flow:** Adding a word triggers three parallel AI calls (`gather`) for meaning, sentences, and synonyms.
- **TTS Voice:** Configured via `TTS_VOICE` env var (default: `en-US-AvaMultilingualNeural`).

## Important Files
- `main.py`: The "brain" — contains API routes, DB logic, and WebSocket management.
- `static/words.html`: Contains all frontend logic, styles, and templates.
- `dictionary.db`: SQLite database file (ignored by git but central to local dev).
