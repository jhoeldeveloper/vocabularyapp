import os
import sys
import asyncio
import threading
import sqlite3
import uuid
from typing import List, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Body, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import HTMLResponse, FileResponse, Response
from tts_engine import synth_wav
from fastapi.staticfiles import StaticFiles
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
load_dotenv()

# --- Configuration ---
DATABASE_URL = "dictionary.db"

# --- AI Provider Selection ---
AI_PROVIDER = os.getenv("AI_PROVIDER", "gemini").lower()
TTS_VOICE = os.getenv("TTS_VOICE", "af_heart")

if AI_PROVIDER == "groq":
    from groq_agent import sync_get_meanings_of, sync_get_sentences_with, sync_get_synonyms_of, is_ready
elif AI_PROVIDER == "openrouter":
    from openrouter_agent import (
        sync_get_meanings_of, sync_get_sentences_with, sync_get_synonyms_of, is_ready,
    )
else:
    from gemini_agent import sync_get_meanings_of, sync_get_sentences_with, sync_get_synonyms_of, is_ready

import openrouter_agent

ai_ready = is_ready()
print(f"AI provider: '{AI_PROVIDER}' | ready: {ai_ready}")
print(f"OpenRouter ready: {openrouter_agent.is_ready()}")

# --- FastAPI App Initialization ---
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Audio files generated for stories (read-aloud) are persisted here.
os.makedirs("audio", exist_ok=True)
app.mount("/audio", StaticFiles(directory="audio"), name="audio")


# --- TTS model warm-up (background) ---
# Loads the Kokoro pipeline + voice and downloads the model once, in a daemon
# thread, so the first real TTS request isn't hit with the one-time cost.
@app.on_event("startup")
def warm_up_tts():
    # Capture the running loop so broadcast_from_sync works from worker threads.
    try:
        manager._loop = asyncio.get_running_loop()
    except RuntimeError:
        pass

    def _warm():
        try:
            synth_wav("warm up", TTS_VOICE)
            print("[TTS] warm-up complete", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"[TTS] warm-up skipped: {e}", file=sys.stderr, flush=True)

    threading.Thread(target=_warm, daemon=True).start()



# --- Database Setup (FIXED) ---
def init_db():
    """
    Initializes the database. Creates the table if it doesn't exist,
    and performs a safe migration if the schema is outdated.
    """
    conn = sqlite3.connect(DATABASE_URL)
    cursor = conn.cursor()

    # Always create the table if it doesn't exist, with the ideal schema.
    # This works for brand new databases.
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS dictionary (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        word TEXT NOT NULL UNIQUE,
        meaning TEXT,
        sentences TEXT,
        synonyms TEXT,
        encounters INTEGER DEFAULT 1,
        createdAt TEXT DEFAULT CURRENT_TIMESTAMP,
        updatedAt TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # Check which columns exist to determine if migration is needed.
    cursor.execute("PRAGMA table_info(dictionary)")
    columns = [col[1] for col in cursor.fetchall()]

    needs_migration = (
        'frequency' in columns or   # old column name
        'createdAt' not in columns or
        'updatedAt' not in columns or
        'synonyms' not in columns    # new column added
    )

    if needs_migration:
        print("MIGRATION: Schema outdated. Rebuilding table...")
        # Determine the correct source column for encounters
        freq_col = 'frequency' if 'frequency' in columns else 'encounters'
        created_col = 'createdAt' if 'createdAt' in columns else None
        try:
            cursor.execute("BEGIN TRANSACTION;")

            # 1. Rename the old table.
            cursor.execute("ALTER TABLE dictionary RENAME TO dictionary_old;")

            # 2. Create the new table with the correct, final schema.
            cursor.execute("""
            CREATE TABLE dictionary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                word TEXT NOT NULL UNIQUE,
                meaning TEXT,
                sentences TEXT,
                synonyms TEXT,
                encounters INTEGER DEFAULT 1,
                createdAt TEXT DEFAULT CURRENT_TIMESTAMP,
                updatedAt TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """)

            # 3. Copy the data from the old table to the new one.
            if created_col:
                cursor.execute(f"""
                INSERT INTO dictionary (id, word, meaning, sentences, synonyms, encounters, createdAt, updatedAt)
                SELECT id, word, meaning, sentences, NULL, {freq_col}, createdAt, createdAt FROM dictionary_old;
                """)
            else:
                cursor.execute(f"""
                INSERT INTO dictionary (id, word, meaning, sentences, synonyms, encounters)
                SELECT id, word, meaning, sentences, NULL, {freq_col} FROM dictionary_old;
                """)

            # 4. Drop the old, temporary table.
            cursor.execute("DROP TABLE dictionary_old;")

            # 5. Commit all changes.
            conn.commit()
            print("MIGRATION SUCCESSFUL: The 'dictionary' table has been updated.")

        except Exception as e:
            print(f"MIGRATION FAILED: {e}. Rolling back changes.")
            conn.rollback()
        finally:
            conn.close()
    else:
        # If the schema is already up to date, no migration is needed.
        conn.close()

    # --- Feature tables: stories, story_words, settings ---
    _conn = sqlite3.connect(DATABASE_URL)
    _cur = _conn.cursor()
    _cur.execute("""
    CREATE TABLE IF NOT EXISTS stories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT,
        content TEXT,
        audio_path TEXT,
        model TEXT,
        duration_ms INTEGER,
        cost REAL,
        status TEXT DEFAULT 'ready',
        error TEXT,
        createdAt TEXT DEFAULT CURRENT_TIMESTAMP,
        updatedAt TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """)
    _cur.execute("""
    CREATE TABLE IF NOT EXISTS story_words (
        story_id INTEGER NOT NULL,
        word_id INTEGER NOT NULL,
        PRIMARY KEY (story_id, word_id),
        FOREIGN KEY (story_id) REFERENCES stories(id) ON DELETE CASCADE,
        FOREIGN KEY (word_id) REFERENCES dictionary(id) ON DELETE CASCADE
    )
    """)
    _cur.execute("""
    CREATE TABLE IF NOT EXISTS settings (
        key TEXT PRIMARY KEY,
        value TEXT
    )
    """)

    # Migrate: add story metadata columns if they don't exist yet.
    _cur.execute("PRAGMA table_info(stories)")
    _story_cols = [col[1] for col in _cur.fetchall()]
    for col, ddl in (("model", "TEXT"), ("duration_ms", "INTEGER"), ("cost", "REAL"), ("status", "TEXT DEFAULT 'ready'"), ("error", "TEXT")):
        if col not in _story_cols:
            _cur.execute(f"ALTER TABLE stories ADD COLUMN {col} {ddl}")
            print(f"MIGRATION: Added stories.{col} column.")

    # Stories left in 'generating' belong to jobs that died with a previous
    # process (e.g. server restart). Mark them so the UI doesn't spin forever.
    _cur.execute(
        "UPDATE stories SET status = 'failed', error = ? WHERE status = 'generating'",
        ("Interrupted by server restart",),
    )
    if _cur.rowcount:
        print(f"MIGRATION: Marked {_cur.rowcount} stale 'generating' story(ies) as failed.")

    # Cleanup legacy model setting keys (now stored as 'meanings_model' / 'story_model').
    _cur.execute("DELETE FROM settings WHERE key IN ('openrouter_last_meanings_model', 'openrouter_last_model')")
    if _cur.rowcount:
        print(f"MIGRATION: Removed {_cur.rowcount} legacy model setting(s).")

    _conn.commit()
    _conn.close()

init_db()


# --- Settings helpers (persist last-used model, etc.) ---
def get_setting(key: str, default=None):
    conn = sqlite3.connect(DATABASE_URL)
    cur = conn.cursor()
    cur.execute("SELECT value FROM settings WHERE key = ?", (key,))
    row = cur.fetchone()
    conn.close()
    return row[0] if row else default


def set_setting(key: str, value):
    conn = sqlite3.connect(DATABASE_URL)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO settings(key, value) VALUES(?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )
    conn.commit()
    conn.close()


def get_story_counts():
    """Return a dict mapping word_id -> number of stories it appears in."""
    conn = sqlite3.connect(DATABASE_URL)
    cur = conn.cursor()
    cur.execute("SELECT word_id, COUNT(*) FROM story_words GROUP BY word_id")
    counts = {row[0]: row[1] for row in cur.fetchall()}
    conn.close()
    return counts


def _extract_title(content: str, fallback=None):
    for line in content.splitlines():
        if line.lower().startswith("title:"):
            return line.split(":", 1)[1].strip()
    return fallback or "Untitled Story"


# --- Pydantic Models ---
class WordEntry(BaseModel):
    id: int
    word: str
    meaning: Optional[str] = ""
    sentences: Optional[str] = ""
    synonyms: Optional[str] = ""
    encounters: int
    createdAt: Optional[str] = None
    updatedAt: Optional[str] = None

# --- WebSocket Connection Manager ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        # Event loop used to schedule broadcasts from worker threads.
        # Captured from an async context; falls back to this if a sync
        # (worker-thread) call happens first.
        self._loop = None

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        tasks = [conn.send_text(message) for conn in self.active_connections]
        await asyncio.gather(*tasks, return_exceptions=True)

    def broadcast_from_sync(self, message: str):
        # Prefer the running loop (async context); otherwise reuse the loop
        # captured from an async call. This keeps the method safe to call
        # from worker threads (e.g. TTS progress callbacks).
        try:
            loop = asyncio.get_running_loop()
            self._loop = loop
        except RuntimeError:
            loop = self._loop
        asyncio.run_coroutine_threadsafe(self.broadcast(message), loop)

manager = ConnectionManager()


# --- WebSocket Endpoint ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)


# --- API Routes ---
@app.get("/", response_class=FileResponse)
async def get_words_html_page():
    file_path = os.path.join("static", "words.html")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="words.html not found")
    return FileResponse(file_path, media_type="text/html")


@app.get("/api/words")
def api_get_words(
    sort_by: str = Query('updatedAt', enum=['id', 'encounters', 'createdAt', 'updatedAt', 'word']),
    order: str = Query('desc', enum=['asc', 'desc']),
):
    valid_sort_keys = {"id", "encounters", "createdAt", "updatedAt", "word"}
    if sort_by not in valid_sort_keys:
        raise HTTPException(status_code=400, detail="Invalid sort key.")
    if order not in ("asc", "desc"):
        order = "desc"
    sql_dir = "ASC" if order == "asc" else "DESC"

    if sort_by == 'updatedAt':
        order_clause = f"ORDER BY COALESCE(updatedAt, createdAt) {sql_dir}, id {sql_dir}"
    elif sort_by == 'createdAt':
        order_clause = f"ORDER BY createdAt {sql_dir}, id {sql_dir}"
    elif sort_by == 'encounters':
        order_clause = f"ORDER BY encounters {sql_dir}, id {sql_dir}"
    elif sort_by == 'word':
        order_clause = f"ORDER BY word COLLATE NOCASE {sql_dir}, id {sql_dir}"
    else:
        order_clause = f"ORDER BY id {sql_dir}"

    query = f"SELECT id, word, meaning, sentences, synonyms, encounters, createdAt, updatedAt FROM dictionary {order_clause}"
    conn = sqlite3.connect(DATABASE_URL)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


@app.post("/api/word")
async def add_word(
    word: str = Body(..., embed=True),
    meanings_model: Optional[str] = Body(None, embed=True),
):
    if not ai_ready:
         raise HTTPException(status_code=503, detail="AI service is not available.")

    word = word.lower()

    # Provider from backend (AI_PROVIDER), model from client or DB fallback.
    # meanings_model may be None (Default) or explicit id from picker.
    if meanings_model is None and AI_PROVIDER == "openrouter":
        meanings_model = get_setting("meanings_model") or "openrouter/free"
    resolved_model = meanings_model if AI_PROVIDER == "openrouter" else None
    lookup_args = (word, resolved_model) if AI_PROVIDER == "openrouter" else (word,)

    MAX_ATTEMPTS = 2
    err = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(f"[add_word] Attempt {attempt}/{MAX_ATTEMPTS} for '{word}'", flush=True)
        meanings, sentences, synonyms = await asyncio.gather(
            run_in_threadpool(sync_get_meanings_of, *lookup_args),
            run_in_threadpool(sync_get_sentences_with, *lookup_args),
            run_in_threadpool(sync_get_synonyms_of, *lookup_args)
        )
        if not any(s.startswith("Error fetching") for s in (meanings, sentences, synonyms)):
            print(f"[add_word] Success on attempt {attempt} for '{word}'", flush=True)
            break
        err = next(s for s in (meanings, sentences, synonyms) if s.startswith("Error fetching"))
        print(f"[add_word] Attempt {attempt} failed for '{word}': {err}", flush=True)
        if attempt < MAX_ATTEMPTS:
            await asyncio.sleep(5)

    # Fail-closed: do not mutate DB if all attempts failed
    if err and any(s.startswith("Error fetching") for s in (meanings, sentences, synonyms)):
        raise HTTPException(status_code=502, detail=err)

    def db_operation():
        conn = sqlite3.connect(DATABASE_URL)
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT id, encounters FROM dictionary WHERE word = ?", (word,))
            result = cursor.fetchone()
            if result:
                new_count = result[1] + 1
                cursor.execute(
                    "UPDATE dictionary SET encounters = ?, meaning = ?, sentences = ?, synonyms = ?, updatedAt = CURRENT_TIMESTAMP WHERE word = ?",
                    (new_count, meanings, sentences, synonyms, word)
                )
                message = f"'{word}' updated (count: {new_count})"
            else:
                # The createdAt and updatedAt columns will be filled by the DB's DEFAULT rule.
                cursor.execute(
                    "INSERT INTO dictionary (word, meaning, sentences, synonyms, encounters) VALUES (?, ?, ?, ?, ?)",
                    (word, meanings, sentences, synonyms, 1)
                )
                message = f"'{word}' registered (count: 1)"
            conn.commit()
            return True, message
        except Exception as e:
            if conn: conn.rollback()
            return False, f"Database error: {e}"
        finally:
            if conn: conn.close()

    success, response_message = await run_in_threadpool(db_operation)
    print(response_message)
    if success:
        manager.broadcast_from_sync(f"update_words:{response_message}")
    return {"message": response_message}


@app.put("/api/words")
async def update_word(data: WordEntry):
    def db_operation():
        conn = sqlite3.connect(DATABASE_URL)
        cursor = conn.cursor()
        try:
            cursor.execute(
                "UPDATE dictionary SET word = ?, meaning = ?, sentences = ?, synonyms = ?, encounters = ?, updatedAt = CURRENT_TIMESTAMP WHERE id = ?",
                (data.word, data.meaning, data.sentences, data.synonyms, data.encounters, data.id)
            )
            if cursor.rowcount == 0:
                return False, "Word not found"
            conn.commit()
            return True, "Word updated successfully"
        except Exception as e:
            conn.rollback()
            return False, str(e)
        finally:
            conn.close()

    success, message = await run_in_threadpool(db_operation)
    if not success:
        raise HTTPException(status_code=404 if message == "Word not found" else 500, detail=message)
    await manager.broadcast("update_words")
    return {"message": message}


@app.delete("/api/words/{word_id}")
async def delete_word(word_id: int):
    def db_operation():
        conn = sqlite3.connect(DATABASE_URL)
        cursor = conn.cursor()
        try:
            cursor.execute("DELETE FROM story_words WHERE word_id = ?", (word_id,))
            cursor.execute("DELETE FROM dictionary WHERE id = ?", (word_id,))
            if cursor.rowcount == 0:
                return False, "Word not found"
            conn.commit()
            return True, "Word deleted successfully"
        except Exception as e:
            conn.rollback()
            return False, str(e)
        finally:
            conn.close()

    success, message = await run_in_threadpool(db_operation)
    if not success:
        raise HTTPException(status_code=404 if message == "Word not found" else 500, detail=message)
    await manager.broadcast("update_words")
    return {"message": message}


# --- Advanced word filter (for story selection) ---
VALID_FILTER_FIELDS = {"id", "encounters", "createdAt", "updatedAt"}

@app.get("/api/words/filter")
def api_filter_words(
    field: str = Query("id"),
    direction: str = Query("desc"),
    min_val: str = Query(None, alias="min"),
    max_val: str = Query(None, alias="max"),
):
    if field not in VALID_FILTER_FIELDS:
        raise HTTPException(status_code=400, detail="Invalid filter field.")
    if direction not in ("asc", "desc"):
        direction = "desc"
    sql_direction = "ASC" if direction == "asc" else "DESC"

    # min/max define a 1-based result-index range within the sorted list
    # (e.g. from=1, to=100 shows the first 100 results; 101-200 the next slice).
    offset = 0
    limit = 200
    try:
        if min_val not in (None, ""):
            start = max(1, int(min_val))
            offset = start - 1
            if max_val not in (None, ""):
                end = max(start, int(max_val))
                limit = end - start + 1
        elif max_val not in (None, ""):
            limit = max(1, int(max_val))
    except ValueError:
        raise HTTPException(status_code=400, detail="min/max must be integers (result positions).")

    query = (
        f"SELECT id, word, meaning, sentences, synonyms, encounters, createdAt, updatedAt "
        f"FROM dictionary ORDER BY {field} {sql_direction}, id {sql_direction} LIMIT ? OFFSET ?"
    )
    conn = sqlite3.connect(DATABASE_URL)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(query, (limit, offset))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


# --- Stories (generated from words via OpenRouter) ---
class StoryCreate(BaseModel):
    words: List[str]
    title: Optional[str] = None
    model: Optional[str] = None
    provider: Optional[str] = None


class StoryUpdate(BaseModel):
    content: str
    title: Optional[str] = None


@app.post("/api/story")
async def create_story(data: StoryCreate):
    if not openrouter_agent.is_ready():
        raise HTTPException(status_code=503, detail="OpenRouter service is not available.")

    words = [w.lower().strip() for w in data.words if w.strip()]
    if not words:
        raise HTTPException(status_code=400, detail="No words provided.")

    # Model from localStorage (client), fallback hardcoded openrouter/free (same as meanings).
    actual_model = data.model or "openrouter/free"
    actual_provider = data.provider or None

    def db_operation():
        conn = sqlite3.connect(DATABASE_URL)
        cursor = conn.cursor()
        placeholders = ",".join("?" * len(words))
        cursor.execute(f"SELECT id FROM dictionary WHERE word IN ({placeholders})", words)
        word_ids = [row[0] for row in cursor.fetchall()]
        cursor.execute(
            "INSERT INTO stories (title, content, audio_path, model, duration_ms, cost, status) "
            "VALUES (?, NULL, NULL, ?, NULL, NULL, 'generating')",
            (data.title or "Generating story...", actual_model),
        )
        story_id = cursor.lastrowid
        for wid in word_ids:
            cursor.execute(
                "INSERT OR IGNORE INTO story_words (story_id, word_id) VALUES (?, ?)",
                (story_id, wid),
            )
        conn.commit()
        conn.close()
        return story_id

    story_id = await run_in_threadpool(db_operation)

    # Heavy AI generation runs as a background task so a page refresh or
    # dropped connection can't lose the result — it is written to the DB
    # and announced over the WebSocket when done.
    asyncio.create_task(generate_story_job(story_id, words, data.title, actual_model, actual_provider))
    manager.broadcast_from_sync("update_stories")
    return {"id": story_id, "status": "generating"}


async def generate_story_job(story_id: int, words: List[str], title_hint: Optional[str], model: str, provider_tag: Optional[str] = None):
    """Generate story content in the background and persist it when done."""
    try:
        result = await run_in_threadpool(openrouter_agent.sync_generate_story, words, title_hint, model, provider_tag, story_id)
        if not result.get("ok"):
            raise RuntimeError(result.get("error", "Story generation failed"))

        content = result["content"]
        duration_ms = int(result.get("elapsed", 0.0) * 1000)
        cost = result.get("cost", 0.0)
        title = _extract_title(content, title_hint)

        def db_operation():
            conn = sqlite3.connect(DATABASE_URL)
            cursor = conn.cursor()
            # Row may have been deleted by the user while generating; the
            # UPDATE is then simply a no-op.
            cursor.execute(
                "UPDATE stories SET title = ?, content = ?, model = ?, duration_ms = ?, cost = ?, "
                "status = 'ready', error = NULL, updatedAt = CURRENT_TIMESTAMP WHERE id = ?",
                (title, content, model, duration_ms, cost, story_id),
            )
            conn.commit()
            conn.close()

        await run_in_threadpool(db_operation)
        manager.broadcast_from_sync(f"story_ready:{story_id}")
    except Exception as e:
        print(f"[STORY] background generation failed for story {story_id}: {e}")

        def fail_operation():
            conn = sqlite3.connect(DATABASE_URL)
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE stories SET status = 'failed', error = ?, updatedAt = CURRENT_TIMESTAMP WHERE id = ?",
                (str(e)[:500], story_id),
            )
            conn.commit()
            conn.close()

        await run_in_threadpool(fail_operation)
        manager.broadcast_from_sync(f"story_error:{story_id}")


@app.get("/api/story_counts")
def api_story_counts():
    return get_story_counts()


@app.get("/api/openrouter/models")
def api_openrouter_models():
    """Full OpenRouter model catalogue (server-side cached ~10 min)."""
    result = openrouter_agent.sync_list_models()
    if not result.get("ok"):
        raise HTTPException(status_code=502, detail=result.get("error", "Failed to fetch models"))
    return {"models": result["models"]}


@app.get("/api/openrouter/providers")
def api_openrouter_providers(model: str = Query(...)):
    """Per-provider endpoint data for a given model (authenticated, cached ~5 min)."""
    result = openrouter_agent.sync_list_providers(model)
    if not result.get("ok"):
        raise HTTPException(status_code=502, detail=result.get("error", "Failed to fetch providers"))
    return {"providers": result["providers"]}


@app.get("/api/config")
def api_config():
    """Small public config the frontend needs to render provider-specific UI."""
    return {
        "ai_provider": AI_PROVIDER,
        "openrouter_ready": openrouter_agent.is_ready(),
        "meanings_model": get_setting("meanings_model") or "openrouter/free",
        "story_model": get_setting("story_model") or "openrouter/free",
        "story_provider": get_setting("story_provider") or "",
    }


@app.post("/api/config")
def update_config(
    meanings_model: Optional[str] = Body(None),
    story_model: Optional[str] = Body(None),
    story_provider: Optional[str] = Body(None),
):
    """Save model selections to DB (single source of truth for all clients)."""
    if meanings_model is not None:
        set_setting("meanings_model", meanings_model)
    if story_model is not None:
        set_setting("story_model", story_model)
    if story_provider is not None:
        set_setting("story_provider", story_provider)
    return {"ok": True}


@app.get("/api/stories")
def api_list_stories(word_id: Optional[int] = Query(None)):
    conn = sqlite3.connect(DATABASE_URL)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    if word_id is not None:
        cursor.execute(
            "SELECT s.id, s.title, s.content, s.audio_path, s.model, s.duration_ms, s.cost, s.status, s.error, s.createdAt, s.updatedAt "
            "FROM stories s JOIN story_words sw ON sw.story_id = s.id "
            "WHERE sw.word_id = ? ORDER BY s.createdAt DESC, s.id DESC",
            (word_id,),
        )
    else:
        cursor.execute(
            "SELECT id, title, content, audio_path, model, duration_ms, cost, status, error, createdAt, updatedAt "
            "FROM stories ORDER BY createdAt DESC, id DESC"
        )
    stories = [dict(row) for row in cursor.fetchall()]

    for s in stories:
        cursor.execute(
            "SELECT d.id, d.word FROM story_words sw "
            "JOIN dictionary d ON d.id = sw.word_id WHERE sw.story_id = ?",
            (s["id"],),
        )
        s["words"] = [dict(row) for row in cursor.fetchall()]

    conn.close()
    return stories


@app.get("/api/stories/{story_id}")
def api_get_story(story_id: int):
    conn = sqlite3.connect(DATABASE_URL)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, title, content, audio_path, model, duration_ms, cost, status, error, createdAt, updatedAt "
        "FROM stories WHERE id = ?",
        (story_id,),
    )
    row = cursor.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Story not found")
    story = dict(row)
    cursor.execute(
        "SELECT d.id, d.word FROM story_words sw "
        "JOIN dictionary d ON d.id = sw.word_id WHERE sw.story_id = ?",
        (story_id,),
    )
    story["words"] = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return story


@app.post("/api/stories/{story_id}/retry")
async def retry_story(story_id: int):
    """Re-run background generation for a failed story (same words + model)."""
    def fetch_and_reset():
        conn = sqlite3.connect(DATABASE_URL)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT id, title, model, status FROM stories WHERE id = ?", (story_id,))
        story = cursor.fetchone()
        if not story:
            conn.close()
            return None, None, None
        if story["status"] != "failed":
            conn.close()
            return "not_failed", None, None
        cursor.execute(
            "SELECT LOWER(d.word) FROM story_words sw "
            "JOIN dictionary d ON d.id = sw.word_id WHERE sw.story_id = ? ORDER BY d.id",
            (story_id,),
        )
        words = [row[0] for row in cursor.fetchall()]
        if not words:
            conn.close()
            return "no_words", None, None
        cursor.execute(
            "UPDATE stories SET status = 'generating', error = NULL, updatedAt = CURRENT_TIMESTAMP WHERE id = ?",
            (story_id,),
        )
        conn.commit()
        conn.close()
        return "ok", words, story["model"]

    outcome, words, model = await run_in_threadpool(fetch_and_reset)
    if outcome is None:
        raise HTTPException(status_code=404, detail="Story not found")
    if outcome == "not_failed":
        raise HTTPException(status_code=400, detail="Only failed stories can be retried")
    if outcome == "no_words":
        raise HTTPException(status_code=400, detail="Story has no linked words to regenerate from")

    actual_model = model or "openrouter/free"

    # Fresh title: let the AI invent one unless the user supplied a custom one
    # (placeholder titles like 'Generating story...' are treated as no hint).
    title_hint = None
    conn = sqlite3.connect(DATABASE_URL)
    row = conn.execute("SELECT title FROM stories WHERE id = ?", (story_id,)).fetchone()
    conn.close()
    if row and row[0] and row[0] != "Generating story...":
        title_hint = row[0]

    asyncio.create_task(generate_story_job(story_id, words, title_hint, actual_model))
    manager.broadcast_from_sync("update_stories")
    return {"id": story_id, "status": "generating"}


@app.post("/api/stories/{story_id}/cancel")
async def cancel_story(story_id: int):
    """Cancel an in-flight story generation request."""
    # 1. Close the HTTP request if it's in-flight.
    openrouter_agent.cancel_story_request(story_id)
    # 2. Mark DB as failed (only if still generating).
    def fail_operation():
        conn = sqlite3.connect(DATABASE_URL)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE stories SET status = 'failed', error = ?, updatedAt = CURRENT_TIMESTAMP "
            "WHERE id = ? AND status = 'generating'",
            ("Cancelled by user", story_id),
        )
        changed = cursor.rowcount
        conn.commit()
        conn.close()
        return changed
    changed = await run_in_threadpool(fail_operation)
    if changed:
        manager.broadcast_from_sync(f"story_error:{story_id}")
    return {"ok": True}


@app.put("/api/stories/{story_id}")
async def update_story(story_id: int, data: StoryUpdate):
    def db_operation():
        conn = sqlite3.connect(DATABASE_URL)
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM stories WHERE id = ?", (story_id,))
        if not cursor.fetchone():
            conn.close()
            return False, "Story not found"
        title = _extract_title(data.content, data.title)
        # Content changed -> previously saved audio no longer matches, clear it.
        cursor.execute(
            "UPDATE stories SET title = ?, content = ?, audio_path = NULL, updatedAt = CURRENT_TIMESTAMP "
            "WHERE id = ?",
            (title, data.content, story_id),
        )
        # Remove orphaned audio file if it existed.
        conn.commit()
        conn.close()
        return True, "Story updated"
    success, message = await run_in_threadpool(db_operation)
    if not success:
        raise HTTPException(status_code=404, detail=message)
    await manager.broadcast("update_stories")
    return {"message": message}


@app.delete("/api/stories/{story_id}")
async def delete_story(story_id: int):
    def db_operation():
        conn = sqlite3.connect(DATABASE_URL)
        cursor = conn.cursor()
        cursor.execute("SELECT audio_path FROM stories WHERE id = ?", (story_id,))
        row = cursor.fetchone()
        if not row:
            conn.close()
            return False, "Story not found"
        audio_path = row[0]
        cursor.execute("DELETE FROM story_words WHERE story_id = ?", (story_id,))
        cursor.execute("DELETE FROM stories WHERE id = ?", (story_id,))
        conn.commit()
        conn.close()
        if audio_path:
            full = os.path.join("audio", audio_path)
            if os.path.exists(full):
                try:
                    os.remove(full)
                except Exception:
                    pass
        return True, "Story deleted"
    success, message = await run_in_threadpool(db_operation)
    if not success:
        raise HTTPException(status_code=404, detail=message)
    await manager.broadcast("update_stories")
    return {"message": message}


@app.post("/api/tts")
async def text_to_speech(text: str = Body(..., embed=True)):
    wav = await run_in_threadpool(synth_wav, text, TTS_VOICE)
    return Response(wav, media_type="audio/wav")


@app.post("/api/tts/save")
async def text_to_speech_save(
    text: str = Body(..., embed=True),
    story_id: Optional[int] = Body(None, embed=True),
):
    """Generate audio with Kokoro and persist it.

    - With a `story_id`: the (heavy) generation runs in the background. The
      client gets an immediate `{"status": "generating"}` and is notified over
      the WebSocket (`story_audio_progress:<id>:<pct>` / `story_audio_ready:<id>`)
      when done. This prevents a multi-minute blocking HTTP request.
    - Without a `story_id`: generates synchronously and returns the filename.
    """
    if story_id is not None:
        asyncio.create_task(generate_story_audio(story_id, text))
        return {"status": "generating", "story_id": story_id}

    wav = await run_in_threadpool(synth_wav, text, TTS_VOICE)
    os.makedirs("audio", exist_ok=True)
    filename = f"{uuid.uuid4().hex}.wav"
    with open(os.path.join("audio", filename), "wb") as f:
        f.write(wav)
    return {"filename": filename}


async def generate_story_audio(story_id: int, text: str):
    """Background worker: synthesize, save to disk, update DB, notify clients."""
    try:
        def on_progress(pct: float):
            manager.broadcast_from_sync(f"story_audio_progress:{story_id}:{int(pct)}")

        wav = await run_in_threadpool(synth_wav, text, TTS_VOICE, on_progress)

        os.makedirs("audio", exist_ok=True)
        filename = f"{uuid.uuid4().hex}.wav"
        with open(os.path.join("audio", filename), "wb") as f:
            f.write(wav)

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
        manager.broadcast_from_sync(f"story_audio_ready:{story_id}")
        manager.broadcast_from_sync("update_stories")
    except Exception as e:
        print(
            f"[TTS] background generation failed for story {story_id}: {e}",
            file=sys.stderr,
            flush=True,
        )
        manager.broadcast_from_sync(f"story_audio_error:{story_id}")



