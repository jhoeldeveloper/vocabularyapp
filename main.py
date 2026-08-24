import os
import asyncio
import sqlite3
import uuid
from typing import List, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Body, WebSocket, WebSocketDisconnect, Query
import edge_tts
from fastapi.responses import HTMLResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
load_dotenv()

# --- Configuration ---
DATABASE_URL = "dictionary.db"

# --- AI Provider Selection ---
AI_PROVIDER = os.getenv("AI_PROVIDER", "gemini").lower()
TTS_VOICE = os.getenv("TTS_VOICE", "en-US-AvaMultilingualNeural")

if AI_PROVIDER == "groq":
    from groq_agent import sync_get_meanings_of, sync_get_sentences_with, sync_get_synonyms_of, is_ready
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
    for col, ddl in (("model", "TEXT"), ("duration_ms", "INTEGER"), ("cost", "REAL")):
        if col not in _story_cols:
            _cur.execute(f"ALTER TABLE stories ADD COLUMN {col} {ddl}")
            print(f"MIGRATION: Added stories.{col} column.")

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
        # No loop stored here — grab it lazily when needed

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
        # Grab the running loop at call time, not at init time
        loop = asyncio.get_running_loop()
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
def api_get_words(sort_by: str = Query('updatedAt', enum=['id', 'encounters', 'createdAt', 'updatedAt'])):
    valid_sort_keys = {"id", "encounters", "createdAt", "updatedAt"}
    if sort_by not in valid_sort_keys:
        raise HTTPException(status_code=400, detail="Invalid sort key.")

    if sort_by == 'updatedAt':
        order_clause = "ORDER BY COALESCE(updatedAt, createdAt) DESC, id DESC"
    elif sort_by == 'createdAt':
        order_clause = "ORDER BY createdAt DESC, id DESC"
    elif sort_by == 'encounters':
        order_clause = "ORDER BY encounters DESC, id DESC"
    else:
        order_clause = "ORDER BY id DESC"

    query = f"SELECT id, word, meaning, sentences, synonyms, encounters, createdAt, updatedAt FROM dictionary {order_clause}"
    conn = sqlite3.connect(DATABASE_URL)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


@app.post("/api/word")
async def add_word(word: str = Body(..., embed=True)):
    if not ai_ready:
         raise HTTPException(status_code=503, detail="AI service is not available.")

    word = word.lower()
    meanings, sentences, synonyms = await asyncio.gather(
        run_in_threadpool(sync_get_meanings_of, word),
        run_in_threadpool(sync_get_sentences_with, word),
        run_in_threadpool(sync_get_synonyms_of, word)
    )


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

    # Resolve model: explicit request -> persisted last free -> env default.
    persisted = get_setting("openrouter_last_model")
    actual_model = data.model or persisted or os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")

    result = await run_in_threadpool(openrouter_agent.sync_generate_story, words, data.title, actual_model)
    if not result.get("ok"):
        raise HTTPException(status_code=502, detail=result.get("error", "Story generation failed"))
    content = result["content"]
    duration_ms = int(result.get("elapsed", 0.0) * 1000)
    cost = result.get("cost", 0.0)

    title = _extract_title(content, data.title)

    def db_operation():
        conn = sqlite3.connect(DATABASE_URL)
        cursor = conn.cursor()
        placeholders = ",".join("?" * len(words))
        cursor.execute(f"SELECT id, word FROM dictionary WHERE word IN ({placeholders})", words)
        word_map = {row[0]: row[1] for row in cursor.fetchall()}  # word -> id
        # Map back to id by word
        id_by_word = {w: wid for wid, w in word_map.items()}
        cursor.execute(
            "INSERT INTO stories (title, content, audio_path, model, duration_ms, cost) VALUES (?, ?, NULL, ?, ?, ?)",
            (title, content, actual_model, duration_ms, cost),
        )
        story_id = cursor.lastrowid
        for w in words:
            wid = id_by_word.get(w)
            if wid:
                cursor.execute(
                    "INSERT OR IGNORE INTO story_words (story_id, word_id) VALUES (?, ?)",
                    (story_id, wid),
                )
        conn.commit()
        conn.close()
        return story_id

    story_id = await run_in_threadpool(db_operation)

    # Persist the last-used model (free or paid) for default resolution.
    if actual_model:
        set_setting("openrouter_last_model", actual_model)

    manager.broadcast_from_sync("update_stories")
    return {
        "id": story_id,
        "title": title,
        "content": content,
        "audio_path": None,
        "model": actual_model,
        "duration_ms": duration_ms,
        "cost": cost,
    }


@app.get("/api/story_counts")
def api_story_counts():
    return get_story_counts()


@app.get("/api/stories")
def api_list_stories(word_id: Optional[int] = Query(None)):
    conn = sqlite3.connect(DATABASE_URL)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    if word_id is not None:
        cursor.execute(
            "SELECT s.id, s.title, s.content, s.audio_path, s.model, s.duration_ms, s.cost, s.createdAt, s.updatedAt "
            "FROM stories s JOIN story_words sw ON sw.story_id = s.id "
            "WHERE sw.word_id = ? ORDER BY s.createdAt DESC, s.id DESC",
            (word_id,),
        )
    else:
        cursor.execute(
            "SELECT id, title, content, audio_path, model, duration_ms, cost, createdAt, updatedAt "
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
        "SELECT id, title, content, audio_path, model, duration_ms, cost, createdAt, updatedAt "
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
    communicate = edge_tts.Communicate(text, TTS_VOICE)
    audio_chunks = []
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_chunks.append(chunk["data"])
    return Response(b"".join(audio_chunks), media_type="audio/mpeg")


@app.post("/api/tts/save")
async def text_to_speech_save(
    text: str = Body(..., embed=True),
    story_id: Optional[int] = Body(None, embed=True),
):
    """Generate audio with edge-tts, persist it to disk, and return the filename."""
    communicate = edge_tts.Communicate(text, TTS_VOICE)
    audio_chunks = []
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_chunks.append(chunk["data"])
    audio_bytes = b"".join(audio_chunks)

    os.makedirs("audio", exist_ok=True)
    filename = f"{uuid.uuid4().hex}.mp3"
    with open(os.path.join("audio", filename), "wb") as f:
        f.write(audio_bytes)

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



