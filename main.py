import os
import asyncio
import sqlite3
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

ai_ready = is_ready()
print(f"AI provider: '{AI_PROVIDER}' | ready: {ai_ready}")

# --- FastAPI App Initialization ---
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


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

init_db()


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

@app.post("/api/tts")
async def text_to_speech(text: str = Body(..., embed=True)):
    communicate = edge_tts.Communicate(text, TTS_VOICE)
    audio_chunks = []
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_chunks.append(chunk["data"])
    return Response(b"".join(audio_chunks), media_type="audio/mpeg")



