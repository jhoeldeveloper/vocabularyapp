import os
import sys
import re
import json
import time
import asyncio
import threading
import sqlite3
import uuid
from typing import List, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Body, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import HTMLResponse, FileResponse, Response
from tts import CONTEXTS as TTS_CONTEXTS
from tts import get_engine as get_tts_engine
from tts import selected as tts_selected
from tts import set_selection as set_tts_selection
from tts import status as tts_status
from tts import synth_wav, warm_up as warm_up_tts_selected
from tts.base import EngineUnavailable
from fastapi.staticfiles import StaticFiles
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
load_dotenv()

# --- Configuration ---
DATABASE_URL = "dictionary.db"

# --- AI Provider Selection ---
AI_PROVIDER = os.getenv("AI_PROVIDER", "gemini").lower()
# TTS_ENGINE in the environment is a *server-side* default for deployments that
# want to pin one engine; the picker still overrides it. There is deliberately
# no TTS_VOICE counterpart: a voice is meaningless without its engine's voice
# list (Kokoro `af_heart`, Pocket `alba`, Inflect one voice), so an env value
# could only ever be right for one of them. The settings table is the single
# source of truth for both.

if AI_PROVIDER == "groq":
    from groq_agent import sync_get_meanings_of, sync_get_use_of, sync_get_synonyms_of, is_ready
elif AI_PROVIDER == "openrouter":
    from openrouter_agent import (
        sync_get_meanings_of, sync_get_use_of, sync_get_synonyms_of, is_ready,
    )
else:
    from gemini_agent import sync_get_meanings_of, sync_get_use_of, sync_get_synonyms_of, is_ready

import lexicon

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


# --- TTS engine selection + warm-up ---
# The engine (model) and voice live in the settings table like the model picks,
# so the choice is shared across clients, survives a restart, and is the *only*
# source of truth. Nothing here reads an environment voice: see the note by the
# AI provider block.
#
# There are TWO selections, one per context: "meanings" (the short word audio
# behind the per-cell Listen buttons) and "stories" (a whole story narrated).
# A word wants clarity, a story wants expressiveness, and one global pick forced
# the wrong compromise on one of them. Keys are tts_<context>_engine /
# tts_<context>_voice.
_TTS_LEGACY_ENGINE_KEY = "tts_engine"
_TTS_LEGACY_VOICE_KEY = "tts_voice"


def _tts_engine_key(context: str) -> str:
    return f"tts_{context}_engine"


def _tts_voice_key(context: str) -> str:
    return f"tts_{context}_voice"


def _check_tts_context(context: str) -> str:
    context = context or "stories"
    if context not in TTS_CONTEXTS:
        raise HTTPException(status_code=400, detail=f"Unknown TTS context {context!r}")
    return context


def _restore_tts_selection():
    """Apply both persisted selections. Called once, after the settings helpers."""
    # A selection made before the split lived in one global pair; seed both
    # contexts from it rather than silently resetting the user to Kokoro on
    # upgrade. It is left in place, not deleted: an older build reading the same
    # database still finds what it expects.
    legacy_engine = get_setting(_TTS_LEGACY_ENGINE_KEY) or ""
    legacy_voice = get_setting(_TTS_LEGACY_VOICE_KEY) or ""
    for context in TTS_CONTEXTS:
        engine = get_setting(_tts_engine_key(context)) or legacy_engine
        voice = get_setting(_tts_voice_key(context)) or legacy_voice
        try:
            selection = set_tts_selection(context, engine=engine, voice=voice)
            # Persist what was resolved, so the split is a one-way migration:
            # once the per-context rows exist, deleting the legacy pair (or
            # upgrading from a build that no longer reads it) cannot silently
            # reset both contexts back to Kokoro.
            set_setting(_tts_engine_key(context), selection["engine"])
            set_setting(_tts_voice_key(context), selection["voice"])
        except Exception as error:  # noqa: BLE001 - never block boot on TTS
            print(
                f"[TTS:{context}] stored selection ignored: {error}",
                file=sys.stderr,
                flush=True,
            )


# Loads each context's model once, in a daemon thread, so the first real request
# isn't hit with the one-time cost.
@app.on_event("startup")
def warm_up_tts():
    # Capture the running loop so broadcast_from_sync works from worker threads.
    try:
        manager._loop = asyncio.get_running_loop()
    except RuntimeError:
        pass

    def _warm():
        for context in TTS_CONTEXTS:
            warm_up_tts_selected(context=context, voice=_selected_voice(context))

    threading.Thread(target=_warm, daemon=True).start()


def _voice_belongs(voice: str, engine) -> bool:
    """True when `voice` is a voice of `engine` (empty is always fine).

    The stored voice is global but each engine has its own list, so switching
    from Kokoro (af_heart) to Inflect (one voice, "default") must not hand
    Pocket a Kokoro id. Silently falling back to the engine's own default is the
    right behaviour: the user asked to switch model, not to be told the old
    voice is invalid.
    """
    if not voice:
        return True
    try:
        return voice in {v.id for v in engine.voices()}
    except Exception:  # noqa: BLE001 - a broken voice list must not block TTS
        return False



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
        synonyms TEXT,
        use_region TEXT,
        freq_zipf REAL,
        family TEXT,
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
        'updatedAt' not in columns
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
                use_region TEXT,
                freq_zipf REAL,
                family TEXT,
                encounters INTEGER DEFAULT 1,
                createdAt TEXT DEFAULT CURRENT_TIMESTAMP,
                updatedAt TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """)

            # 3. Copy the data over. `meaning` and `synonyms` survive: only
            #    `sentences` is discarded (its examples now live inside the
            #    numbered meaning), and the new measured columns are left NULL
            #    rather than invented for rows that predate them.
            def _col(name):
                return name if name in columns else "NULL"
            meaning_src, synonyms_src = _col("meaning"), _col("synonyms")
            if created_col:
                cursor.execute(f"""
                INSERT INTO dictionary (id, word, meaning, synonyms, encounters, createdAt, updatedAt)
                SELECT id, word, {meaning_src}, {synonyms_src}, {freq_col}, createdAt, createdAt FROM dictionary_old;
                """)
            else:
                cursor.execute(f"""
                INSERT INTO dictionary (id, word, meaning, synonyms, encounters)
                SELECT id, word, {meaning_src}, {synonyms_src}, {freq_col} FROM dictionary_old;
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

    # --- Additive/destructive column migration for an up-to-date table ---
    # Reached by every existing installation: the table is otherwise current, so
    # the rebuild above is skipped, but it still carries the retired columns and
    # lacks the three new ones.
    _conn = sqlite3.connect(DATABASE_URL)
    _cur = _conn.cursor()
    try:
        _cur.execute("PRAGMA table_info(dictionary)")
        _columns = [c[1] for c in _cur.fetchall()]

        # The three fields the new detail panel renders. Zipf and family are
        # MEASURED (lexicon.py) rather than model-written, which is why they are
        # NULL rather than 0 for every pre-existing row -- and why they are never
        # backfilled: a number we did not measure stays absent.
        for name, decl in (("use_region", "TEXT"),
                           ("freq_zipf", "REAL"),
                           ("family", "TEXT"),
                           ("synonyms", "TEXT"),
                           # A mark the USER sets, never something measured or
                           # inferred, so a plain flag with a default rather than
                           # a nullable column. NOT NULL 0 means "not marked",
                           # which is true of every pre-existing word -- it is
                           # the absence of a mark, not a claim about anyone's
                           # vocabulary. Never backfilled, as above.
                           ("learned", "INTEGER NOT NULL DEFAULT 0")):
            if name not in _columns:
                _cur.execute(f"ALTER TABLE dictionary ADD COLUMN {name} {decl}")
                print(f"MIGRATION: added dictionary.{name}")

        # Only `sentences` is dropped. Its content is superseded -- the examples
        # now live inside the numbered sense they illustrate -- so keeping a dead
        # column full of text nothing renders is worse than removing it.
        # `synonyms` is deliberately KEPT: it is the model's own judgement, asked
        # once when the word is added and stored, not derived per render. DROP
        # COLUMN needs SQLite 3.35+; on anything older the column is left in
        # place and simply ignored, because no query references it.
        for name in ("sentences",):
            if name in _columns:
                try:
                    _cur.execute(f"ALTER TABLE dictionary DROP COLUMN {name}")
                    print(f"MIGRATION: dropped dictionary.{name}")
                except sqlite3.Error as e:
                    print(f"MIGRATION: could not drop dictionary.{name}: {e}")
        _conn.commit()
    except Exception as e:
        print(f"MIGRATION: dictionary column pass failed: {e}")
        _conn.rollback()
    finally:
        _conn.close()

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
    for col, ddl in (
        ("model", "TEXT"),
        ("duration_ms", "INTEGER"),
        ("cost", "REAL"),
        ("status", "TEXT DEFAULT 'ready'"),
        ("error", "TEXT"),
        ("prompt_tokens", "INTEGER"),
        ("completion_tokens", "INTEGER"),
        ("reasoning_tokens", "INTEGER"),
        ("words_used", "INTEGER"),
        ("words_total", "INTEGER"),
        ("prose_words", "INTEGER"),
        ("warnings", "TEXT"),
        ("prompt_used", "TEXT"),
        ("prompt_preset", "TEXT"),
        ("style", "TEXT"),
        # The request config this run was made with, as JSON: temperature,
        # output budget, reasoning level and reasoning cap. These moved OUT of
        # the prompt preset (they depend on the model, the preset does not), so
        # without this column a story row could no longer say what it ran with.
        # Written from the frozen recipe, never recomputed, and NULL on every
        # row that predates it -- never backfilled, like the token columns.
        ("request_config", "TEXT"),
    ):
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


# The TTS selection is restored here, not at import: it reads two settings
# rows, and this is the first point where both the schema and the helpers
# exist. Doing it at import time read settings from a database that had not
# been migrated yet, which raised NameError before init_db() was reached.
_restore_tts_selection()


def get_story_counts():
    """Return a dict mapping word_id -> number of stories it appears in."""
    conn = sqlite3.connect(DATABASE_URL)
    cur = conn.cursor()
    cur.execute("SELECT word_id, COUNT(*) FROM story_words GROUP BY word_id")
    counts = {row[0]: row[1] for row in cur.fetchall()}
    conn.close()
    return counts


def _extract_title(content: str, fallback=None):
    """Read the Title: line the model emitted. The system message requires that
    label, and validate_prompt refuses a system message without it, so the
    fallback is only reached if a provider ignored the contract entirely."""
    for line in content.splitlines():
        if line.lower().startswith("title:"):
            return line.split(":", 1)[1].strip()
    return fallback or "Untitled Story"


# --- Pydantic Models ---
class WordEntry(BaseModel):
    id: int
    word: str
    meaning: Optional[str] = ""
    synonyms: Optional[str] = ""
    use_region: Optional[str] = None
    freq_zipf: Optional[float] = None
    family: Optional[str] = None
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
        # Story ids each connection wants live text for. Story deltas are only
        # sent to subscribers so a client parked on the Words view doesn't
        # receive a token stream for a story it cannot see.
        self.subs: dict = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.subs[websocket] = set()

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        self.subs.pop(websocket, None)

    def subscribe(self, websocket: WebSocket, story_id: int):
        self.subs.setdefault(websocket, set()).add(story_id)

    def unsubscribe(self, websocket: WebSocket, story_id: int):
        ids = self.subs.get(websocket)
        if ids:
            ids.discard(story_id)

    async def broadcast(self, message: str):
        tasks = [conn.send_text(message) for conn in self.active_connections]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def broadcast_story_delta(self, story_id: int, message: str):
        """Send a story_delta/story_title frame to that story's subscribers."""
        targets = [conn for conn, ids in self.subs.items() if story_id in ids]
        if not targets:
            return
        tasks = [conn.send_text(message) for conn in targets]
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
        if loop is None:
            return
        asyncio.run_coroutine_threadsafe(self.broadcast(message), loop)

    def broadcast_story_delta_from_sync(self, story_id: int, message: str):
        """Worker-thread variant of broadcast_story_delta (story text deltas)."""
        try:
            loop = asyncio.get_running_loop()
            self._loop = loop
        except RuntimeError:
            loop = self._loop
        if loop is None:
            return
        asyncio.run_coroutine_threadsafe(
            self.broadcast_story_delta(story_id, message), loop
        )


manager = ConnectionManager()

# Live text of in-flight stories, so clients that subscribe (or reconnect)
# mid-generation can be caught up. Keyed by story_id; cleaned up by
# generate_story_job's finally block.
#
# Each entry carries the token of the job that owns it, so a job that is still
# unwinding (e.g. cancelled just before the user hit Retry) can never clobber
# the buffer of the newer job for the same story.
_story_streams: dict = {}
_story_streams_lock = threading.Lock()
_story_stream_seq = 0

# Live generation tasks, so cancel/delete can unwind the coroutine instead of
# just flagging the DB row.
_story_tasks: dict = {}

# Delta batching: a token stream is chunked far faster than any UI can render,
# so deltas are coalesced before hitting the WebSocket.
_STREAM_FLUSH_SECONDS = 0.12
_STREAM_FLUSH_CHARS = 400


def _new_story_stream_token() -> int:
    global _story_stream_seq
    with _story_streams_lock:
        _story_stream_seq += 1
        return _story_stream_seq


def _story_stream_append(story_id: int, token: int, text: str) -> None:
    with _story_streams_lock:
        entry = _story_streams.get(story_id)
        if not entry or entry["token"] != token:
            entry = {"token": token, "text": "", "title": None}
            _story_streams[story_id] = entry
        entry["text"] += text


def _story_stream_snapshot(story_id: int):
    with _story_streams_lock:
        entry = _story_streams.get(story_id)
        if not entry:
            return "", None
        return entry["text"], entry["title"]


def _story_stream_clear(story_id: int, token: int) -> None:
    """Drop the buffer, but only if it still belongs to the given job."""
    with _story_streams_lock:
        entry = _story_streams.get(story_id)
        if entry and entry["token"] == token:
            _story_streams.pop(story_id, None)


def _story_stream_set_title(story_id: int, token: int, title: str) -> None:
    with _story_streams_lock:
        entry = _story_streams.get(story_id)
        if entry and entry["token"] == token:
            entry["title"] = title


def _make_story_delta_sink(story_id: int, token: int):
    """Build the on_delta callback for one story job.

    Called from the worker thread once per SSE chunk. Accumulates the full text
    (for replay) and flushes coalesced deltas to subscribed WebSocket clients.
    """
    pending = []
    pending_chars = 0
    last_flush = time.monotonic()
    state = {"title_sent": False}

    def flush():
        nonlocal pending, pending_chars, last_flush
        if not pending:
            return
        chunk = "".join(pending)
        pending = []
        pending_chars = 0
        last_flush = time.monotonic()
        # json.dumps keeps newlines and colons out of the single-string frame.
        manager.broadcast_story_delta_from_sync(
            story_id, f"story_delta:{story_id}:{json.dumps(chunk)}"
        )

    def on_delta(text: str):
        nonlocal pending_chars
        if not text:
            return
        _story_stream_append(story_id, token, text)
        pending.append(text)
        pending_chars += len(text)

        full, _ = _story_stream_snapshot(story_id)
        if not state["title_sent"]:
            # The prompt asks for "Title: ..." first. Wait for the newline that
            # ends the line, otherwise we'd publish a half-written title.
            match = re.match(r"^\s*Title:[ \t]*([^\n]+)", full, re.IGNORECASE)
            if match and match.end() < len(full) and full[match.end()] == "\n":
                title = match.group(1).strip()
                if title:
                    _story_stream_set_title(story_id, token, title)
                    state["title_sent"] = True
                    manager.broadcast_story_delta_from_sync(
                        story_id, f"story_title:{story_id}:{json.dumps(title)}"
                    )

        if pending_chars >= _STREAM_FLUSH_CHARS or \
                time.monotonic() - last_flush >= _STREAM_FLUSH_SECONDS:
            flush()

    on_delta.flush = flush
    return on_delta


# --- WebSocket Endpoint ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            raw = await websocket.receive_text()
            # Commands are additive: the UI only uses them to follow live story
            # text. Anything unparseable is ignored rather than fatal.
            try:
                msg = json.loads(raw)
            except (ValueError, TypeError):
                continue
            if not isinstance(msg, dict):
                continue
            kind = msg.get("type")
            story_id = msg.get("id")
            if story_id is None:
                continue
            story_id = int(story_id)
            if kind == "subscribe_story":
                manager.subscribe(websocket, story_id)
                # Catch the client up on whatever has been generated so far.
                text, title = _story_stream_snapshot(story_id)
                if title:
                    await websocket.send_text(f"story_title:{story_id}:{json.dumps(title)}")
                if text:
                    await websocket.send_text(
                        f"story_delta:{story_id}:{json.dumps(text)}"
                    )
            elif kind == "unsubscribe_story":
                manager.unsubscribe(websocket, story_id)
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


def _zipf_label(value):
    """Human-readable frequency band for a measured Zipf value, or None.

    A bare 5.49 tells a learner nothing actionable; "top 40k words" does. The
    rank is derived here rather than in the browser for the same reason the story
    coverage percentage is stored server-side: one definition, so the API and any
    future surface cannot disagree about what a number means. Zipf is
    log10(rank per billion), so rank = 10 ** (7 - zipf).
    """
    if value is None:
        return None
    # Clamped at 1: "the" is Zipf 7.7, which is a rank below one per billion and
    # rounds to "top 0 words" without this.
    rank = max(1.0, 10 ** (7 - float(value)))
    if rank < 1000:
        band = f"top {max(10, int(round(rank / 10) * 10))}"
    elif rank < 1_000_000:
        band = f"top {round(rank / 1000)}k"
    else:
        band = f"top {round(rank / 100_000) / 10:.1f}M"
    return f"Zipf {value:.1f} · {band} words"


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

    query = f"SELECT id, word, meaning, synonyms, use_region, freq_zipf, family, learned, encounters, createdAt, updatedAt FROM dictionary {order_clause}"
    conn = sqlite3.connect(DATABASE_URL)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()
    words = [dict(row) for row in rows]
    for entry in words:
        entry["freq_label"] = _zipf_label(entry.get("freq_zipf"))
        # Sense labels are DERIVED from the meaning text, not stored. They are
        # the first word of each sense, so the label vocabulary has to live in
        # exactly one place: `sense_labels()` reads it, and the browser consumes
        # this list positionally instead of parsing Markdown of its own. Edit a
        # label while double-clicking and the chip changes with the text.
        entry["sense_labels"] = openrouter_agent.sense_labels(entry.get("meaning"))
    return words


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
        meanings, use_region, synonyms = await asyncio.gather(
            run_in_threadpool(sync_get_meanings_of, *lookup_args),
            run_in_threadpool(sync_get_use_of, *lookup_args),
            run_in_threadpool(sync_get_synonyms_of, *lookup_args)
        )
        if not any(s.startswith("Error fetching") for s in (meanings, use_region, synonyms)):
            print(f"[add_word] Success on attempt {attempt} for '{word}'", flush=True)
            break
        err = next(s for s in (meanings, use_region, synonyms) if s.startswith("Error fetching"))
        print(f"[add_word] Attempt {attempt} failed for '{word}': {err}", flush=True)
        if attempt < MAX_ATTEMPTS:
            await asyncio.sleep(5)

    # Fail-closed: do not mutate DB if all attempts failed
    if err and any(s.startswith("Error fetching") for s in (meanings, use_region, synonyms)):
        raise HTTPException(status_code=502, detail=err)

    # The measured fields. Computed here rather than asked for: a model asked for
    # a frequency produces a confident invented number, and one asked for
    # inflections produces "runing". Both come from wordfreq/lemminflect, and both
    # are None rather than 0 when the word is unknown -- an absent measurement is
    # displayed as a dash, and 0 would read as "utterly rare", which is a claim.
    freq_zipf = lexicon.zipf_for(word)
    family = lexicon.family_for(word)

    def db_operation():
        conn = sqlite3.connect(DATABASE_URL)
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT id, encounters FROM dictionary WHERE word = ?", (word,))
            result = cursor.fetchone()
            if result:
                new_count = result[1] + 1
                cursor.execute(
                    "UPDATE dictionary SET encounters = ?, meaning = ?, synonyms = ?, use_region = ?, freq_zipf = ?, family = ?, updatedAt = CURRENT_TIMESTAMP WHERE word = ?",
                    (new_count, meanings, synonyms, use_region, freq_zipf, family, word)
                )
                message = f"'{word}' updated (count: {new_count})"
            else:
                # The createdAt and updatedAt columns will be filled by the DB's DEFAULT rule.
                cursor.execute(
                    "INSERT INTO dictionary (word, meaning, synonyms, use_region, freq_zipf, family, encounters) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (word, meanings, synonyms, use_region, freq_zipf, family, 1)
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
                "UPDATE dictionary SET word = ?, meaning = ?, synonyms = ?, use_region = ?, freq_zipf = ?, family = ?, encounters = ?, updatedAt = CURRENT_TIMESTAMP WHERE id = ?",
                (data.word, data.meaning, data.synonyms, data.use_region, data.freq_zipf, data.family, data.encounters, data.id)
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
    include_learnt: Optional[bool] = Query(None),
    include_unlearnt: Optional[bool] = Query(None),
    min_encounter: Optional[int] = Query(None),
):
    if field not in VALID_FILTER_FIELDS:
        raise HTTPException(status_code=400, detail="Invalid filter field.")
    if direction not in ("asc", "desc"):
        direction = "desc"
    sql_direction = "ASC" if direction == "asc" else "DESC"

    # min/max define a 1-based result-index range within the sorted list
    # (e.g. from=1, to=100 shows the first 100 results; 101-200 the next slice).
    #
    # An EMPTY "to" means NO LIMIT, and an empty "from" means start at the
    # beginning. That is the request, not an oversight: the number inputs are
    # pre-filled 1 and 200 so the row is honest about the usual page size, and
    # clearing them is how you ask for the whole list. limit stays None for
    # "unbounded" and the LIMIT clause is then left out of the SQL entirely,
    # rather than being given a huge number that looks like a real bound.
    offset = 0
    limit = None
    try:
        start = max(1, int(min_val)) if min_val not in (None, "") else None
        if start is not None:
            offset = start - 1
        if max_val not in (None, ""):
            end = max(1, int(max_val))
            limit = end - offset if start is not None else end
    except ValueError:
        raise HTTPException(status_code=400, detail="min/max must be integers (result positions).")

    # Which words the user wants offered for the next story: learnt, unlearnt,
    # both, or neither. Two independent booleans rather than one three-way
    # enum, so each state is one checkbox away instead of a dropdown cycle.
    #
    # These default to None, NOT False, and the difference is load-bearing:
    # absent means "no filter", which is what an old client sends, while both
    # explicitly false means "neither" -- the user unticked everything and
    # should see nothing. Collapsing the two would silently turn "I want none"
    # into "give me all 200".
    learned_clause = ""
    if include_learnt is not None or include_unlearnt is not None:
        parts = []
        if include_learnt:
            parts.append("learned = 1")
        if include_unlearnt:
            parts.append("learned = 0")
        # Applied in SQL and BEFORE the LIMIT, which is load-bearing rather than
        # tidiness: min/max are RESULT POSITIONS ("show me 1-100"), so filtering
        # after the limit would return the wrong page -- you would tick through
        # 100 rows, hit the end at 60, and quietly miss the rest.
        learned_clause = ("(" + " OR ".join(parts) + ")") if parts else "(0)"

    # Minimum encounters: the same rule as the learnt filter, and for the same
    # reason -- it narrows the SET before the range is applied, so "1 to 100" is
    # a slice of the words that meet the threshold rather than of the whole
    # dictionary. `encounters` is the dictionary's own counter, not the Zipf
    # measurement: this is "words I have actually met", which is what you want
    # when choosing material for a story.
    encounter_clause, encounter_args = "", []
    if min_encounter is not None:
        encounter_clause = "encounters >= ?"
        encounter_args = [max(1, int(min_encounter))]

    clauses = [c for c in (learned_clause, encounter_clause) if c]
    where_clause = ("WHERE " + " AND ".join(clauses)) if clauses else ""

    query = (
        f"SELECT id, word, meaning, synonyms, use_region, freq_zipf, family, learned, encounters, createdAt, updatedAt "
        f"FROM dictionary {where_clause} ORDER BY {field} {sql_direction}, id {sql_direction}"
    )
    # LIMIT/OFFSET are appended only when a range was actually asked for. With
    # no "to" the whole matching set is returned, and SQLite is happier with the
    # clause absent than with a sentinel limit.
    params = list(encounter_args)
    if limit is not None:
        query += " LIMIT ? OFFSET ?"
        params += [limit, offset]
    conn = sqlite3.connect(DATABASE_URL)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(query, tuple(params))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


@app.patch("/api/words/{word_id}/learned")
async def set_word_learned(word_id: int, learned: bool = Query(...)):
    """Mark or unmark one word as learnt.

    A dedicated single-field endpoint rather than another field on PUT
    /api/words. That PUT rewrites EVERY column from its payload, so folding a
    one-bit toggle into it means the toggle has to carry the whole row back --
    and that is exactly how the read-only fields were zeroed once already. One
    field cannot damage meaning, synonyms or the measured values, because it
    cannot name them.

    Deliberately does NOT touch updatedAt: that column means "this word's data
    changed", and a personal mark is not a change to the word -- letting it
    count would reshuffle the default "recently updated" sort every time the
    user reviews their vocabulary.
    """
    value = 1 if learned else 0

    def db_operation():
        conn = sqlite3.connect(DATABASE_URL)
        cursor = conn.cursor()
        try:
            cursor.execute(
                "UPDATE dictionary SET learned = ? WHERE id = ?", (value, word_id)
            )
            if cursor.rowcount == 0:
                return False, "Word not found"
            conn.commit()
            return True, "Word marked learnt" if value else "Word marked not learnt"
        except Exception as e:
            conn.rollback()
            return False, str(e)
        finally:
            conn.close()

    success, message = await run_in_threadpool(db_operation)
    if not success:
        raise HTTPException(status_code=404 if message == "Word not found" else 500, detail=message)
    # Same broadcast as PUT /api/words, so the grid and any open story modal
    # repaint from one signal and a mark made in either place shows everywhere.
    await manager.broadcast("update_words")
    return {"message": message, "learned": bool(value)}


# --- Stories (generated from words via OpenRouter) ---
def _sanitize_style(raw):
    """A per-story style override, accepted only if it is a known entry.

    The override arrives from the browser, so it is validated rather than
    trusted: an unknown string would otherwise be spliced straight into the
    prompt. Returns None for empty/unknown, which means "no override".
    """
    value = (raw or "").strip()
    if not value:
        return None
    if value in openrouter_agent.REDDIT_STYLES:
        return value
    print(f"[STORY] ignoring unknown style override: {value[:60]!r}")
    return None


class StoryCreate(BaseModel):
    words: List[str]
    # The style to use for THIS story, set from the style the preview showed, so
    # what you read in Story Setup is what gets sent. None means "follow the
    # preset", which is how a client that predates this field keeps working.
    style: Optional[str] = None
    # The reasoning level for THIS story only. Deliberately NOT stored: the rule
    # is that the default is always the cheapest thing the chosen model offers,
    # so a remembered value could only ever be a value the user did not choose
    # now. Carried on the request instead, validated against the model, and
    # recorded on the row by request_config.
    reasoning: Optional[str] = None
    # No `title`: the story title is always invented by the model (the prompt
    # carries an explicit instruction, not a placeholder to substitute into), so
    # a title field here could only ever be ignored. Pydantic drops it silently
    # if an old client still sends one.
    model: Optional[str] = None
    provider: Optional[str] = None


class StoryUpdate(BaseModel):
    content: str
    title: Optional[str] = None


class StoryPreview(BaseModel):
    words: List[str] = []
    model: Optional[str] = None
    provider: Optional[str] = None
    style: Optional[str] = None


# Shown in place of the style line when the preset says "random". The draw
# happens at generation time, so a preview that named a style would be a guess;
# this says what is actually true at that point.
# No random-style marker any more. main.py used to pass this literal string as
# the style to the preview endpoints, and _resolve_style() correctly treats a
# non-empty string as a concrete style -- so the PREVIEW rendered "Style for this
# story: (random -- one is drawn when the story starts)." and asked the model, in
# the preview at least, to imitate a sentence in parentheses. A preview is more
# useful showing a real REDDIT_STYLES entry, which is what the real request will
# carry anyway. The UI still says "random" on its own, from the empty preset
# field, so nothing needs to survive in the prompt text.


@app.post("/api/story/preview")
def api_preview_story(data: StoryPreview):
    """Everything a story generation will send, without sending it.

    Visual confirmation only: the same prompt the row will snapshot, plus the
    resolved request config. Unlike POST /api/prompts/preview this knows the
    model, so it can also report the reasoning config that will actually go out
    -- a preset asking for "off" against a mandatory-reasoning model cannot be
    honoured, and silently showing "off" there would be a lie.
    """
    words = [w.lower().strip() for w in data.words if w.strip()]
    if not words:
        raise HTTPException(status_code=400, detail="No words provided.")
    model = data.model or "openrouter/free"
    preset = _active_preset()
    config = _story_request_config()
    # The cap rides along here so the preview reports what will actually be
    # SENT, not what is stored: a non-reasoning model gets no cap at all, and
    # saying "2000" there would imply a bound that does not exist.
    reasoning = openrouter_agent.use_reasoning_status(
        model, None, config["reasoning_max_tokens"])

    # Resolve the style ONCE here and pass it down. build_prompt would draw its
    # own random entry internally, which is fine for the prompt but means the
    # config grid could only say "random". Resolving up front lets the preview
    # show a concrete style while the prompt still carries exactly that string.
    override = _sanitize_style(data.style)
    style = override or openrouter_agent._resolve_style(preset.get("style"))
    # Reuse the stored template so the preview cannot drift from what
    # create_story would render.
    messages = openrouter_agent.build_messages(
        words, template=preset.get("template"),
        style=style,
        system=preset.get("system"),
    )
    return {
        "preset": preset.get("name", ""),
        "model": model,
        "provider": data.provider or "auto",
        "style": (preset.get("style") or "").strip() or "random",
        # Which of the three sources supplied the style: the per-story pick, the
        # preset, or the draw. The UI needs it to word the row honestly.
        "style_source": "picked" if override else (
            "preset" if (preset.get("style") or "").strip() else "random"),
        "styles": openrouter_agent.REDDIT_STYLES,
        # The concrete entry when the preset has none, so the config grid can
        # show the style the preview actually rendered. It is a SAMPLE: the real
        # request draws its own, so the UI must label it as one.
        "resolved_style": style,
        "temperature": config["temperature"],
        "max_tokens": config["max_tokens"],
        # Reasoning cap, resolved the same way -- what will actually be sent,
        # not what the preset happens to say.
        "reasoning_max_tokens": (reasoning.get("config") or {}).get("max_tokens", 0),
        # ...and what the preset ASKED for. These differ whenever the model has
        # no token budget, and the UI needs both: the first to say what goes
        # out, the second to tell the user their stored value is kept rather
        # than lost ("2,000 kept for other models") instead of silently
        # disappearing along with the control.
        "reasoning_cap_requested": config["reasoning_max_tokens"],
        "reasoning": reasoning,
        "words_total": len(words),
        "system": messages[0]["content"],
        "user": messages[1]["content"],
        "findings": openrouter_agent.validate_prompt(preset.get("template"), preset.get("system")),
    }


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

    # Freeze the active prompt preset now, so the row records the prompt that
    # produced it and a later preset edit cannot alter this run.
    recipe = _resolve_prompt_recipe(words, data.style)
    # A reasoning level picked for THIS story, validated against the model that
    # will actually serve it. Applied to the frozen recipe rather than to the
    # settings, so request_config records the level the story really used.
    override = _sanitize_reasoning(data.reasoning, actual_model)
    if override:
        recipe["gen_kwargs"]["reasoning"] = override

    def db_operation():
        conn = sqlite3.connect(DATABASE_URL)
        cursor = conn.cursor()
        placeholders = ",".join("?" * len(words))
        cursor.execute(f"SELECT id FROM dictionary WHERE word IN ({placeholders})", words)
        word_ids = [row[0] for row in cursor.fetchall()]
        cursor.execute(
            "INSERT INTO stories (title, content, audio_path, model, duration_ms, cost, status, "
            "prompt_used, prompt_preset) "
            "VALUES (?, NULL, NULL, ?, NULL, NULL, 'generating', ?, ?)",
            ("Generating story...", actual_model,
             recipe["prompt_used"], _active_preset().get("name", "")),
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
    _start_story_job(story_id, words, actual_model, actual_provider,
                     recipe["gen_kwargs"])
    manager.broadcast_from_sync("update_stories")
    return {"id": story_id, "status": "generating"}


async def generate_story_job(story_id: int, words: List[str], model: str,
                             provider_tag: Optional[str] = None, gen_kwargs: Optional[dict] = None):
    """Generate story content in the background and persist it when done."""
    token = _new_story_stream_token()
    on_delta = _make_story_delta_sink(story_id, token)
    if gen_kwargs:
        # Logged because the health line further down cannot say which prompt
        # produced the run, and "which preset is better" is answered by
        # comparing health lines across presets.
        print(f"[STORY] story {story_id} using prompt preset "
              f"temp={gen_kwargs.get('temperature')} max_tokens={gen_kwargs.get('max_tokens')} "
              f"reasoning={gen_kwargs.get('reasoning', 'auto')}",
              flush=True)
    try:
        result = await run_in_threadpool(
            openrouter_agent.sync_generate_story,
            words, model, provider_tag, story_id, on_delta,
            **(gen_kwargs or {}),
        )
        # Always ship the tail of the stream before the terminal event so the UI
        # never renders a truncated story.
        try:
            on_delta.flush()
        except Exception:
            pass

        if not result.get("ok"):
            error = result.get("error", "Story generation failed")
            if error == openrouter_agent._STORY_ERR_CANCELLED:
                # cancel_story already wrote the failed row and announced it.
                print(f"[STORY] generation cancelled for story {story_id}", flush=True)
                return
            raise RuntimeError(error)

        content = result["content"]
        duration_ms = int(result.get("elapsed", 0.0) * 1000)
        cost = result.get("cost", 0.0)
        # Advisory only: display text for the warning pill, NULL when the run was
        # clean. Deliberately NOT recomputed by update_story — these describe what
        # the model produced, and re-running the density check on hand-edited
        # prose would flag the user's own writing as word salad.
        warnings = "; ".join(result.get("warnings") or []) or None
        # The model always writes the Title: line (the system message requires
        # it), so this is a read, not a request. There is no user-supplied title
        # to fall back to any more.
        title = _extract_title(content)
        # A story that hit max_tokens stops mid-sentence. Publish it anyway --
        # most of it is readable -- but as 'truncated' so the UI can say so and
        # offer a retry. A plain 'ready' would hide the missing ending.
        final_status = "truncated" if result.get("truncated") else "ready"

        def db_operation():
            conn = sqlite3.connect(DATABASE_URL)
            cursor = conn.cursor()
            # Only publish if the row is still ours and still generating: the
            # user may have cancelled or deleted it while we were streaming,
            # and a late completion must never resurrect it.
            cursor.execute("SELECT status FROM stories WHERE id = ?", (story_id,))
            row = cursor.fetchone()
            if not row or row[0] != "generating":
                conn.close()
                return False
            cursor.execute(
                "UPDATE stories SET title = ?, content = ?, model = ?, duration_ms = ?, cost = ?, "
                "prompt_tokens = ?, completion_tokens = ?, reasoning_tokens = ?, "
                "words_used = ?, words_total = ?, prose_words = ?, warnings = ?, "
                "style = ?, request_config = ?, "
                "status = ?, error = NULL, updatedAt = CURRENT_TIMESTAMP WHERE id = ? "
                "AND status = 'generating'",
                (
                    title, content, model, duration_ms, cost,
                    int(result.get("prompt_tokens") or 0),
                    int(result.get("completion_tokens") or 0),
                    int(result.get("reasoning_tokens") or 0),
                    int(result.get("words_used") or 0),
                    int(result.get("words_total") or 0),
                    int(result.get("prose_words") or 0),
                    warnings,
                    # The style actually sent, from the frozen recipe rather than
                    # the settings table: the row must describe THIS run even if
                    # the preset has been edited since.
                    (gen_kwargs or {}).get("style"),
                    # The request config this run was made with. It lives in
                    # shared settings now, so without this snapshot the row could
                    # not say what it ran with. Written once, never recomputed --
                    # same rule as warnings.
                    #
                    # `reasoning_effective` is what the reasoning setting
                    # RESOLVED to, alongside the level that was asked for. Both
                    # are needed: "off" is a request, not a guarantee, and a
                    # mandatory-reasoning model has no off switch -- so a row
                    # storing only the request would claim a story thought
                    # nothing when it thought 26,000 tokens. Missing on rows
                    # written before this change, which the reader falls back
                    # from, exactly as it falls back for `reasoning`.
                    json.dumps({
                        **{k: (gen_kwargs or {}).get(k) for k in
                           ("temperature", "max_tokens", "reasoning",
                            "reasoning_max_tokens")},
                        "reasoning_effective": result.get("reasoning_effective"),
                    }),
                    final_status,
                    story_id,
                ),
            )
            published = cursor.rowcount
            conn.commit()
            conn.close()
            return published

        published = await run_in_threadpool(db_operation)
        if not published:
            print(f"[STORY] discarded late result for story {story_id} (cancelled or deleted)", flush=True)
            return
        manager.broadcast_from_sync(f"story_ready:{story_id}")
    except asyncio.CancelledError:
        # The HTTP request is aborted by openrouter_agent.cancel_story_request;
        # this just stops the write and the broadcast.
        print(f"[STORY] generation task cancelled for story {story_id}", flush=True)
        raise
    except Exception as e:
        print(f"[STORY] background generation failed for story {story_id}: {e}", flush=True)

        def fail_operation():
            conn = sqlite3.connect(DATABASE_URL)
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE stories SET status = 'failed', error = ?, updatedAt = CURRENT_TIMESTAMP "
                "WHERE id = ? AND status = 'generating'",
                (str(e)[:500], story_id),
            )
            changed = cursor.rowcount
            conn.commit()
            conn.close()
            return changed

        changed = await run_in_threadpool(fail_operation)
        if changed:
            manager.broadcast_from_sync(f"story_error:{story_id}")
    finally:
        _story_stream_clear(story_id, token)
        # Only drop the registry entry if it still points at this job: the user
        # may have hit Retry, which installs a fresh task for the same story.
        current = asyncio.current_task()
        if _story_tasks.get(story_id) is current:
            _story_tasks.pop(story_id, None)


def _start_story_job(story_id: int, words: List[str], model: str,
                     provider_tag: Optional[str] = None, gen_kwargs: Optional[dict] = None):
    """Create the background generation task and remember it for cancel/delete."""
    task = asyncio.create_task(
        generate_story_job(story_id, words, model, provider_tag, gen_kwargs)
    )
    _story_tasks[story_id] = task
    return task


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


def _sanitize_reasoning(raw, model_id):
    """A per-story reasoning override, accepted only if THIS model offers it.

    Same reasoning as the style override: the string arrives from the browser, so
    it is checked against the model rather than trusted. A level the model does
    not list would either be rejected by the API or silently clamped to something
    the user never picked, and "off" on a model that always reasons is a switch
    that does nothing. Returns None for empty/unknown, which means "no override"
    -- and then the cheapest level is used.
    """
    value = (raw or "").strip()
    if not value:
        return None
    supported, _reason = openrouter_agent._reasoning_cap_support(model_id)  # noqa: F841
    meta = (openrouter_agent._reasoning_model(model_id) or {}).get("reasoning") or {}
    offered = list(meta.get("supported_efforts") or [])
    if value == "off" and not meta.get("mandatory", False):
        return value
    if value in offered:
        return value
    print(f"[STORY] ignoring reasoning override {value!r}: {model_id} does not offer it")
    return None


@app.get("/api/config")
def api_config():
    """Small public config the frontend needs to render provider-specific UI."""
    return {
        "ai_provider": AI_PROVIDER,
        "openrouter_ready": openrouter_agent.is_ready(),
        "meanings_model": get_setting("meanings_model") or "openrouter/free",
        "story_model": get_setting("story_model") or "openrouter/free",
        "story_provider": get_setting("story_provider") or "",
        # The request config, so the Story Setup controls can render without a
        # second request and every client agrees on what a story will be sent.
        **_story_request_config(),
        # Name of the active prompt preset, so the UI can label Generate
        # without a second request.
        "story_prompt_preset": _active_preset().get("name", ""),
    }


@app.post("/api/config")
def update_config(
    meanings_model: Optional[str] = Body(None),
    story_model: Optional[str] = Body(None),
    story_provider: Optional[str] = Body(None),
    temperature: Optional[float] = Body(None),
    max_tokens: Optional[int] = Body(None),
    reasoning_max_tokens: Optional[int] = Body(None),
):
    """Save model selections and the story request config to the DB.

    Single source of truth for all clients, matching how the model picks already
    behave. Each field is optional so one caller can change one thing; values
    are clamped on the way IN as well as on the way out, so a nonsense number
    cannot even be stored.
    """
    if meanings_model is not None:
        set_setting("meanings_model", meanings_model)
    if story_model is not None:
        set_setting("story_model", story_model)
    if story_provider is not None:
        set_setting("story_provider", story_provider)
    # The payload names are the preset field names, so the frontend sends the
    # same keys the config came back with and _story_request_config() can read
    # them back without a translation table.
    if temperature is not None:
        set_setting("story_temperature", str(max(0.0, min(2.0, float(temperature)))))
    if max_tokens is not None:
        set_setting("story_max_tokens", str(max(1000, min(32000, int(max_tokens)))))
    # 0 is allowed here (it means "no cap"), which is why it is not held to the
    # 1000 floor the output budget uses.
    if reasoning_max_tokens is not None:
        set_setting("story_reasoning_cap", str(max(0, min(100000, int(reasoning_max_tokens)))))
    return {"ok": True}


# --- Prompt presets (the editable story prompt) ---
#
# A preset is a full generation recipe (template + style + temperature +
# max_tokens), not just text: "preset A at whatever temperature was current" is
# not reproducible, and reproducibility is the whole point of storing them.
# The active preset id lives alongside them in the same settings row so one
# read/write covers the lot and the two can never disagree.
#
# The JSON is written whole (last write wins). That is deliberate and matches
# how the model picks already behave: single user, one browser in practice.
_PROMPTS_SETTING_KEY = "story_prompt_presets"
_DEFAULT_PRESET_ID = "default"

class PromptPreset(BaseModel):
    id: str
    name: str
    template: str
    style: str = ""
    system: str = openrouter_agent._DEFAULT_SYSTEM


class PresetPayload(BaseModel):
    """Upsert body for POST /api/prompts. Omit id to create.

    Deliberately has no temperature / max_tokens / reasoning / reasoning_cap:
    those are properties of the REQUEST, not of the prompt, and they depend on
    the model the user is about to pick. They live in shared settings beside
    story_model and story_provider instead -- see _story_request_config().
    """
    id: Optional[str] = None
    name: str
    template: str
    style: str = ""
    system: str = openrouter_agent._DEFAULT_SYSTEM


def _default_preset():
    return {
        "id": _DEFAULT_PRESET_ID,
        "name": "Reddit default",
        "template": openrouter_agent._DEFAULT_TEMPLATE,
        "style": "",  # empty = draw a random REDDIT_STYLES entry per story
        "system": openrouter_agent._DEFAULT_SYSTEM,
    }


def _load_prompt_presets():
    """Read the preset store, seeding the code default on first use.

    Also self-heals: a row that is not valid JSON, or one whose active id no
    longer exists, falls back to the default preset rather than leaving the UI
    with nothing selected. A corrupt prompt store must never be able to stop
    story generation, so the fallback is always a working preset.
    """
    raw = get_setting(_PROMPTS_SETTING_KEY)
    state = None
    if raw:
        try:
            state = json.loads(raw)
        except (ValueError, TypeError):
            print("[PROMPTS] stored presets are not valid JSON; using the code default.")
            state = None
    if not isinstance(state, dict) or not isinstance(state.get("presets"), list) or not state["presets"]:
        state = {"presets": [_default_preset()], "active": _DEFAULT_PRESET_ID}
        _save_prompt_presets(state)
    presets = [p for p in state["presets"] if isinstance(p, dict) and p.get("template") is not None]
    if not presets:
        presets = [_default_preset()]
    ids = {p.get("id") for p in presets}
    active = state.get("active")
    if active not in ids:
        active = presets[0].get("id")
    return {"presets": presets, "active": active}


def _save_prompt_presets(state):
    set_setting(_PROMPTS_SETTING_KEY, json.dumps(state))


# --- The request config (temperature / budget / reasoning) ---
#
# These four used to live in each prompt preset, which was the wrong home twice
# over: they depend on the model, and a preset does not know the model; and
# their two dependencies (story_model, story_provider) are already global
# settings, so the derived values were global's odd cousins. They now sit beside
# those two, are shared across clients like them, and are edited on the Story
# Setup screen -- the only surface that knows the model.
#
# The payoff is the comparison the app actually leans on: two presets that
# differ only in their prompt now produce comparable health lines, because
# nothing else differs between the two runs.
_STORY_CONFIG_SETTINGS = {
    "temperature": "story_temperature",
    "max_tokens": "story_max_tokens",
    # "reasoning" is deliberately NOT here: the level is chosen per story and
    # never stored, so a saved one would only ever be stale.
    "reasoning_max_tokens": "story_reasoning_cap",
}


def _story_request_config():
    """The shared request config, validated and clamped on the way out.

    Clamping lives HERE rather than at save time, deliberately: a stored value
    can be out of range (hand-edited row, older write) and this is the one place
    every consumer goes through, so a bad number cannot reach a request.
    """
    def _num(key, default, lo, hi):
        try:
            return max(lo, min(hi, int(float(get_setting(key)))))
        except (TypeError, ValueError):
            return default

    def _float(key, default):
        try:
            return max(0.0, min(2.0, float(get_setting(key))))
        except (TypeError, ValueError):
            return default

    return {
        "temperature": _float("story_temperature", openrouter_agent._STORY_TEMPERATURE),
        "max_tokens": _num("story_max_tokens", openrouter_agent._STORY_MAX_TOKENS, 1000, 32000),
        "reasoning_max_tokens": _num("story_reasoning_cap", 0, 0, 100000),
    }


def _migrate_story_config_to_settings():
    """Move the request config out of the presets, once.

    Seeding happens BEFORE anything is deleted, which is the whole safety
    property: the values land in settings first, so a crash between the two
    steps loses nothing. Idempotent -- an already-migrated store has neither the
    settings keys nor the preset fields, and does nothing.
    """
    state = _load_prompt_presets()
    active = next((p for p in state["presets"] if p.get("id") == state.get("active")), None)
    seeded = False
    for field, key in _STORY_CONFIG_SETTINGS.items():
        if get_setting(key) is not None:
            continue
        value = (active or {}).get(field)
        if value is None:
            continue  # nothing to move; the default in _story_request_config applies
        set_setting(key, str(value))
        seeded = True

    stripped = False
    for p in state["presets"]:
        for field in _STORY_CONFIG_SETTINGS:
            if field in p:
                p.pop(field)
                stripped = True
    if stripped:
        _save_prompt_presets(state)
    if seeded or stripped:
        print("[PROMPTS] moved temperature / output budget / reasoning out of the presets "
              "into shared settings.")


_migrate_story_config_to_settings()


def _active_preset():
    state = _load_prompt_presets()
    active = state["active"]
    for p in state["presets"]:
        if p.get("id") == active:
            return p
    return _default_preset()


def _resolve_prompt_recipe(words, style_override=None):
    """Freeze the active preset into everything one generation needs.

    Called once, when the story row is created, for two reasons:

    1. The prompt stored on the row is then exactly the prompt sent, so a story
       stays reproducible after the preset is edited or deleted.
    2. Editing the preset (or deleting the active preset) mid-generation cannot
       change a request already in flight.

    The random style is resolved here rather than inside the generation, so the
    snapshot cannot disagree with what was sent.
    """
    preset = _active_preset()
    override = _sanitize_style(style_override)
    style = override or openrouter_agent._resolve_style(preset.get("style"))
    messages = openrouter_agent.build_messages(words, template=preset.get("template"),
                                               style=style, system=preset.get("system"))
    config = _story_request_config()
    return {
        "prompt_used": f"[system]\n{messages[0]['content']}\n\n[user]\n{messages[1]['content']}",
        "gen_kwargs": {
            "template": preset.get("template"),
            "style": style,
            # The request config comes from shared settings, not the preset, and
            # is read here so the snapshot cannot shift under a run in flight.
            "temperature": config["temperature"],
            "max_tokens": config["max_tokens"],
            # No "reasoning" key at all: there is no stored level, so the
            # generation resolves the cheapest the model offers unless the
            # request carried a per-story override (see create_story).
            "reasoning_max_tokens": config["reasoning_max_tokens"],
            "system": preset.get("system"),
        },
    }


@app.get("/api/prompts")
def api_get_prompts():
    state = _load_prompt_presets()
    return {
        "presets": state["presets"],
        "active": state["active"],
        "default_template": openrouter_agent._DEFAULT_TEMPLATE,
        "default_system": openrouter_agent._DEFAULT_SYSTEM,
        "styles": openrouter_agent.REDDIT_STYLES,
        "reasoning_modes": list(openrouter_agent.REASONING_MODES),
        # The effort levels any catalogue model supports, so the dropdown offers
        # what exists rather than a hardcoded list that would drift.
        "reasoning_efforts": openrouter_agent.all_reasoning_efforts(),
    }


@app.post("/api/prompts")
def api_save_preset(payload: PresetPayload):
    """Create or update one preset. Whole-object upsert keyed by id."""
    name = (payload.name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="A preset needs a name.")
    if len(name) > 60:
        raise HTTPException(status_code=400, detail="Preset name is too long (60 characters max).")

    errors = [f for f in openrouter_agent.validate_prompt(payload.template, payload.system)
              if f["level"] == "error"]
    if errors:
        # Only an unrenderable template is refused. Warnings are advisory and
        # the UI shows them without blocking the save.
        raise HTTPException(status_code=400, detail=errors[0]["message"])

    temperature = max(0.0, min(2.0, float(payload.temperature)))
    max_tokens = max(1000, min(32000, int(payload.max_tokens)))
    preset = {
        "id": payload.id or uuid.uuid4().hex[:8],
        "name": name,
        "template": payload.template,
        "style": (payload.style or "").strip(),
        "system": payload.system,
    }
    existing = next((p for p in state["presets"] if p.get("id") == preset["id"]), None)
    if existing:
        existing.update(preset)
    else:
        state["presets"].append(preset)
        # A newly created preset does not steal activation; that stays an
        # explicit act so a stray click cannot change what Generate uses.
    _save_prompt_presets(state)
    return {"ok": True, "preset": preset, "active": state["active"]}


@app.post("/api/prompts/{preset_id}/activate")
def api_activate_preset(preset_id: str):
    state = _load_prompt_presets()
    if not any(p.get("id") == preset_id for p in state["presets"]):
        raise HTTPException(status_code=404, detail="No such preset.")
    state["active"] = preset_id
    _save_prompt_presets(state)
    return {"ok": True, "active": preset_id}


@app.delete("/api/prompts/{preset_id}")
def api_delete_preset(preset_id: str):
    state = _load_prompt_presets()
    if len(state["presets"]) <= 1:
        raise HTTPException(status_code=400, detail="The last preset cannot be deleted.")
    state["presets"] = [p for p in state["presets"] if p.get("id") != preset_id]
    if state["active"] == preset_id:
        # Deleting the active preset falls back rather than leaving a dangling
        # id that the generate path would have to special-case.
        state["active"] = state["presets"][0]["id"]
    _save_prompt_presets(state)
    return {"ok": True, "presets": state["presets"], "active": state["active"]}


class PromptValidate(BaseModel):
    # Echoed straight back. The Lab keeps a counter and drops any response whose
    # seq is not the newest, so two overlapping requests cannot paint a result
    # for text the user has already edited. Without the echo the client sees
    # undefined and discards EVERY response, findings included.
    seq: Optional[int] = None
    template: str
    system: Optional[str] = None


@app.post("/api/prompts/validate")
def api_validate_prompt(payload: PromptValidate):
    """Check a template without saving it. The UI calls this on a debounce."""
    return {"findings": openrouter_agent.validate_prompt(payload.template, payload.system),
            "seq": payload.seq}


class PromptPreview(BaseModel):
    template: str
    words: List[str] = []
    style: str = ""
    system: Optional[str] = None
    seq: Optional[int] = None


@app.post("/api/prompts/preview")
def api_preview_prompt(payload: PromptPreview):
    """Render the exact messages that would be sent, for the live preview.

    Server-side on purpose: the word list format and the system contract are
    the two things a reimplementation in JS would silently get wrong.
    """
    words = payload.words or ["barn", "ephemeral", "piece of cake", "run", "dread"]
    messages = openrouter_agent.build_messages(words, template=payload.template,
                                               style=payload.style or None,
                                               system=payload.system)
    return {
        "system": messages[0]["content"],
        "user": messages[1]["content"],
        "findings": openrouter_agent.validate_prompt(payload.template, payload.system),
        "seq": payload.seq,
    }


# Columns returned by the story read endpoints. Defined once so the list view,
# the word-filtered view and the detail view cannot drift apart when a column is
# added (the token columns are easy to forget in one of the three).
_STORY_FIELDS = (
    "id", "title", "content", "audio_path", "model", "duration_ms", "cost",
    "prompt_tokens", "completion_tokens", "reasoning_tokens",
    "words_used", "words_total", "prose_words", "warnings",
    # Name only: cheap enough for the list view, and it is how you later tell
    # which prompt produced which story. The prompt text itself is several KB
    # and is deliberately NOT here -- it would ride along on every list fetch.
    # It has its own endpoint, _STORY_PROMPT_SELECT below.
    "prompt_preset",
    # The request config this run was made with, as a small JSON blob. It is
    # here rather than parsed out of prompt_used because it is a handful of
    # numbers and the snapshot is several KB of prose. NULL on rows written
    # before the column existed, and never backfilled.
    "request_config",
    # The REDDIT_STYLES entry this story was written in. A bare TEXT column
    # because it has to render on the list row alongside the title, where the
    # snapshot is not fetched -- parsing it back out of prompt_used would be a
    # regex over prose the user may have edited. NULL on rows written before
    # the column existed, and the pill hides rather than guesses.
    "style",
    "status", "error", "createdAt", "updatedAt",
)
_STORY_SELECT = ", ".join(_STORY_FIELDS)
_STORY_SELECT_PREFIXED = ", ".join(f"s.{f}" for f in _STORY_FIELDS)
# The full prompt is fetched on demand, never with the list.
_STORY_PROMPT_SELECT = "prompt_used, prompt_preset"


def _reasoning_pill(raw):
    """(label, verified) for a story's reasoning. Label is None to show nothing.

    Built server-side from `request_config` for the same reason `freq_label` is:
    one definition, so the list row, the modal and any future surface cannot
    disagree about what a stored number means.

    Prefers the EFFECTIVE value, which is the only honest answer when a preset
    asks `off` against a mandatory-reasoning model that has no off switch.

    `verified` says whether the label came from the effective value or is only
    the level that was requested. Rows written before `reasoning_effective`
    existed fall back to the request, and a request is NOT proof the model
    complied -- so the pill may say `off` while the story thought 26,000
    tokens. The UI qualifies those in the tooltip instead of pretending.
    """
    if not raw:
        return None, False
    try:
        config = json.loads(raw) if isinstance(raw, str) else dict(raw)
    except (ValueError, TypeError):
        return None, False
    if not config:
        return None, False

    cap = config.get("reasoning_max_tokens") or 0
    if cap:
        # A cap REPLACES the level: OpenRouter rejects both together, so a label
        # like "cap 2000/low" would name something that was never sent. Formatted
        # without a thousands separator to match the health log's `cap 2000`.
        return f"cap {int(cap)}", True

    effective = config.get("reasoning_effective")
    if effective:
        return str(effective), True

    requested = config.get("reasoning")
    if not requested or requested == "not-set":
        # No reasoning was sent at all -- a non-reasoning model, or no setting.
        # Nothing was requested either, so "none" is not a claim about the model.
        return "none", True
    return str(requested), False


def _reasoning_label(raw):
    """Just the pill text. Kept as its own name for callers that don't care
    whether the value was verified."""
    return _reasoning_pill(raw)[0]


@app.get("/api/stories")
def api_list_stories(word_id: Optional[int] = Query(None)):
    conn = sqlite3.connect(DATABASE_URL)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    if word_id is not None:
        cursor.execute(
            f"SELECT {_STORY_SELECT_PREFIXED} "
            "FROM stories s JOIN story_words sw ON sw.story_id = s.id "
            "WHERE sw.word_id = ? ORDER BY s.createdAt DESC, s.id DESC",
            (word_id,),
        )
    else:
        cursor.execute(
            f"SELECT {_STORY_SELECT} "
            "FROM stories ORDER BY createdAt DESC, id DESC"
        )
    stories = [dict(row) for row in cursor.fetchall()]

    for s in stories:
        s["reasoning_label"], s["reasoning_verified"] = _reasoning_pill(s.get("request_config"))
        cursor.execute(
            "SELECT d.id, d.word, d.learned FROM story_words sw "
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
        f"SELECT {_STORY_SELECT} FROM stories WHERE id = ?",
        (story_id,),
    )
    row = cursor.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Story not found")
    story = dict(row)
    # Same derived label as the list endpoint, so a story opened directly by URL
    # shows the same pill as one clicked in the list.
    story["reasoning_label"], story["reasoning_verified"] = _reasoning_pill(
        story.get("request_config"))
    cursor.execute(
        "SELECT d.id, d.word, d.learned FROM story_words sw "
        "JOIN dictionary d ON d.id = sw.word_id WHERE sw.story_id = ?",
        (story_id,),
    )
    story["words"] = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return story


@app.get("/api/stories/{story_id}/prompt")
def api_story_prompt(story_id: int):
    """The exact prompt that produced this story, for the Lab's as-used view.

    NULL for stories generated before the column existed; the UI hides the link
    rather than showing an empty panel. Never backfilled -- a reconstructed
    prompt would be indistinguishable from a real one.
    """
    conn = sqlite3.connect(DATABASE_URL)
    row = conn.execute(
        f"SELECT {_STORY_PROMPT_SELECT} FROM stories WHERE id = ?", (story_id,)
    ).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Story not found")
    return {"id": story_id, "prompt_used": row[0], "prompt_preset": row[1]}


@app.get("/api/stories/{story_id}/word-usage")
def api_story_word_usage(story_id: int):
    """Per-word "did it appear" flags for the "Built from" tab.

    Uses openrouter_agent._word_pattern -- the same matcher behind
    _count_words_used -- so these chips can never disagree with the coverage
    pill, which matters because that matcher handles inflections, irregulars
    and multi-word phrases.

    A separate endpoint rather than extra fields on api_get_story because the
    per-word pass costs ~160-380ms on a 309-word list, and most modal opens
    never look at the tab. Being on demand also means it works for rows whose
    words_used is still NULL (pre-migration stories): it measures rather than
    reading a stored figure.

    Synchronous, like its neighbours, so FastAPI runs it in the threadpool.
    """
    conn = sqlite3.connect(DATABASE_URL)
    cursor = conn.cursor()
    cursor.execute("SELECT content FROM stories WHERE id = ?", (story_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Story not found")
    content = row[0]
    cursor.execute(
        "SELECT d.id, d.word, d.learned FROM story_words sw "
        "JOIN dictionary d ON d.id = sw.word_id WHERE sw.story_id = ? ORDER BY d.id",
        (story_id,),
    )
    rows = cursor.fetchall()
    conn.close()
    if not rows:
        return []
    # The id and the current mark ride along because the modal marks words from
    # here. Keying that on the word TEXT would be fragile: a rename while the
    # modal is open would let a stale click write to whichever word now holds
    # that string.
    words = [{"id": r[0], "word": r[1], "learned": bool(r[2])} for r in rows]
    body = openrouter_agent._story_body(content)
    used = []
    for entry in words:
        pattern = openrouter_agent._word_pattern(entry["word"])
        used.append({
            "id": entry["id"],
            "word": entry["word"],
            "learned": entry["learned"],
            "used": bool(pattern and pattern.search(body)),
        })
    return used


@app.post("/api/stories/{story_id}/retry")
async def retry_story(story_id: int):
    """Re-run background generation for a failed or truncated story.

    Truncated stories are retryable too: they were cut off by max_tokens, so a
    second run of the same word list is the natural way to get the full story.
    """
    def fetch_and_reset():
        conn = sqlite3.connect(DATABASE_URL)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT id, title, model, status FROM stories WHERE id = ?", (story_id,))
        story = cursor.fetchone()
        if not story:
            conn.close()
            return None, None, None
        if story["status"] not in ("failed", "truncated"):
            conn.close()
            return "not_retryable", None, None
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
    if outcome == "not_retryable":
        raise HTTPException(status_code=400, detail="Only failed or truncated stories can be retried")
    if outcome == "no_words":
        raise HTTPException(status_code=400, detail="Story has no linked words to regenerate from")

    actual_model = model or "openrouter/free"

    # A retry regenerates the content, so it snapshots the CURRENT active preset
    # and overwrites the old snapshot. The invariant is "prompt_used describes
    # the content currently in this row", not "the first prompt ever tried" --
    # keeping the old one would misdescribe the prose the user is now reading.
    recipe = _resolve_prompt_recipe(words)
    conn = sqlite3.connect(DATABASE_URL)
    conn.execute(
        "UPDATE stories SET prompt_used = ?, prompt_preset = ? WHERE id = ?",
        (recipe["prompt_used"], _active_preset().get("name", ""), story_id),
    )
    conn.commit()
    conn.close()

    _start_story_job(story_id, words, actual_model, None, recipe["gen_kwargs"])
    manager.broadcast_from_sync("update_stories")
    return {"id": story_id, "status": "generating"}


@app.post("/api/stories/{story_id}/cancel")
async def cancel_story(story_id: int):
    """Cancel an in-flight story generation request."""
    # 1. Close the streamed HTTP request. This is what actually stops the work.
    aborted = openrouter_agent.cancel_story_request(story_id)
    # 2. Unwind the coroutine so it stops before writing/announcing anything.
    task = _story_tasks.pop(story_id, None)
    if task is not None and not task.done():
        task.cancel()
    # 3. Mark DB as failed (only if still generating).
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
    return {"ok": True, "aborted": aborted}


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
        # Coverage is a measurement of the prose, so a hand-edit makes the stored
        # figures wrong. Re-measure against the story's linked words instead of
        # leaving a number that describes text that no longer exists.
        cursor.execute(
            "SELECT LOWER(d.word) FROM story_words sw "
            "JOIN dictionary d ON d.id = sw.word_id WHERE sw.story_id = ? ORDER BY d.id",
            (story_id,),
        )
        words = [row[0] for row in cursor.fetchall()]
        if words:
            cursor.execute(
                "UPDATE stories SET words_used = ?, words_total = ?, prose_words = ? WHERE id = ?",
                (
                    openrouter_agent._count_words_used(data.content, words),
                    len(words),
                    openrouter_agent._prose_word_count(openrouter_agent._story_body(data.content)),
                    story_id,
                ),
            )
        else:
            cursor.execute(
                "UPDATE stories SET words_used = NULL, words_total = NULL, prose_words = NULL WHERE id = ?",
                (story_id,),
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
    # Abort any in-flight generation so it stops burning tokens against a row
    # that is about to disappear.
    if openrouter_agent.cancel_story_request(story_id):
        task = _story_tasks.pop(story_id, None)
        if task is not None and not task.done():
            task.cancel()
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


@app.get("/api/tts/engines")
def api_tts_engines(context: str = Query("stories")):
    """Every local TTS engine with its availability and voices, for one context.

    Unavailable engines come back with a `reason` rather than being hidden:
    an engine whose model has not been downloaded yet is a fact the picker
    needs in order to say *why* instead of just omitting the row.
    """
    ctx = _check_tts_context(context)
    return {
        "context": ctx,
        "selected": tts_selected(ctx),
        "engines": [
            {
                "id": s.id,
                "label": s.label,
                "available": s.available,
                "reason": s.reason,
                "note": s.note,
                "default_voice": s.default_voice,
                "voices": [{"id": v.id, "label": v.label, "lang": v.lang} for v in s.voices],
            }
            for s in tts_status()
        ],
    }


@app.post("/api/tts/selection")
def api_tts_selection(
    context: str = Body("stories"),
    engine: Optional[str] = Body(None),
    voice: Optional[str] = Body(None),
):
    """Persist the chosen engine and voice for one context (shared by all clients).

    Both are validated against the engine's own voice list: a voice that does
    not exist for the chosen engine is a 400, not a silent fallback, because the
    alternative is audio in a voice the user did not pick and cannot see the
    name of anywhere in the UI.
    """
    ctx = _check_tts_context(context)
    # Validated here rather than left to `set_selection` (which is the forgiving
    # startup path): a voice that does not exist is the caller's mistake, and
    # answering 200 while quietly keeping the old one would leave the UI showing
    # a selection the server never accepted.
    wanted_engine = engine or ""
    if wanted_engine:
        try:
            target = get_tts_engine(ctx, engine_id=wanted_engine)
        except KeyError as bad:
            raise HTTPException(status_code=400, detail=str(bad))
        if voice and voice not in [v.id for v in target.voices()]:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unknown voice {voice!r} for {target.id}; "
                    f"have {sorted(v.id for v in target.voices())}"
                ),
            )
    elif voice:
        target = get_tts_engine(ctx)
        if voice not in [v.id for v in target.voices()]:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unknown voice {voice!r} for {target.id}; "
                    f"have {sorted(v.id for v in target.voices())}"
                ),
            )
    try:
        set_tts_selection(ctx, engine=wanted_engine, voice=voice or "")
    except KeyError as bad:
        raise HTTPException(status_code=400, detail=str(bad))
    # The stored pair must always be coherent: switching engine without naming a
    # voice would otherwise keep the previous engine's voice id, which synthesis
    # silently ignores (see _selected_voice) while /api/tts/engines reports it as
    # the selection. Resolving it here means the picker, the API and the audio
    # that actually plays all name the same voice.
    selection = tts_selected(ctx)
    set_setting(_tts_engine_key(ctx), selection["engine"])
    set_setting(_tts_voice_key(ctx), selection["voice"])
    return {"ok": True, "context": ctx, **selection}


def _selected_voice(context: str) -> str:
    """The voice to synthesize with for a context, or '' for the engine default."""
    stored = get_setting(_tts_voice_key(context)) or ""
    if _voice_belongs(stored, get_tts_engine(context)):
        return stored
    return ""


@app.post("/api/tts/sample")
def api_tts_sample(
    text: str = Body(..., embed=True),
    words: int = Body(10, embed=True),
    context: str = Body("stories"),
):
    """Synthesize a short sample: the first `words` of the text, or all of it.

    Used by the "try this voice" button in two places -- the story modal and the
    toolbar -- with different intents, which is why the count is a parameter:
    the story modal samples that story's opening (10 words, in context), while
    the toolbar samples a fixed reference line (`words: 0` = all of it) so two
    voices are compared on exactly the same text.
    """
    ctx = _check_tts_context(context)
    # words <= 0 means "no truncation". A falsy-zero folded into the default
    # would silently cut the reference line to 10 words, which is precisely the
    # thing the caller asked not to happen.
    limit = max(0, int(words or 0))
    excerpt = " ".join(str(text).split())
    if limit:
        excerpt = " ".join(excerpt.split()[:limit])
    if not excerpt:
        raise HTTPException(status_code=400, detail="no text to sample")
    try:
        wav = synth_wav(excerpt, _selected_voice(ctx), context=ctx)
    except EngineUnavailable as error:
        raise HTTPException(status_code=503, detail=str(error))
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error))
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    selection = tts_selected(ctx)
    return Response(
        wav,
        media_type="audio/wav",
        headers={
            # The excerpt is deterministic given (engine, voice, words), so the
            # browser may cache it briefly and re-clicking "sample" is instant.
            "Cache-Control": "no-store",
            "X-TTS-Engine": selection["engine"],
            "X-TTS-Voice": selection["voice"],
            "X-TTS-Context": ctx,
        },
    )


@app.post("/api/tts")
async def text_to_speech(text: str = Body(..., embed=True)):
    """Word/meaning audio -- the per-cell Listen buttons, so the meanings voice."""
    wav = await run_in_threadpool(synth_wav, text, _selected_voice("meanings"), context="meanings")
    return Response(wav, media_type="audio/wav")


def _clear_story_audio(story_id: int) -> bool:
    """Delete a story's audio file and clear the column. Returns True if there was one.

    Used by both "remove audio" and "regenerate": a regenerated file must not
    leave the old one behind, and `audio/` is otherwise append-only, so every
    edit-then-regenerate cycle quietly grew the directory.
    """
    conn = sqlite3.connect(DATABASE_URL)
    cursor = conn.cursor()
    cursor.execute("SELECT audio_path FROM stories WHERE id = ?", (story_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Story not found")
    audio_path = row[0]
    cursor.execute(
        "UPDATE stories SET audio_path = NULL WHERE id = ?", (story_id,)
    )
    conn.commit()
    conn.close()
    if audio_path:
        full = os.path.join("audio", audio_path)
        if os.path.exists(full):
            try:
                os.remove(full)
            except OSError as error:
                print(f"[TTS] could not delete {full}: {error}", file=sys.stderr, flush=True)
    return bool(audio_path)


@app.delete("/api/stories/{story_id}/audio")
async def delete_story_audio(story_id: int):
    """Remove a story's saved narration, keeping the story itself.

    Separate from editing the prose because the two are independent decisions:
    you may want to re-read an unchanged story in a new voice, or drop an audio
    file you no longer want, without touching a single character of the text.
    """
    removed = await run_in_threadpool(_clear_story_audio, story_id)
    await manager.broadcast("update_stories")
    return {"ok": True, "removed": removed}


@app.post("/api/tts/save")
async def text_to_speech_save(
    text: str = Body(..., embed=True),
    story_id: Optional[int] = Body(None, embed=True),
    force: bool = Body(False, embed=True),
):
    """Generate audio with the stories engine and persist it.

    - With a `story_id`: the (heavy) generation runs in the background. The
      client gets an immediate `{"status": "generating"}` and is notified over
      the WebSocket (`story_audio_progress:<id>:<pct>` / `story_audio_ready:<id>`)
      when done. This prevents a multi-minute blocking HTTP request.
    - With `force`: an existing file is discarded first, which is what the
      "Regenerate" button means. The old file is removed rather than orphaned;
      see `_clear_story_audio`.
    - Without a `story_id`: generates synchronously and returns the filename.
    """
    if story_id is not None:
        if force:
            await run_in_threadpool(_clear_story_audio, story_id)
            await manager.broadcast("update_stories")
        asyncio.create_task(generate_story_audio(story_id, text))
        return {"status": "generating", "story_id": story_id}

    wav = await run_in_threadpool(synth_wav, text, _selected_voice("stories"), context="stories")
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

        # The story narration uses the "stories" selection, which is a different
        # row from the "meanings" one -- a word can be read by a clear voice
        # while the story is narrated by an expressive one.
        wav = await run_in_threadpool(
            synth_wav, text, _selected_voice("stories"), on_progress, "stories"
        )

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



