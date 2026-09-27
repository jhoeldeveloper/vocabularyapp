import os
import re
import sys
import json
import time
import threading
import requests
import random
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# Both meanings and stories default to the free-tier router. Users override
# per-request via the picker (localStorage). Provider is backend AI_PROVIDER.
FALLBACK_MODEL = "openrouter/free"
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", FALLBACK_MODEL)
OPENROUTER_MEANINGS_TEMPERATURE = float(os.getenv("OPENROUTER_MEANINGS_TEMPERATURE", "0.3"))
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
MODELS_URL = "https://openrouter.ai/api/v1/models"

# Lazily-loaded cache of OpenRouter's public model catalogue. Shared by
# sync_list_models() (picker) and _get_pricing() (cost calculation) so the
# /models endpoint is fetched once per TTL window, with stale-on-error fallback.
_models_cache = {"data": None, "fetched_at": 0.0}
_MODELS_TTL = 600  # seconds

# Cache of per-model provider endpoint data (authenticated). Keyed by model_id.
_providers_cache: dict = {}
_PROVIDERS_TTL = 300  # seconds

# In-flight de-dup of concurrent lookups for the same (word, model). main.py
# fans out to sync_get_meanings_of / sync_get_sentences_with / sync_get_synonyms_of
# in parallel — without this, three identical HTTP requests would race. Keys
# are (word, model); values are threading.Event + the shared result dict.
_lookup_inflight: dict = {}

# Active story generation requests keyed by story_id. The story request is
# streamed, so a Response object exists for the whole generation and closing it
# genuinely aborts the in-flight body read (see sync_generate_story).
# Values: {"cancelled": bool, "response": requests.Response | None}.
_story_controls: dict = {}
_story_controls_lock = threading.Lock()

# Hard total elapsed timeout for story generation (8 minutes), enforced inside
# the stream loop rather than after the fact.
_STORY_MAX_ELAPSED = 480

# Connect / per-read socket timeouts for the streamed story request. The read
# timeout is deliberately short: it only has to cover the gap between two SSE
# chunks, so a stalled provider fails fast instead of hanging until the
# overall deadline.
_STORY_CONNECT_TIMEOUT = 10
_STORY_READ_TIMEOUT = 30

# Error strings shared with the UI / log.
_STORY_ERR_CANCELLED = "Story generation cancelled."
_STORY_ERR_DISCONNECTED = "Error generating story: connection closed."
_STORY_ERR_TIMEOUT = (
    "Error generating story: request timed out. The model may be slow or rate-limited "
    "for this many words. Try a smaller word set or a different model."
)
_STORY_ERR_DEADLINE = (
    "Story generation timed out after 8 minutes. The provider may be too slow "
    "— try a different model or provider."
)
# A story cut off by max_tokens is still published (status 'truncated'), but only
# if there is a useful amount of prose left. Below this many visible tokens the
# "story" is a sentence or two and is not worth keeping.
_STORY_MIN_VISIBLE_TOKENS = 50

# raw_decode on this gives (record, chars_consumed), which is how the stream
# reader tells one SSE record from several packed onto a single line.
_JSON_DECODER = json.JSONDecoder()


def _register_story_control(story_id) -> None:
    """Mark a story generation as active and not yet cancelled."""
    if story_id is None:
        return
    with _story_controls_lock:
        _story_controls[story_id] = {"cancelled": False, "response": None}


def _set_story_response(story_id, response) -> None:
    """Attach the streaming Response so cancel_story_request can close it."""
    if story_id is None:
        return
    with _story_controls_lock:
        control = _story_controls.get(story_id)
        if control is None:
            return
        control["response"] = response
        # Cancelled before the response even arrived: close it right away.
        if control["cancelled"]:
            _safe_close(response)


def _release_story_control(story_id) -> None:
    if story_id is None:
        return
    with _story_controls_lock:
        control = _story_controls.pop(story_id, None)
    if control:
        _safe_close(control.get("response"))


def _story_cancelled(story_id) -> bool:
    """True if the user asked to cancel this story (before or during the request)."""
    if story_id is None:
        return False
    with _story_controls_lock:
        control = _story_controls.get(story_id)
        return bool(control and control["cancelled"])


def _safe_close(response) -> None:
    if response is None:
        return
    try:
        response.close()
    except Exception:
        pass


def cancel_story_request(story_id: int) -> bool:
    """Abort the in-flight HTTP request for a story.

    Flips the story's control flag (so a request still waiting on response
    headers bails out as soon as it can) and closes the streaming Response to
    tear down the socket immediately. Returns True if a live request was found.
    """
    with _story_controls_lock:
        control = _story_controls.get(story_id)
        if control is None:
            return False
        control["cancelled"] = True
        response = control.get("response")
    _safe_close(response)
    return True


def _get_reasoning_config(model_id: str):
    """Return the reasoning config dict for a model, or None if not a reasoning model.

    Logic:
    - No reasoning field → non-reasoning model, return None
    - mandatory=False → disable reasoning (effort: "none")
    - mandatory=True → can't disable, use lowest supported effort
    - Meta-routers (openrouter/free, openrouter/auto) → disable reasoning (safe default)
    """
    if model_id in ("openrouter/free", "openrouter/auto"):
        return {"effort": "none", "exclude": True}
    models, _ = _fetch_models()
    model = next((m for m in models if m["id"] == model_id), None)
    if not model:
        return None
    reasoning = model.get("reasoning")
    if not reasoning:
        return None
    if not reasoning.get("mandatory", False):
        return {"effort": "none", "exclude": True}
    efforts = reasoning.get("supported_efforts") or ["low"]
    lowest = efforts[-1] if efforts else "low"
    return {"effort": lowest, "exclude": True}


def _reason_cfg_label(reasoning_config) -> str:
    """Compact label for request logs: none / low / not-set."""
    if not reasoning_config:
        return "not-set"
    effort = reasoning_config.get("effort")
    return effort if effort else "on"


def _fetch_models(force=False):
    now = time.monotonic()
    if not force and _models_cache["data"] is not None and now - _models_cache["fetched_at"] < _MODELS_TTL:
        return _models_cache["data"], None
    try:
        r = requests.get(MODELS_URL, timeout=15)
        if r.ok:
            models = []
            for m in r.json().get("data", []):
                p = m.get("pricing") or {}
                models.append({
                    "id": m["id"],
                    "name": m.get("name") or m["id"],
                    "pricing": {
                        "prompt": float(p.get("prompt") or 0),
                        "completion": float(p.get("completion") or 0),
                    },
                    "reasoning": m.get("reasoning"),
                })
            _models_cache["data"] = models
            _models_cache["fetched_at"] = now
        elif _models_cache["data"] is None:
            return [], f"OpenRouter /models returned {r.status_code}"
    except Exception as e:
        if _models_cache["data"] is None:
            return [], str(e)
    return _models_cache["data"] or [], None


def sync_list_models():
    """Return the full OpenRouter model catalogue (cached ~10 min)."""
    models, error = _fetch_models()
    return {"ok": error is None, "models": models, **({"error": error} if error else {})}


def _get_pricing():
    models, _ = _fetch_models()
    return {m["id"]: m["pricing"] for m in models}


def sync_list_providers(model_id: str) -> dict:
    """Return providers for a model with real throughput/latency data.

    Calls the authenticated /endpoints API so throughput_last_30m and
    latency_last_30m are populated (they are null without auth).
    Results are cached per model for _PROVIDERS_TTL seconds.

    Returns ``{"ok": True, "providers": [...]}`` on success, or
    ``{"ok": False, "error": "..."}`` on failure.  Models that don't
    support endpoints (e.g. ``openrouter/free``) return an empty list.
    """
    if not OPENROUTER_API_KEY:
        return {"ok": False, "error": "OpenRouter API key not set."}

    # openrouter/free and openrouter/auto don't have per-provider endpoints.
    if model_id in ("openrouter/free", "openrouter/auto", ""):
        return {"ok": True, "providers": []}

    now = time.monotonic()
    cached = _providers_cache.get(model_id)
    if cached and now - cached["fetched_at"] < _PROVIDERS_TTL:
        return {"ok": True, "providers": cached["data"]}

    try:
        # Build the full endpoint URL.  Model ids can contain slashes
        # (e.g. "z-ai/glm-5.3-flash"), so split only on the first one.
        parts = model_id.split("/", 1)
        if len(parts) == 2:
            author, slug = parts
        else:
            author, slug = "openrouter", model_id
        url = f"https://openrouter.ai/api/v1/models/{author}/{slug}/endpoints"

        r = requests.get(
            url,
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
            timeout=15,
        )
        if not r.ok:
            return {"ok": False, "error": f"OpenRouter endpoints returned {r.status_code}"}

        raw_endpoints = r.json().get("data", {}).get("endpoints", [])

        # Filter out providers that are down (status == -5).
        providers = []
        for ep in raw_endpoints:
            if ep.get("status") == -5:
                continue
            tp = ep.get("throughput_last_30m") or {}
            lat = ep.get("latency_last_30m") or {}
            providers.append({
                "tag": ep.get("tag", ""),
                "provider_name": ep.get("provider_name", ""),
                "quantization": ep.get("quantization", "unknown"),
                "pricing": {
                    "prompt": float((ep.get("pricing") or {}).get("prompt") or 0),
                    "completion": float((ep.get("pricing") or {}).get("completion") or 0),
                },
                "discount": float((ep.get("pricing") or {}).get("discount") or 0),
                "throughput_p50": tp.get("p50"),
                "throughput_p90": tp.get("p90"),
                "latency_p50": lat.get("p50"),
                "latency_p90": lat.get("p90"),
                "status": ep.get("status", 0),
                "uptime_last_1d": ep.get("uptime_last_1d"),
            })

        # Sort by composite score: cheap-first with speed as tiebreaker.
        # Price dominates; fast providers get up to 50% score reduction.
        def _provider_score(p):
            price = p["pricing"]["prompt"] + p["pricing"]["completion"]
            tp = p.get("throughput_p50") or 1
            speed_bonus = min(0.5, 50 / tp)
            return price * (1 + speed_bonus)

        providers.sort(key=_provider_score)

        _providers_cache[model_id] = {"data": providers, "fetched_at": now}
        return {"ok": True, "providers": providers}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def is_ready() -> bool:
    return bool(OPENROUTER_API_KEY)

REDDIT_STYLES = [
    "r/tifu: the narrator causes an embarrassing or chaotic disaster by mistake",
    "r/AmItheAsshole: a conflict with a friend, roommate or family member, told so readers can judge who was right",
    "r/nosleep: a creepy, suspenseful story the narrator insists is true",
    "r/pettyrevenge: someone gets even in a clever, satisfying way",
    "r/MaliciousCompliance: someone follows a ridiculous instruction to the letter, with hilarious results",
    "r/entitledparents: the narrator deals with an unreasonable person in an everyday situation",
    "r/talesfromretail: a strange day at work with unforgettable customers or coworkers",
    "r/relationships: a personal situation with a friend, partner or family, told honestly and emotionally",
]

def _build_prompt(words, title):
    style = random.choice(REDDIT_STYLES)
    # A numbered block keeps multi-word phrases intact and stops the model from
    # reading the words as part of the instructions.
    word_list = "\n".join(f"{i + 1}. {w}" for i, w in enumerate(words))
    title_instruction = (
        f"Use exactly this title, word for word: '{title}'." if title
        else "Invent a short, catchy title of at most 8 words."
    )
    return f"""Write one enjoyable, coherent story that reads like a real post on Reddit, and that uses every word in this list:

WORDS TO USE
{word_list}

Style for this story: {style}.

PLOT
- First person, casual and conversational, like telling a friend what happened. Open with a hook in the first two sentences.
- One narrator with a clear goal, one place that changes as the story moves, a problem in the middle, and an unexpected ending whose twist is set up by details planted earlier.
- Every scene follows from the previous one. No restarting, no unrelated jumps.
- Vivid details, realistic dialogue, honest reactions, small funny or awkward moments.

USING THE WORDS
- The list is in random order. Do not work through it in order: place each word in whichever scene it fits most naturally, and never force two unrelated words into one sentence.
- Use each word at least once, at most twice, in its standard sense, in a sentence where the meaning is clear from context alone.
- A variant of a word counts as using it: run/running/ran, quick/quickly, look/looked/looking, and for a phrase, one word of the phrase inflected.

BOLDING (graded strictly)
- Bold exactly one occurrence of each word in the list, using **double asterisks**. The bolded word may be a variant of the listed one.
- Bold NOTHING else. A bolded word that is not in the list is a failure.
  Example with a list of [run, ephemeral, piece of cake]:
  Right: a **run** in the rain, an **ephemeral** victory, a **piece of cake** to fix it.
  Wrong: I **use** my phone, I was **already** late, **not** a good idea. <-- none of these are listed words
- If a listed word is an everyday word, use it in its less obvious sense and still bold only that one occurrence.
- The title must not be bolded.

NEVER NARRATE YOUR OWN WORK
- Your answer is the story. Nothing else goes in it.
- Do not mention the list, the checklist, the draft, or which words you have or have not used. Never write a sentence like "Not a listed word." or "use edible already used" or "not listed".
- No self-corrections: nothing in parentheses, and no aside after a dash, comma or "and" that comments on the writing instead of the events.
- If you notice a mistake while writing, fix it silently in the text you output. Never describe the fix.
- Counting is silent. Do the check in your thinking, never in the answer.

LANGUAGE
- Learner level B1-C1: mostly short-to-medium sentences (under 25 words), plain words everywhere except the listed ones, no idioms the narrator would have to explain.
- No swearing. Never write the word "Reddit". No "Edit" or "Update" line at the end.
- Plain prose only: no headings, lists, block quotes, code fences or commentary around the story.

{title_instruction}

OUTPUT — emit exactly this and nothing else, no preamble and no closing remarks:
Title: <the title>

Story: <the story text>"""

def _normalize_term(text):
    """Lowercase a word or phrase and keep only its letters and spaces."""
    cleaned = re.sub(r"[^a-z0-9 ]+", " ", str(text).lower())
    return re.sub(r"\s+", " ", cleaned).strip()

# Cheap suffix stripping so an inflected form still matches its dictionary entry.
# Deliberately conservative: it only tries the endings English actually adds to
# these words, and never strips below 3 characters.
_SUFFIXES = ("ingly", "edly", "ing", "ers", "er", "ies", "ied", "es", "ed", "ly", "s")

# Irregular forms no suffix stripper can reach. The prompt explicitly lets the
# model inflect a listed word, so "run" has to keep "ran" bolded or the
# sanitizer would delete a perfectly good highlight. Every value is a real
# tuple -- a bare ("children") would be a string and update() would splatter
# single characters into the allowed set.
_IRREGULAR = {
    "run": ("ran", "running"), "go": ("went", "gone", "going"), "eat": ("ate", "eaten"),
    "write": ("wrote", "written"), "take": ("took", "taken"), "give": ("gave", "given"),
    "see": ("saw", "seen"), "come": ("came", "coming"), "do": ("did", "done", "doing"),
    "make": ("made", "making"), "say": ("said", "saying"), "tell": ("told", "telling"),
    "get": ("got", "gotten"), "find": ("found", "finding"), "think": ("thought", "thinking"),
    "bring": ("brought",), "hold": ("held",), "leave": ("left",), "keep": ("kept",),
    "feel": ("felt",), "sleep": ("slept",), "lose": ("lost",), "meet": ("met",),
    "pay": ("paid",), "put": ("putting",), "read": ("reading",), "sit": ("sat", "sitting"),
    "sell": ("sold",), "send": ("sent",), "set": ("setting",), "stand": ("stood",),
    "understand": ("understood",), "win": ("won",), "wear": ("wore", "worn"),
    "choose": ("chose", "chosen"), "drive": ("drove", "driven"), "fall": ("fell", "fallen"),
    "draw": ("drew", "drawn"), "grow": ("grew", "grown"), "know": ("knew", "known"),
    "speak": ("spoke", "spoken"), "throw": ("threw", "thrown"), "break": ("broke", "broken"),
    "become": ("became",), "begin": ("began", "begun"), "bite": ("bit", "bitten"),
    "blow": ("blew", "blown"), "build": ("built",), "buy": ("bought",), "catch": ("caught",),
    "cut": ("cutting",), "deal": ("dealt",), "dig": ("dug",), "feed": ("fed",),
    "fight": ("fought",), "fly": ("flew", "flown"), "forget": ("forgot", "forgotten"),
    "freeze": ("froze", "frozen"), "hide": ("hid", "hidden"), "hurt": ("hurt",),
    "lead": ("led",), "lend": ("lent",), "lie": ("lay", "lain"), "light": ("lit",),
    "ride": ("rode", "ridden"), "ring": ("rang", "rung"), "rise": ("rose", "risen"),
    "shake": ("shook", "shaken"), "shrink": ("shrank",), "sink": ("sank", "sunk"),
    "spread": ("spread",), "steal": ("stole", "stolen"), "swim": ("swam", "swum"),
    "teach": ("taught",), "tear": ("tore", "torn"), "wake": ("woke", "woken"),
    "good": ("better", "best"), "bad": ("worse", "worst"), "many": ("more", "most"),
    "much": ("more", "most"), "little": ("less", "least"), "far": ("farther", "further"),
    "well": ("better", "best"), "badly": ("worse", "worst"), "child": ("children",),
    "foot": ("feet",), "tooth": ("teeth",), "person": ("people",), "man": ("men",),
    "woman": ("women",), "life": ("lives",), "wife": ("wives",), "knife": ("knives",),
    "leaf": ("leaves",), "wolf": ("wolves",), "half": ("halves",), "thief": ("thieves",),
}

# Endings the model may legitimately add to a listed word. Used to *widen* the
# allowed set (over-permitting only means a stray bold survives; under-
# permitting would delete a correct highlight), so it is deliberately generous.
_DERIVATIONS = ("s", "es", "ed", "d", "ing", "ly", "er", "est", "ies", "ied")

def _variants_of(term):
    """The term itself plus a few stripped forms, for bold-matching."""
    seen = {term}
    if not term:
        return seen
    for suffix in _SUFFIXES:
        if not term.endswith(suffix):
            continue
        base = term[: -len(suffix)]
        # "ies"/"ied" lose a character, not just a suffix: "tried" -> "try".
        if suffix in ("ies", "ied"):
            if base.endswith("i") and len(base) >= 3:
                seen.add(base[:-1] + "y")
            continue
        if len(base) < 3:
            continue
        seen.add(base)
        # doubled consonant: "running" -> "run"
        if len(base) > 3 and base[-1] == base[-2]:
            seen.add(base[:-1])
    # Irregular forms, plus the reverse direction ("ran" should match "run").
    for base, forms in _IRREGULAR.items():
        if term == base or term in forms:
            seen.add(base)
            seen.update(forms)
    return seen

def _allowed_bold_terms(words):
    """Every single token that may legitimately be bolded for this word list."""
    allowed = set()
    for word in words or []:
        normalized = _normalize_term(word)
        if not normalized:
            continue
        allowed.add(normalized)
        bases = _variants_of(normalized)
        # A phrase entry also licenses any single word of it, since the prompt
        # lets the model inflect one word of a phrase.
        for part in normalized.split():
            bases |= _variants_of(part)
        for base in bases:
            allowed.add(base)
            # A listed word may be the one that got inflected in the story, so
            # accept the derived forms of every form we know about.
            for ending in _DERIVATIONS:
                allowed.add(base + ending)
                if base.endswith("e"):
                    allowed.add(base[:-1] + ending)
            # consonant + y: "try" -> "tried", "tries", "trier".
            if base.endswith("y") and len(base) >= 3:
                stem = base[:-1]
                for ending in ("ied", "ies", "ier", "iest", "ily", "ying"):
                    allowed.add(stem + ending)
    return allowed

def _strip_stray_bold(content, words):
    """Un-bold anything that is not a listed word or a variant of one.

    Models over-apply bold even when told not to (bolding "use", "already",
    "not"), which reads as sloppy in the rendered story. The DB is the source of
    truth for what a word looked like, so the fix is applied before persisting.
    Deliberately one-sided: it only ever removes bold, never adds it, so a word
    the model failed to highlight stays un-highlighted rather than being mangled.
    """
    allowed = _allowed_bold_terms(words)
    if not allowed:
        return content

    def fix(match):
        inner = match.group(1)
        # A bolded multi-word span is left alone: it is almost certainly a listed
        # phrase, and guessing which word inside it is the target is not worth it.
        if " " in inner.strip():
            return match.group(0)
        return f"**{inner}**" if _normalize_term(inner) in allowed else inner

    return re.sub(r"\*\*(.+?)\*\*", fix, content, flags=re.S)


def _read_story_stream(response, started, model, story_id, on_delta):
    """Consume an OpenRouter SSE body and reassemble the final payload.

    Returns ``(data, None)`` on success or ``(None, error_message)`` when the
    user cancelled or the overall deadline was hit. ``data`` is shaped like a
    non-streaming chat completion so the caller can treat both the same.

    Providers that ignore ``"stream": true`` and answer with a single plain
    JSON body are handled too: the raw body is parsed and emitted as one delta.
    """
    content_parts = []
    reasoning_parts = []
    usage = {}
    provider_responses = []
    seen_model = None
    finish_reason = None
    native_finish_reason = None
    native_tokens_reasoning = None
    saw_sse = False
    done = False
    plain_lines = []

    for raw_line in response.iter_lines():
        # Decode explicitly as UTF-8: "text/event-stream" carries no charset,
        # so requests would otherwise fall back to ISO-8859-1 and mangle every
        # em dash and curly quote in the story.
        if isinstance(raw_line, bytes):
            raw_line = raw_line.decode("utf-8", "replace")
        if not raw_line or not raw_line.strip():
            continue

        line = raw_line.lstrip()
        if not line.startswith("data:"):
            # Not SSE — keep the raw line for the whole-body fallback below.
            if not saw_sse:
                plain_lines.append(line)
            continue

        if not saw_sse:
            saw_sse = True
            plain_lines = []

        body = line[len("data:"):]
        while body:
            body = body.strip()
            if not body:
                break
            if body == "[DONE]":
                # Do NOT break out of the read loop: with
                # stream_options.include_usage the final usage chunk arrives
                # *after* this sentinel. Keep reading so token/cost accounting
                # stays intact; the read timeout and the deadline below bound
                # the wait if a provider forgets to close the stream.
                done = True
                break

            # Bail out as early as possible once cancelled or out of time.
            if _story_cancelled(story_id):
                return None, _STORY_ERR_CANCELLED
            if time.monotonic() - started > _STORY_MAX_ELAPSED:
                return None, _STORY_ERR_DEADLINE

            # raw_decode tells us exactly where this record ended, so we can
            # tell a well-formed record apart from several glued onto one line
            # without ever splitting inside the story text.
            try:
                chunk, consumed = _JSON_DECODER.raw_decode(body)
            except ValueError:
                nxt = body.find("data:")
                if nxt == -1:
                    break       # malformed record, nothing to resync to
                body = body[nxt + len("data:"):]
                continue
            if not isinstance(chunk, dict):
                break

            if chunk.get("model"):
                seen_model = chunk["model"]
            if chunk.get("usage"):
                usage = chunk["usage"]
            if chunk.get("provider_responses"):
                provider_responses = chunk["provider_responses"]
            if chunk.get("finish_reason"):
                finish_reason = chunk["finish_reason"]
            if chunk.get("native_finish_reason"):
                native_finish_reason = chunk["native_finish_reason"]
            if chunk.get("native_tokens_reasoning") is not None:
                native_tokens_reasoning = chunk["native_tokens_reasoning"]

            for choice in chunk.get("choices") or []:
                if done:
                    break
                # Streamed chunks use "delta"; some providers send a full "message".
                piece = choice.get("delta") or choice.get("message") or {}
                text = piece.get("content")
                if text:
                    content_parts.append(text)
                    if on_delta:
                        on_delta(text)
                reasoning = piece.get("reasoning")
                if reasoning:
                    reasoning_parts.append(reasoning)

            body = body[consumed:]
            nxt = body.find("data:")
            if nxt == -1:
                break       # that was the last record on this line
            body = body[nxt + len("data:"):]

    if not saw_sse and plain_lines:
        # Non-streaming provider: parse the whole body at once.
        try:
            payload = json.loads("\n".join(plain_lines))
        except ValueError:
            payload = None
        if isinstance(payload, dict):
            seen_model = payload.get("model") or seen_model
            usage = payload.get("usage") or usage
            provider_responses = payload.get("provider_responses") or provider_responses
            finish_reason = payload.get("finish_reason") or payload.get("native_finish_reason") or finish_reason
            native_finish_reason = payload.get("native_finish_reason") or native_finish_reason
            if payload.get("native_tokens_reasoning") is not None:
                native_tokens_reasoning = payload["native_tokens_reasoning"]
            for choice in payload.get("choices") or []:
                msg = choice.get("message") or choice.get("delta") or {}
                if msg.get("content"):
                    content_parts.append(msg["content"])
                if msg.get("reasoning"):
                    reasoning_parts.append(msg["reasoning"])

    # Last chance to honour a cancel that landed while we were reading.
    if _story_cancelled(story_id):
        return None, _STORY_ERR_CANCELLED

    if not saw_sse and content_parts and on_delta:
        # Emit the whole (non-streamed) story as a single live update.
        on_delta("".join(content_parts))

    return {
        "model": seen_model or model,
        "choices": [{
            "message": {
                "content": "".join(content_parts),
                "reasoning": "".join(reasoning_parts),
            }
        }],
        "usage": usage,
        "provider_responses": provider_responses,
        "finish_reason": finish_reason,
        "native_finish_reason": native_finish_reason,
        "native_tokens_reasoning": native_tokens_reasoning,
    }, None


def sync_generate_story(words, title=None, model=None, provider_tag=None, story_id=None, on_delta=None):
    if not OPENROUTER_API_KEY:
        return {"ok": False, "error": "OpenRouter API key not set. Add OPENROUTER_API_KEY to your .env file."}
    if not words:
        return {"ok": False, "error": "No words were provided to build a story."}

    model = model or OPENROUTER_MODEL
    prompt = _build_prompt(words, title)

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "Vocabulary Story Builder",
    }

    # Provider routing: pin to a specific provider or use default routing.
    if provider_tag:
        provider_config = {"order": [provider_tag], "allow_fallbacks": True}
    else:
        provider_config = {
            "sort": "throughput",
        }

    payload = {
        "model": model,
        "provider": provider_config,
        "messages": [
            {
                "role": "system",
                "content": "You are a creative writing assistant that weaves words into engaging, coherent stories.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
        # Output budget. It is the only length bound: the prompt deliberately
        # sets no target length. A story cut off here is still published, as
        # status 'truncated' (see _STORY_MIN_VISIBLE_TOKENS).
        "max_tokens": 20000,
        # Stream so the request can be cancelled mid-generation and the UI can
        # render the story as it arrives. include_usage keeps the token/cost
        # accounting identical to the non-streaming path.
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    # Auto-detect reasoning config from model metadata.
    reasoning_config = _get_reasoning_config(model)
    if reasoning_config:
        payload["reasoning"] = reasoning_config

    def post():
        return requests.post(
            OPENROUTER_BASE_URL,
            headers=headers,
            json=payload,
            stream=True,
            timeout=(_STORY_CONNECT_TIMEOUT, _STORY_READ_TIMEOUT),
        )

    started = time.monotonic()
    # Register before the request so a cancel arriving during the connect phase
    # is not lost; _set_story_response closes the socket once it exists.
    _register_story_control(story_id)
    try:
        id_part = f"id={story_id} " if story_id is not None else ""
        print(
            f"story: {id_part}requesting model '{model}' for {len(words)} words "
            f"| reason_cfg={_reason_cfg_label(reasoning_config)}",
            flush=True,
        )
        try:
            response = post()
            _set_story_response(story_id, response)
        except requests.exceptions.ConnectionError:
            # May be caused by cancel_story_request closing the socket.
            if _story_cancelled(story_id):
                return {"ok": False, "error": _STORY_ERR_CANCELLED}
            return {"ok": False, "error": _STORY_ERR_DISCONNECTED}
        except requests.exceptions.Timeout:
            return {"ok": False, "error": _STORY_ERR_TIMEOUT}
        except requests.exceptions.RequestException as e:
            return {"ok": False, "error": f"Error generating story: {e}"}

        if response.status_code != 200:
            detail = response.text[:400].replace("\n", " ")
            _safe_close(response)
            print(f"[story] OpenRouter {response.status_code} for model='{model}' for {len(words)} words: {detail}", file=sys.stderr, flush=True)
            should_retry = (
                model != FALLBACK_MODEL and (
                    (response.status_code == 404 and "unavailable for free" in detail.lower())
                    or response.status_code == 429
                    or response.status_code >= 500
                )
            )
            if should_retry:
                print(f"[story] retry fallback {FALLBACK_MODEL} after {response.status_code} {model} for {len(words)} words", file=sys.stderr, flush=True)
                payload["model"] = FALLBACK_MODEL
                try:
                    # No deltas were emitted by the failed attempt, so replaying
                    # the same on_delta against the fallback model is safe.
                    response = post()
                    _set_story_response(story_id, response)
                except requests.exceptions.RequestException as e:
                    return {"ok": False, "error": f"Error generating story: {e}"}
                if response.status_code != 200:
                    detail = response.text[:400].replace("\n", " ")
                    _safe_close(response)
                    return {"ok": False, "error": f"Error generating story ({response.status_code}): {detail}"}
                model = FALLBACK_MODEL
            else:
                return {"ok": False, "error": f"Error generating story ({response.status_code}): {detail}"}

        # The deadline is enforced inside this loop, chunk by chunk.
        try:
            data, stream_error = _read_story_stream(response, started, model, story_id, on_delta)
        except requests.exceptions.Timeout:
            if _story_cancelled(story_id):
                return {"ok": False, "error": _STORY_ERR_CANCELLED}
            return {"ok": False, "error": _STORY_ERR_TIMEOUT}
        except requests.exceptions.ConnectionError:
            if _story_cancelled(story_id):
                return {"ok": False, "error": _STORY_ERR_CANCELLED}
            return {"ok": False, "error": _STORY_ERR_DISCONNECTED}
        finally:
            _safe_close(response)
        if stream_error:
            return {"ok": False, "error": stream_error}

        elapsed = time.monotonic() - started
        msg = (data.get("choices") or [{}])[0].get("message") or {}
        # Content only. A reasoning-only response used to be stored AS the story
        # (the `or msg.get("reasoning")` fallback), which put the model's
        # scratchpad in the database and fed it to TTS. Reasoning is a separate
        # channel and must never become content.
        content = msg.get("content") or ""
        if not content.strip():
            reason = ("the model returned only reasoning and no story text"
                      if msg.get("reasoning") else "the model returned an empty response")
            return {
                "ok": False,
                "error": (
                    f"Error generating story: {reason}. Try again, or use a "
                    f"different model."
                ),
            }

        usage = data.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens") or 0
        completion_tokens = usage.get("completion_tokens") or 0
        reasoning_tokens = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
        if not reasoning_tokens and msg.get("reasoning"):
            # Provider sent no usage block. Approximate from the reasoning text
            # so the log line still separates visible from hidden tokens.
            reasoning_tokens = max(1, len(msg["reasoning"]) // 4)
            if not completion_tokens:
                completion_tokens = reasoning_tokens
        visible_tokens = max(0, completion_tokens - reasoning_tokens)
        total_tokens = usage.get("total_tokens") or (prompt_tokens + completion_tokens)
        cost = usage.get("cost")
        if cost is None:
            pricing = _get_pricing().get(model) or {}
            cost = prompt_tokens * pricing.get("prompt", 0) + completion_tokens * pricing.get("completion", 0)
        provider_responses = data.get("provider_responses") or []
        provider_name = provider_responses[0].get("provider_name", "?") if provider_responses else "?"
        finish = data.get("finish_reason") or data.get("native_finish_reason") or "?"
        # "length" means the model ran into max_tokens and stopped mid-sentence.
        # The prose is still worth showing, so it is reported to the caller
        # rather than treated as a failure (main.py stores it as a 'truncated'
        # story the user can retry). The exception is a reasoning model that
        # spent the whole budget thinking and emitted almost no visible text:
        # there is nothing worth keeping, so that fails outright.
        truncated = finish == "length"
        if truncated and visible_tokens < _STORY_MIN_VISIBLE_TOKENS:
            return {
                "ok": False,
                "error": (
                    "Error generating story: the model used the whole token budget "
                    "reasoning and left almost no story text. Try a different model "
                    "or generate with fewer words."
                ),
            }
        actual_model = data.get("model", model)
        chars = len(content)
        native_reason = data.get("native_tokens_reasoning")
        native_part = f" native_reason={native_reason}" if native_reason is not None else ""
        id_part = f"id={story_id} " if story_id is not None else ""
        print(
            f"[story] {id_part}ok | model={actual_model} via {provider_name} | words={len(words)} | "
            f"in={prompt_tokens} out={completion_tokens} (reason={reasoning_tokens}, visible={visible_tokens}) "
            f"total={total_tokens}{native_part} | reason_cfg={_reason_cfg_label(reasoning_config)} | "
            f"{elapsed:.1f}s | {finish} | ${cost:.4f} | chars={chars}",
            flush=True,
        )

        return {
            "ok": True,
            "content": _strip_stray_bold(content, words),
            "elapsed": elapsed,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "reasoning_tokens": reasoning_tokens,
            # True when finish_reason == "length": the prose stops mid-sentence
            # because it hit max_tokens. main.py publishes it anyway, as a
            # 'truncated' story, so the user can read it and retry.
            "truncated": truncated,
            "cost": round(cost, 6),
        }
    except Exception as e:
        return {"ok": False, "error": f"Error generating story: {e}"}
    finally:
        _release_story_control(story_id)


# --- Word-lookup (meanings / sentences / synonyms) ----------------------------
# A single batched request returns all three fields as JSON, which is faster
# and cheaper than three parallel calls (one round-trip, smaller total tokens)
# and keeps the three fields consistent with each other. Public functions
# return strings so they remain drop-in replacements for the Gemini/Groq agents.

def _build_meanings_prompt(word: str) -> str:
    return (
        f"For the English word or short phrase '{word}', produce a JSON object "
        f"with exactly these three keys:\n"
        f'  - "meaning": concise, natural definitions. Consider the more commons definitions. Use Markdown for emphasis '
        f"(e.g. **bold**); do NOT use bullet points or numbered lists.\n"
        f'  - "sentences": exactly 5 example sentences, one per line, with the '
        f"word/phrase highlighted in **bold** on every occurrence.\n"
        f'  - "synonyms": a comma-separated list of 5 synonyms. If the word has '
        f"multiple senses, cover the most common one.\n"
        f"Return ONLY the JSON object. No prose, no code fences, no preamble."
    )


def _parse_meanings_json(content: str, word: str) -> dict:
    """Best-effort parse: strip ```json fences, then json.loads. Raises on failure."""
    text = (content or "").strip()
    # Strip ```json ... ``` or ``` ... ``` fences if the model added them despite
    # the instruction. Handle both opening and closing on their own lines.
    if text.startswith("```"):
        # Drop the first line (``` or ```json)
        first_nl = text.find("\n")
        if first_nl != -1:
            text = text[first_nl + 1 :]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    # Some models wrap the JSON in trailing prose. Find the outermost braces.
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]
    data = json.loads(text, strict=False)
    if not isinstance(data, dict):
        raise ValueError("model did not return a JSON object")
    meaning = str(data.get("meaning") or "").strip()
    sentences = str(data.get("sentences") or "").strip()
    synonyms = str(data.get("synonyms") or "").strip()
    if not (meaning or sentences or synonyms):
        raise ValueError("model returned an empty JSON object")
    return {"meaning": meaning, "sentences": sentences, "synonyms": synonyms}


def _sync_lookup_word(word: str, model: str | None = None) -> dict:
    """One-shot lookup that fetches meaning, sentences and synonyms together.

    Returns ``{"ok": True, "meaning": ..., "sentences": ..., "synonyms": ...,
    "model": ..., "elapsed": ..., "cost": ...}`` on success, or
    ``{"ok": False, "error": "..."}`` on any failure. The three string fields
    may be empty if the model didn't supply them — the caller decides whether
    to surface that as an error.

    When called concurrently for the same (word, model) — main.py fans out
    via ``asyncio.gather`` to three ``sync_get_*`` functions — the second and
    third callers wait on the in-flight result instead of issuing duplicate
    HTTP requests. This is the "single batched prompt" promise: one round-trip
    per word regardless of how many fields the caller asks for.
    """
    if not word:
        return {"ok": False, "error": "No word was provided."}
    if not OPENROUTER_API_KEY:
        return {"ok": False, "error": "OpenRouter API key not set. Add OPENROUTER_API_KEY to your .env file."}

    effective_model = model or FALLBACK_MODEL
    key = (word, effective_model)
    entry = _lookup_inflight.get(key)
    if entry is not None:
        # Another caller is already fetching — wait for their result.
        entry["event"].wait()
        return entry["result"]

    # We're the first caller for this (word, model). Set up a slot for any
    # racing callers, then run the real fetch.
    event = threading.Event()
    _lookup_inflight[key] = {"event": event, "result": None}
    try:
        result = _do_lookup_request(word, effective_model)
        _lookup_inflight[key]["result"] = result
        return result
    finally:
        event.set()
        # Tiny grace period so late-arriving callers find the slot; then drop it.
        def _cleanup(k=key):
            _lookup_inflight.pop(k, None)
        threading.Timer(0.5, _cleanup).start()


def _do_lookup_request(word: str, model: str) -> dict:
    """Issue the actual HTTP request. Called by ``_sync_lookup_word`` after
    de-dup; ``word`` and ``model`` are guaranteed non-empty and the API key
    is verified. Returns the standard result dict.
    """
    prompt = _build_meanings_prompt(word)

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "Vocabulary Lookup",
    }
    # `response_format` is honoured by OpenRouter-compatible models (most
    # instruct-tuned ones). We send it but don't require it — if the model
    # ignores it, _parse_meanings_json still salvages JSON from prose.
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a dictionary assistant. You always reply with a "
                    "single JSON object and nothing else."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": OPENROUTER_MEANINGS_TEMPERATURE,
        "response_format": {"type": "json_object"},
    }

    # Disable/reduce reasoning for reasoning models.
    reasoning_config = _get_reasoning_config(model)
    if reasoning_config:
        payload["reasoning"] = reasoning_config

    started = time.monotonic()
    try:
        reason_label = _reason_cfg_label(reasoning_config)
        if model == "openrouter/free":
            print(f"meanings: OpenRouter free-tier router for '{word}' | reason_cfg={reason_label}", flush=True)
        elif model == "openrouter/auto":
            print(f"meanings: OpenRouter auto router for '{word}' | reason_cfg={reason_label}", flush=True)
        else:
            print(f"meanings: requesting OpenRouter model '{model}' for '{word}' | reason_cfg={reason_label}", flush=True)
        try:
            response = requests.post(
                OPENROUTER_BASE_URL, headers=headers, json=payload, timeout=(10, 60)
            )
        except requests.exceptions.Timeout:
            return {"ok": False, "error": (
                "Error fetching lookup: request timed out. The model may be slow "
                "or rate-limited. Try a different model.")}
        except requests.exceptions.RequestException as e:
            return {"ok": False, "error": f"Error fetching lookup: {e}"}

        if response.status_code != 200:
            detail = response.text[:400].replace("\n", " ")
            print(f"[meanings] OpenRouter {response.status_code} for model='{model}' word='{word}': {detail}", file=sys.stderr, flush=True)
            should_retry = (
                model != FALLBACK_MODEL and (
                    (response.status_code == 404 and "unavailable for free" in detail.lower())
                    or response.status_code == 429
                    or response.status_code >= 500
                )
            )
            if should_retry:
                print(f"[meanings] retry fallback {FALLBACK_MODEL} after {response.status_code} {model} for '{word}'", file=sys.stderr, flush=True)
                payload["model"] = FALLBACK_MODEL
                try:
                    response = requests.post(
                        OPENROUTER_BASE_URL, headers=headers, json=payload, timeout=(10, 60)
                    )
                except requests.exceptions.RequestException as e:
                    return {"ok": False, "error": f"Error fetching lookup: {e}"}
                if response.status_code != 200:
                    detail = response.text[:400].replace("\n", " ")
                    print(f"[meanings] retry also failed {response.status_code}: {detail}", file=sys.stderr, flush=True)
                    return {"ok": False, "error": f"Error fetching lookup ({response.status_code}): {detail}"}
                model = FALLBACK_MODEL
            else:
                return {"ok": False, "error": f"Error fetching lookup ({response.status_code}): {detail}"}

        try:
            data = response.json()
        except ValueError:
            return {"ok": False, "error": f"Error fetching lookup: invalid JSON response ({response.status_code})."}

        elapsed = time.monotonic() - started
        msg = (data.get("choices") or [{}])[0].get("message") or {}
        content = msg.get("content") or msg.get("reasoning") or ""
        if not content.strip():
            return {"ok": False, "error": "Error fetching lookup: model returned an empty response (it may have produced only reasoning). Try a different model."}

        try:
            parsed = _parse_meanings_json(content, word)
        except (ValueError, json.JSONDecodeError) as e:
            return {"ok": False, "error": (
                f"Error fetching lookup: model did not return valid JSON ({e}). "
                f"Try a different model."), "raw": content[:400]}

        usage = data.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens") or 0
        completion_tokens = usage.get("completion_tokens") or 0
        reasoning_tokens = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
        visible_tokens = max(0, completion_tokens - reasoning_tokens)
        total_tokens = usage.get("total_tokens") or (prompt_tokens + completion_tokens)
        cost = usage.get("cost")
        if cost is None:
            pricing = _get_pricing().get(model) or {}
            cost = prompt_tokens * pricing.get("prompt", 0) + completion_tokens * pricing.get("completion", 0)
        provider_responses = data.get("provider_responses") or []
        provider_name = provider_responses[0].get("provider_name", "?") if provider_responses else "?"
        finish = data.get("finish_reason") or data.get("native_finish_reason") or "?"
        actual_model = data.get("model", model)
        native_reason = data.get("native_tokens_reasoning")
        native_part = f" native_reason={native_reason}" if native_reason is not None else ""
        print(
            f"[meanings] {actual_model} via {provider_name} | word='{word}' | "
            f"in={prompt_tokens} out={completion_tokens} (reason={reasoning_tokens}, visible={visible_tokens}) "
            f"total={total_tokens}{native_part} | reason_cfg={_reason_cfg_label(reasoning_config)} | "
            f"{elapsed:.1f}s | {finish} | ${cost:.4f}",
            flush=True,
        )

        return {
            "ok": True,
            "meaning": parsed["meaning"],
            "sentences": parsed["sentences"],
            "synonyms": parsed["synonyms"],
            "model": model,
            "elapsed": elapsed,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost": round(cost, 6),
        }
    except Exception as e:
        return {"ok": False, "error": f"Error fetching lookup: {e}"}


# --- Public functions (drop-in replacements for gemini_agent / groq_agent) ---

def sync_get_meanings_of(word: str, model: str | None = None) -> str:
    r = _sync_lookup_word(word, model)
    if r.get("ok"):
        return r["meaning"] or "(no meaning returned)"
    return f"Error fetching meaning: {r.get('error')}"


def sync_get_sentences_with(word: str, model: str | None = None) -> str:
    r = _sync_lookup_word(word, model)
    if r.get("ok"):
        return r["sentences"] or "(no sentences returned)"
    return f"Error fetching sentences: {r.get('error')}"


def sync_get_synonyms_of(word: str, model: str | None = None) -> str:
    r = _sync_lookup_word(word, model)
    if r.get("ok"):
        return r["synonyms"] or "(no synonyms returned)"
    return f"Error fetching synonyms: {r.get('error')}"
