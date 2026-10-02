import os
import re
import sys
import json
import time
import threading
import requests
import random
from functools import lru_cache
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


# Reasoning modes a preset can ask for. "auto" is the shipped behaviour and the
# default; the rest exist because the auto rules are invisible from the UI --
# you cannot tell from the outside whether a model reasons at all, or how hard.
#
# The first three are names, the rest are effort levels. A level is a REQUEST:
# a model only supports a subset (grok-4.7 has no "medium", for one), and
# mandatory-reasoning models have no "none" at all. _resolve_effort() clamps to
# what the chosen model actually supports and use_reasoning_status() reports
# the substitution.
REASONING_MODES = ("auto", "off", "minimal")
# Canonical cheap-to-expensive order, used to pick the nearest effort a model
# supports. Kept here rather than per model because each model's own
# supported_efforts list is a subset in an arbitrary order.
EFFORT_ORDER = ("none", "minimal", "low", "medium", "high", "xhigh")


def all_reasoning_efforts():
    """Every effort level any catalogue model supports, cheap to expensive.

    The Lab's dropdown is built from this so the levels offered are the ones
    that exist somewhere, instead of a hardcoded list that would drift from the
    catalogue. Which of them the *chosen* model honours is resolved at
    generation time.
    """
    models, _ = _fetch_models()
    seen, extra = set(), []
    for m in models or []:
        for effort in ((m.get("reasoning") or {}).get("supported_efforts") or []):
            if effort in EFFORT_ORDER:
                seen.add(effort)
            elif effort not in extra:
                # Levels outside the canonical order still exist ("max"), and a
                # model that has one should be able to ask for it.
                extra.append(effort)
    return [e for e in EFFORT_ORDER if e in seen] + extra


def _resolve_effort(requested, supported):
    """Clamp a requested effort to what the model supports.

    Returns ``(effort, substituted)``. The clamp prefers a level at or below
    what was asked for, so an unsupported "high" never silently costs more than
    requested; if every supported level is dearer, the cheapest is used and the
    caller reports it.
    """
    supported = list(supported or [])
    if not supported:
        return None, True
    if requested in supported:
        return requested, False
    def rank(name):
        # An unrecognised level (deepseek calls its most expensive "max") sorts
        # ABOVE every known one, so it is never picked as a cheaper substitute.
        return EFFORT_ORDER.index(name) if name in EFFORT_ORDER else len(EFFORT_ORDER)

    if requested in supported:
        return requested, False
    want = rank(requested)
    at_or_below = [e for e in supported if rank(e) <= want]
    if at_or_below:
        return max(at_or_below, key=rank), True
    return min(supported, key=rank), True



def _get_reasoning_config(model_id: str, mode: str = "auto", max_tokens: int = None):
    """Return the reasoning config dict for a model, or None if not a reasoning model.

    mode is "auto", "off", "minimal", or an effort level ("low", "medium", ...).
    Anything else falls back to "auto".

    `max_tokens` is an optional CAP on reasoning tokens, the counterpart to the
    story's output budget: OpenRouter takes `reasoning: {max_tokens: N}` and the
    model stops thinking when it reaches N and produces its answer. 0/None means
    no cap. It is merged into whichever branch returns below rather than built
    here, so every branch honours it. Caveat worth keeping in mind: the model
    catalogue does not advertise support for this sub-field, so a provider may
    ignore it -- which is why the health line still reports reason/vis.
    - No reasoning field -> non-reasoning model, return None
    - mandatory=False -> disable reasoning (effort: "none")
    - mandatory=True -> can't disable, use lowest supported effort
    - Meta-routers (openrouter/free, openrouter/auto) -> disable reasoning (safe default)

    A caveat that belongs in the UI and not only here: "off" is a REQUEST, not
    a guarantee. On a mandatory-reasoning model (grok-4.7, glm-5.3-flash, both
    reasoning.mandatory: true with no "none" effort) there is no off switch, so
    the lowest supported effort is requested instead of an "off" the provider
    would ignore. use_reasoning_status() reports that substitution so the
    preview can say so rather than quietly lying.
    """
    model = _reasoning_model(model_id)
    # A mode is either a named mode or an effort level the catalogue knows --
    # including levels outside the canonical order, like deepseek's "max".
    # Anything else (an old preset value, a typo) falls back to "auto" rather
    # than being sent as-is, because a misread level would be clamped to the
    # most expensive thing the model offers.
    if mode not in REASONING_MODES and mode not in all_reasoning_efforts():
        mode = "auto"

    if model_id in ("openrouter/free", "openrouter/auto"):
        # Meta-routers pick a backend for you, so there is nothing meaningful to
        # hold on or off; reasoning stays off whatever was asked.
        return _with_reason_cap({"effort": "none", "exclude": True}, max_tokens)

    if model is None:
        # Unknown model (custom id, catalogue fetch failed). "off" is still
        # worth sending: a provider that honours it will use it. An explicit
        # effort goes out as requested -- there is nothing to clamp it to.
        if mode == "off":
            return _with_reason_cap({"effort": "none", "exclude": True}, max_tokens)
        return _with_reason_cap({"effort": mode, "exclude": True}, max_tokens) if mode in EFFORT_ORDER else None

    reasoning = model.get("reasoning")
    if not reasoning:
        # Not a reasoning model. Sending "off" would be noise, and "minimal"
        # has nothing to minimise. A cap is not sent either: there is nothing
        # reasoning to cap, and sending it would imply there is.
        return None

    efforts = reasoning.get("supported_efforts") or ["low"]
    mandatory = bool(reasoning.get("mandatory", False))

    if mode == "off" and not mandatory:
        return _with_reason_cap({"effort": "none", "exclude": True}, max_tokens)
    if mode == "minimal":
        lowest, _ = _resolve_effort("minimal", efforts)
        return _with_reason_cap({"effort": lowest or efforts[-1], "exclude": True}, max_tokens)
    if mode not in REASONING_MODES:
        # An explicit effort level. Clamped to what this model supports.
        effort, _ = _resolve_effort(mode, efforts)
        return _with_reason_cap({"effort": effort, "exclude": True}, max_tokens)
    # mandatory: cannot be turned off, so ask for the cheapest thing available.
    if mandatory:
        lowest, _ = _resolve_effort("minimal", efforts)
        return _with_reason_cap({"effort": lowest or efforts[-1], "exclude": True}, max_tokens)
    return _with_reason_cap({"effort": "none", "exclude": True}, max_tokens)


def _with_reason_cap(config, max_tokens):
    """Attach a reasoning-token cap to a reasoning config.

    Every branch of _get_reasoning_config() returns through here, so the cap can
    never be applied in one mode and silently dropped in another -- which is the
    same class of bug as a gen_kwargs key with no matching parameter: it fails
    at request time, on one path only.
    """
    if not config:
        return config
    try:
        cap = int(max_tokens or 0)
    except (TypeError, ValueError):
        return config
    if cap > 0:
        config["max_tokens"] = cap
    return config


def _reasoning_model(model_id: str):
    """The catalogue entry for a model, or None if it is not known."""
    models, _ = _fetch_models()
    return next((m for m in models if m["id"] == model_id), None)


def use_reasoning_status(model_id: str, mode: str = "auto", max_tokens: int = None):
    """What will actually be sent for (model, mode), plus a note when the
    request cannot be honoured. Used by the pre-generation preview, which must
    not claim reasoning is off when the model will think anyway."""
    config = _get_reasoning_config(model_id, mode, max_tokens)
    note = ""
    # Whether the note is a limitation the user should SEE, or just reference
    # detail. "This model does not reason, so nothing is sent" is information;
    # "there is no off switch, expect several times the time and tokens" is a
    # cost the user is about to pay and cannot avoid. The UI must not have to
    # decide that from the wording, so it is decided here.
    warn = False
    model = _reasoning_model(model_id)
    reasoning_meta = (model or {}).get("reasoning") or {}
    supported = reasoning_meta.get("supported_efforts") or []
    mandatory = bool(reasoning_meta.get("mandatory"))
    if mode in all_reasoning_efforts() and supported and mode not in supported:
        chosen, _ = _resolve_effort(mode, supported)
        note = (f"This model does not offer “{mode}”. The nearest it supports "
                f"(“{chosen}”) was requested instead.")
        # Deliberately NOT a warning. The user chose this level; it is simply not
        # on the menu for this model, and substituting the nearest one is the
        # expected outcome of picking a model rather than a surprise. It stays in
        # the tooltip with the other reference detail. Only a cost the user did
        # not choose AND cannot avoid earns an inline row (the mandatory branches
        # below), so the amber line keeps its meaning.
    elif mode == "off" and mandatory:
        note = ("This model always reasons, so 'off' cannot be honoured — the "
                "cheapest effort it supports is requested instead.")
        warn = True
    elif mandatory:
        # The case that was silent, and the one that matters most: "auto" on a
        # mandatory-reasoning model resolves to its cheapest effort, so the model
        # WILL think and nothing said so. The only clue was a small
        # "auto -> low" in the config grid. This is the configuration where the
        # user is about to pay for reasoning they cannot avoid, so it is the one
        # that gets a plain warning.
        if mode in all_reasoning_efforts():
            # They picked the level on purpose, so do not lecture them about an
            # off switch -- just say what this level costs.
            note = (f"This model always reasons. “{mode}” is one of its "
                    f"heaviest settings: expect the longest waits and the "
                    f"largest bills.")
        else:
            note = ("This model always reasons — there is no off switch. It "
                    "will think at its cheapest effort; expect several times "
                    "the time and tokens.")
        warn = True
    elif mode == "off" and model is None:
        note = "Model not in the catalogue, so 'off' is sent as a request only."
    elif config is None:
        note = "This model does not reason, so nothing is sent."
    return {
        "requested": mode if mode in REASONING_MODES or mode in all_reasoning_efforts() else "auto",
        "effective": _reason_cfg_label(config),
        "supported": supported,
        "config": config,
        "note": note,
        "warn": warn,
    }


def _reason_cfg_label(reasoning_config) -> str:
    """Compact label for request logs: none / low / not-set, plus the cap.

    The cap is in the label rather than a separate line because the two are one
    decision: "low (cap 2000)" says what a run actually allowed, where "low"
    alone would read as unlimited right next to a `reason/vis=12.1`.
    """
    if not reasoning_config:
        return "not-set"
    effort = reasoning_config.get("effort")
    label = effort if effort else "on"
    cap = reasoning_config.get("max_tokens")
    if cap:
        label += f" (cap {cap})"
    return label


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

# ---------------------------------------------------------------------------
# Prompt construction.
#
# The prompt is three parts and only the middle one is editable:
#
#   system   Fixed role plus the output contract (the "Title:"/"Story:" labels).
#            It is NOT part of the editable template, so the shape that
#            _extract_title (main.py) and stripStoryLabels (words.html) regex
#            on cannot be edited away by a user.
#   template The prose rules. Stored in the settings table and editable from
#            the UI without touching this file. Sent verbatim: no placeholders,
#            so braces in the text are just text.
#   tail     The chosen style and the word list, appended AFTER the template.
#            Words are bulk data and belong last, where recency helps the model
#            hold them. Rendering them here (rather than letting the template
#            place them) is also what keeps the list un-numbered and quoted,
#            which is what stops indices leaking out as "**scavenge** 1" and
#            stops the model working through the list in order.
# ---------------------------------------------------------------------------

_STORY_TEMPERATURE = 0.7
_STORY_MAX_TOKENS = 20000

# Print the full rendered prompt for every story request. Off by default; turn
# it on while editing the template so what is sent can be read in the log.
_DEBUG_PROMPT = os.environ.get("STORY_DEBUG_PROMPT", "").strip() not in ("", "0", "false")

_SYSTEM_ROLE = (
    "You are a creative writing assistant that weaves words into engaging, "
    "coherent stories."
)

# The output envelope, and the reason the system message exists at all. These
# two labels are load-bearing: main.py:_extract_title looks for the "Title:"
# line and words.html:stripStoryLabels looks for "Story:". Without them every
# story is titled "Untitled Story" AND the labels stay in the prose, which TTS
# then reads aloud ("Title: Barn on Fire. Story: ...") -- so this text is
# editable, but validate_prompt() refuses a version that drops either label.
_SYSTEM_CONTRACT = (
    "Reply with exactly this and nothing else, no preamble and no closing "
    "remarks:\n"
    "Title: <the title>\n"
    "\n"
    "Story: <the story text>"
)

# The whole system message, editable per preset. Exposed to the UI as
# `default_system`; the editable copy lives in the settings table.
_DEFAULT_SYSTEM = f"{_SYSTEM_ROLE}\n\n{_SYSTEM_CONTRACT}"

# The editable middle. "Reset to code default" in the UI restores exactly this,
# so it must stay the shipped prompt and not drift from what is documented in
# AGENTS.md.
#
# The title is a plain instruction in the template. It used to be a "{title}"
# placeholder that expanded to one of two branches, so a user could force a
# title from Story Setup; that is gone by request, because a placeholder in
# prose is one more thing to explain and one more thing to get wrong. The
# "Title:" label in the SYSTEM message is what the parsers need, and that is
# still enforced -- only the user-supplied-title path was removed.
# The task sentence is INSIDE the template, not prepended by build_prompt. It
# used to be a hardcoded _TASK_LINE, which meant the one sentence that framed the
# whole request was the one thing a user could not edit -- delete it from their
# preset and it still came back in the preview. Everything the server adds now is
# the tail (style line + word list), which has a real reason to be fixed: the
# list must stay un-numbered, quoted and last.
_DEFAULT_TEMPLATE = """Write one enjoyable, coherent story that reads like a real post on Reddit, and that uses every word in this list:

PLOT
- First person, casual and conversational, like telling a friend what happened. Open with a hook in the first two sentences.
- One narrator with a clear goal, one place that changes as the story moves, a problem in the middle, and an unexpected ending whose twist is set up by details planted earlier.
- Every scene follows from the previous one. No restarting, no unrelated jumps.
- Vivid details, realistic dialogue, honest reactions, small funny or awkward moments.

USING THE WORDS
- The list is in random order. Do not work through it in order: place each word in whichever scene it fits most naturally, and never force two unrelated words into one sentence.
- Use each word once, in its standard sense, in a sentence where the meaning is clear from context alone. A second use is allowed, but only the first one is bolded.
- A variant of a word counts as using it: run/running/ran, quick/quickly, look/looked/looking, and for a phrase, one word of the phrase inflected.

BOLDING (graded strictly)
- Bold exactly one occurrence of each word in the list, using **double asterisks**. The bolded word may be a variant of the listed one.
- Bold NOTHING else. Every other word in the story stays unbolded, including ordinary words like "use", "already" or "not" unless they are in the list.
  Example with a list of [run, ephemeral, piece of cake]:
  Right: a **run** in the rain, an **ephemeral** victory, a **piece of cake** to fix it.
- If a listed word is an everyday word, use it in its less obvious sense and still bold only that one occurrence.
- The title must not be bolded.

NEVER NARRATE YOUR OWN WORK
- Your answer is the title and then the story. Nothing else goes in it.
- Do not mention the list, the checklist, the draft, or which words you have or have not used. Never write a sentence like "Not a listed word." or "use edible already used" or "not listed".
- No commentary about your own writing. Fix anything you get wrong silently; never describe the fix.

LANGUAGE
- Learner level B1-C1: mostly short-to-medium sentences (under 25 words), plain words everywhere except the listed ones, no idioms the narrator would have to explain.
- No swearing. Never write the word "Reddit". No "Edit" or "Update" line at the end.
- Plain prose only: no headings, lists, block quotes, code fences or commentary around the story.

TITLE
Invent a short, catchy title of at most 8 words."""


def _render_word_list(words):
    # Comma-separated, with multi-word phrases quoted so they stay one item.
    # Deliberately NOT numbered: a numbered list (1. scavenge, 2. traits, ...)
    # was measured leaking into the output as "**scavenge** 1", and it also
    # invited the model to work through the list in order, which the prompt
    # explicitly forbids. See _detect_degeneracy's index-fusion check.
    return ", ".join(f'"{w}"' if " " in w else w for w in words)


def _resolve_style(style=None):
    """A concrete style string. Empty means draw a random one per story.

    Exposed so the caller can resolve it once and pass the result down: if the
    random draw happened inside the generation, the prompt snapshotted on the
    story row and the prompt actually sent could name different styles.
    """
    return (style or "").strip() or random.choice(REDDIT_STYLES)


def build_prompt(words, template=None, style=None):
    """Render the user message from the editable template.

    The template is sent as written -- no placeholders, no substitution. That
    is what makes braces safe in prose: a template mentioning JSON or a set
    literal is just text. (It used to be str.replace, never str.format, for the
    same reason.)
    ``style`` empty means "draw a random one from REDDIT_STYLES", the shipped
    behaviour.

    Only the TAIL is generated: the style line and the word list. The task
    sentence is part of the template, so it is editable like everything else the
    model reads -- an empty template therefore produces no opening sentence at
    all, which is the user's choice, not a bug.
    """
    if template is None:
        template = _DEFAULT_TEMPLATE
    style = _resolve_style(style)
    # Joined from parts rather than one f-string so an empty template cannot
    # leave leading blank lines at the top of the message. The header and the
    # list share ONE part: joining them like the others would put a blank line
    # between "WORDS TO USE" and the list, which is not what shipped.
    parts = [template.strip(),
             f"Style for this story: {style}.",
             "WORDS TO USE\n" + _render_word_list(words)]
    return "\n\n".join(p for p in parts if p)



def build_messages(words, template=None, style=None, system=None):
    """The full message list for a story request.

    `system` overrides the editable system message; the two labels the parsers
    need are checked by validate_prompt(), not here, so a bad one is caught at
    save time rather than turning every story into "Untitled Story".
    """
    return [
        {"role": "system", "content": (system or _DEFAULT_SYSTEM).strip()},
        {"role": "user", "content": build_prompt(words, template, style)},
    ]


# ---------------------------------------------------------------------------
# Template validation.
#
# The template is user-editable, so validation is a warning strip rather than a
# gate: only an unrenderable template is an error, everything else is advice.
# The point is to surface the foot-guns that were once encoded as comments in
# this file, where an editor could actually read them. Nothing here changes the
# prompt -- a warning the user ignores still produces a story.
# ---------------------------------------------------------------------------

# NOTE: the self-check findings ("this asks the model to check its own work")
# were removed deliberately, together with the detector above them. The user can
# now bound the damage instead -- reasoning.max_tokens caps how long a reasoning
# model may think (see _with_reason_cap) -- which is the lever that actually
# holds. The warning was never a gate, only advice, and it fired on honest
# writing instructions ("Then verify that:" in a system message) as readily as on
# the 405s case it was written for.
#
# The measurement is kept here because it is the reason the cap exists and the
# reason reason/vis stays in the health log: telling a reasoning model to check
# its own output cost 405s and 26,518 reasoning tokens for a 286-word story,
# against 68s for the same model without it. `_where()` and the pattern set
# below remain for the findings that are still reported (length_target).


_LENGTH_RE = re.compile(
    r"\b(\d{2,5}\s*[-–]?\s*words?\b|under \d+|at least \d+ words|"
    r"no more than \d+ words|at most \d+ words|around \d+ words|about \d+ words|"
    r"short story|long story)",
    re.IGNORECASE,
)


# A limit on a sub-unit (a sentence, a line, the title) is a different rule from
# a limit on the story, so the check looks at the match's OWN LINE only. An
# earlier version used a +/-60 character window, which was wrong in both
# directions: it warned on the shipped default's own sentence and title limits,
# and once those were carved out it also swallowed a real "Keep the story under
# 300 words" that happened to sit under the title line.
_SUBUNIT_WORDS = ("sentence", "line", "title")


def _line_bounds(text):
    """Byte offset where each line starts, so a match maps back to its line."""
    starts, pos = [], 0
    for line in text.splitlines():
        starts.append(pos)
        pos += len(line) + 1
    return starts


def _line_index(starts, offset):
    idx = 0
    for i, start in enumerate(starts):
        if start <= offset:
            idx = i
        else:
            break
    return idx


def _where(text, match, starts=None):
    """Locate a regex match for the user: 1-based line number plus the trimmed
    line it sits on. A finding that cannot be located is a finding the user
    cannot act on: the prompt is a wall of forty lines, and being told that a
    check crept in somewhere is no use at all."""
    starts = _line_bounds(text) if starts is None else starts
    lines = text.splitlines()
    idx = _line_index(starts, match.start())
    line = lines[idx].strip() if idx < len(lines) else ""
    if len(line) > 110:
        line = line[:110] + "..."
    return f"Line {idx + 1}: {line}"


def _story_length_hit(text):
    """The regex match that asks for a length for the story itself, or None.

    Returns the match rather than a bool so the caller can point at it. The
    decision and the report must come from the same walk: an earlier version
    had the carve-out in two places and they could have drifted apart."""
    lines = text.splitlines()
    starts = _line_bounds(text)
    for m in _LENGTH_RE.finditer(text):
        idx = _line_index(starts, m.start())
        line = lines[idx].lower() if idx < len(lines) else ""
        if any(w in line for w in _SUBUNIT_WORDS):
            continue
        return m
    return None

# Re-introducing a numbered list was measured leaking indices into the prose.
_NUMBERED_RE = re.compile(
    r"\b(number(ed)? (the |your )?(list|words|them)|list them (as )?numbered|"
    r"numbered list)\b",
    re.IGNORECASE,
)

def validate_prompt(template, system=None):
    """Check an editable template (and optionally the system message).

    Each finding is ``{"level", "code", "message"}`` with level in
    "error" / "warning" / "info". Only "error" prevents rendering; the UI shows
    the rest and lets the user save anyway.
    """
    findings = []
    # An empty USER message is legal and was requested: sometimes the system
    # message plus the generated tail (style line + word list) is the whole
    # prompt. build_prompt() joins only its non-empty parts, so the request still
    # renders and the model never receives an empty user message. It is a
    # warning rather than nothing, because everything the model then does is
    # decided by the system message alone and the user should see that stated.
    #
    # The early return that used to sit here is gone, and with it a real bug: it
    # returned BEFORE the system checks, so a preset with an empty template hid a
    # broken system message (a missing "Title:" label) that would have refused
    # the save anyway. Findings are collected for both messages, always.
    empty_user = template is None or not template.strip()

    def add(level, code, message):
        findings.append({"level": level, "code": code, "message": message})

    text = template or ""

    # --- The system message -------------------------------------------------
    # Editable, but the two labels are a hard requirement, so this one is an
    # error rather than advice: without them the title is lost and TTS reads the
    # labels aloud. Everything else about the wording is the user's call.
    if system is not None:
        if not str(system).strip():
            add("error", "empty_system",
                "The system message is empty. Reset to the code default to recover it.")
        else:
            missing = [label for label in ("Title:", "Story:") if label not in str(system)]
            if missing:
                add("error", "missing_labels",
                    "The system message must keep the "
                    + " and ".join(f'"{m}"' for m in missing)
                    + " label(s). Without them the title cannot be read back and "
                      "the labels would end up in the spoken text.")

    # No placeholder checks, and deliberately no brace check either:
    # the template is sent verbatim, so "{" carries no meaning and a
    # prompt that mentions JSON or a set literal must not be refused
    # for being unbalanced.
    # No placeholder checks, and deliberately no brace check either: the template
    # is sent verbatim, so "{" carries no meaning and a prompt that mentions
    # JSON or a set literal must not be refused for being unbalanced.

    length_hit = _story_length_hit(text)
    if length_hit is not None:
        add("warning", "length_target",
            "This sets a target length. The prompt deliberately sets none, and a "
            "length target tends to make the model rush the word list. max_tokens "
            "is the output budget and is set separately. -- "
            + _where(text, length_hit))

    if _NUMBERED_RE.search(text):
        add("warning", "numbered_list",
            "Do not ask for a numbered list. It was measured leaking into the "
            "prose as '**scavenge** 1' and making the model work through the list "
            "in order. The word list is rendered for you, un-numbered.")

    if "Title:" in text or "Story:" in text:
        add("warning", "output_shape",
            "The output shape (the 'Title:' and 'Story:' labels) is set by the "
            "system message and cannot be changed here. Repeating it in the "
            "prompt tends to produce a preamble.")

    # The bolding rule can live in EITHER message, and the two are read as one
    # prompt by the model. Checking only the template reported "the model will
    # probably bold nothing" for a preset whose bolding rule was in the system
    # message -- advice that was simply wrong, and it fired most loudly on an
    # empty user message, where the template cannot mention anything at all.
    if "bold" not in (text + "\n" + ("" if system is None else str(system))).lower():
        add("info", "no_bolding_rule",
            "No bolding rule. The model will probably bold nothing, so the "
            "highlighted vocabulary in the story will be lost.")

    return findings


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

def _token_forms(term):
    """Every single-token form of `term` the model may legitimately produce.

    One definition shared by the bold sanitizer and the health checks, so
    "is this bold allowed?" and "was this word used?" can never disagree.
    Deliberately over-permissive: keeping a stray bold is harmless, deleting a
    correct highlight is not.
    """
    normalized = _normalize_term(term)
    forms = {normalized} if normalized else set()
    if not normalized:
        return forms
    bases = _variants_of(normalized)
    # A phrase entry also licenses any single word of it, since the prompt lets
    # the model inflect one word of a phrase.
    for part in normalized.split():
        bases |= _variants_of(part)
    for base in bases:
        forms.add(base)
        # A listed word may be the one that got inflected in the story, so
        # accept the derived forms of every form we know about.
        for ending in _DERIVATIONS:
            forms.add(base + ending)
            if base.endswith("e"):
                forms.add(base[:-1] + ending)
        # consonant + y: "try" -> "tried", "tries", "trier".
        if base.endswith("y") and len(base) >= 3:
            stem = base[:-1]
            for ending in ("ied", "ies", "ier", "iest", "ily", "ying"):
                forms.add(stem + ending)
    return forms

def _allowed_bold_terms(words):
    """Every single token that may legitimately be bolded for this word list."""
    allowed = set()
    for word in words or []:
        allowed |= _token_forms(word)
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


# --- Generation health checks -------------------------------------------------
# These measure the *quality* of what came back, so a bad run is visible in the
# log instead of only in the stored prose. Every check here is deliberately
# high-precision, because a false positive throws away a good generation and
# costs the user a retry.

# "**scavenge** 47" — a list index leaking into the prose. Only ever produced
# when the prompt presented the word list numbered.
_INDEX_FUSION = re.compile(r"\*\*[^*\n]{1,30}\*\*\s+\d{1,3}\b")

# Fraction of the prose that may be listed words before it is a word salad
# rather than a story. This is deliberately LOOSE: asking for 286 target words
# in a short story is inherently dense (a known-good 286-word story measured
# 22%), so a tight cap would fail the very thing that was requested. 45% is
# where the prose stops reading as English at all.
_MAX_DENSITY = 0.45


def _story_body(content):
    """The prose only, with the Title:/Story: labels removed."""
    body = re.sub(r"^\s*Title:.*$", "", content or "", count=1, flags=re.M)
    return re.sub(r"^\s*Story:\s*", "", body, count=1).strip()


def _prose_word_count(body):
    """Word tokens in the prose, which is the denominator of the density stat.

    One definition, shared by the word-salad guard in _detect_degeneracy and the
    density figure returned for display, so the number the user sees is by
    construction the same number that can fail a generation at _MAX_DENSITY.
    """
    return len(re.findall(r"\b[\w']+\b", body or ""))


@lru_cache(maxsize=4096)
def _word_pattern(term):
    """Regex matching a word or any inflection of it, as a whole word.

    A multi-word entry ("piece of cake", "barge into") requires *all* of its
    parts, in order, with a couple of words allowed between them. Matching any
    single part would make "of" or "into" count as a used word, which would
    inflate the coverage figure.

    Cached because the alternation is large (21+ branches for one term) and
    building it dominates the cost of a coverage pass: on a 299-word list,
    compiling costs ~250ms against ~38ms for the actual searching. Coverage is
    recomputed on every manual story edit, so this path runs outside generation
    too.
    """
    normalized = _normalize_term(term)
    if not normalized:
        return None
    parts = [p for p in normalized.split() if p]
    if not parts:
        return None
    chunks = []
    for part in parts:
        forms = {f for f in _token_forms(part) if f and " " not in f}
        if not forms:
            return None
        chunks.append("(?:" + "|".join(sorted((re.escape(f) for f in forms),
                                             key=len, reverse=True)) + ")")
    if len(chunks) == 1:
        return re.compile(r"\b" + chunks[0] + r"\b", re.I)
    # Up to two intervening words between parts ("barge straight into").
    return re.compile(r"\b" + r"(?:\s+\S+){0,2}\s+".join(chunks) + r"\b", re.I)


def _count_words_used(content, words):
    """How many of the listed words appear in the prose (variants count)."""
    body = _story_body(content)
    if not body:
        return 0
    used = 0
    for word in words or []:
        pattern = _word_pattern(word)
        if pattern and pattern.search(body):
            used += 1
    return used


def _count_stray_bold(content, words):
    """Bolded single-token spans that are not a listed word or a variant."""
    allowed = _allowed_bold_terms(words)
    if not allowed:
        return 0
    stray = 0
    for inner in re.findall(r"\*\*(.+?)\*\*", content or "", flags=re.S):
        if " " in inner.strip():
            continue
        if _normalize_term(inner) not in allowed:
            stray += 1
    return stray


def _detect_degeneracy(content, words):
    """Return a list of reasons the output looks suspect (empty means fine).

    ADVISORY ONLY. These reasons are stored on the story and shown as a warning
    pill; nothing is discarded because of them. The function used to fail the
    generation outright, and the docstring below explains why that history still
    matters: every check here was tuned for precision under the assumption that a
    false positive would cost the user a whole retry. Keep that bar when adding
    checks — a false positive now costs a misleading pill rather than a lost
    story, but a noisy flag trains you to ignore the pill.

    Verified against every story in the local database: all of them pass. Three
    checks were tried and removed — a per-word repetition count (it flagged the
    title word of ordinary stories, "barn" 7x, "cucumber" 12x), a tight density
    cap (it flagged a known-good 286-word story at 22%), and self-talk detection
    (see below). The duplicate-paragraph check already catches a redraft loop
    exactly.

    Self-talk detection ("not a listed word", "already used", "word list", ...)
    was removed on 2026-09-29: it fired ONCE on a healthy 310-word run
    (used=270/310, stray_bold=0, reason/vis=0.4, 148s, 22.8k chars) and threw
    the whole generation away. Those phrases occur in ordinary fiction, and with
    a single-match threshold the false-positive rate scales with story length.
    Do not re-add it as "cheap insurance" — the prompt rule in _DEFAULT_TEMPLATE is
    what keeps the meta text out of the output in the first place.

    Low coverage is deliberately NOT a reason. It is the most useful signal in
    the health log (`used=n/total`) and is surfaced as its own coverage stat,
    which is a measurement rather than a judgement.
    """
    reasons = []
    body = _story_body(content)
    if not body:
        return ["empty body"]

    if _INDEX_FUSION.search(body):
        reasons.append("list indices leaked into the prose")

    # A paragraph repeated verbatim is the visible trace of a redraft loop.
    paragraphs = [p.strip() for p in body.split("\n\n") if len(p.strip()) > 80]
    if len(paragraphs) != len({p for p in paragraphs}):
        reasons.append("duplicate paragraph (redraft loop)")

    total_words = _prose_word_count(body)
    if total_words:
        density = _count_words_used(content, words) / total_words
        if density > _MAX_DENSITY:
            reasons.append(f"target vocabulary is {density:.0%} of the prose (word salad)")

    return reasons


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
    generation_id = ""
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
            if chunk.get("id"):
                # OpenRouter's generation id: the handle for looking this run up
                # in the dashboard, which is the only place finish_reason and the
                # billed token counts appear when the stream omits them.
                generation_id = chunk["id"]
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
            generation_id = payload.get("id") or generation_id
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
        "id": generation_id or None,
        "finish_reason": finish_reason,
        "native_finish_reason": native_finish_reason,
        "native_tokens_reasoning": native_tokens_reasoning,
    }, None


def sync_generate_story(words, model=None, provider_tag=None, story_id=None,
                        on_delta=None, template=None, style=None, system=None,
                        temperature=None, max_tokens=None, reasoning=None,
                        reasoning_max_tokens=None):
    """Generate one story.

    template/system/style/temperature/max_tokens/reasoning/reasoning_max_tokens
    override the shipped prompt and request defaults. They are passed in
    explicitly (rather than read from the settings table here) so the exact
    prompt used is fixed when the story row is created and cannot shift under
    a generation already in flight.
    """
    if not OPENROUTER_API_KEY:
        return {"ok": False, "error": "OpenRouter API key not set. Add OPENROUTER_API_KEY to your .env file."}
    if not words:
        return {"ok": False, "error": "No words were provided to build a story."}

    model = model or OPENROUTER_MODEL
    if temperature is None:
        temperature = _STORY_TEMPERATURE
    if max_tokens is None:
        max_tokens = _STORY_MAX_TOKENS
    prompt = build_prompt(words, template=template, style=style)

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
        "messages": build_messages(words, template=template, style=style,
                                   system=system),
        "temperature": temperature,
        # Output budget for the STORY. It is the only length bound on prose: the
        # prompt deliberately sets no target length. A story cut off here is
        # still published, as status 'truncated' (see _STORY_MIN_VISIBLE_TOKENS).
        # It does NOT bound reasoning -- that is reasoning.max_tokens below.
        "max_tokens": max_tokens,
        # Stream so the request can be cancelled mid-generation and the UI can
        # render the story as it arrives. include_usage keeps the token/cost
        # accounting identical to the non-streaming path.
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    if _DEBUG_PROMPT:
        # The whole prompt is the contract with the model, so being able to
        # read exactly what was sent is worth a debug switch while the
        # template is being edited from the UI.
        print(f"--- prompt for story {story_id} ---\n{prompt}\n--- end prompt ---", flush=True)

    # Output budget, needed again below to recognise a capped response when the
    # provider omits finish_reason.
    max_tokens = payload["max_tokens"]

    # Reasoning config: derived from the model catalogue, or from the preset's
    # preference. `reasoning` is only a request -- see use_reasoning_status().
    reasoning_config = _get_reasoning_config(model, reasoning, reasoning_max_tokens)
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
            f"| reason_cfg={_reason_cfg_label(reasoning_config)}"
            + (f" (asked: {reasoning})" if reasoning and reasoning != "auto" else ""),
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
        generation_id = data.get("id") or ""
        finish = data.get("finish_reason") or data.get("native_finish_reason") or "?"
        # "length" means the model ran into max_tokens and stopped mid-sentence.
        # The prose is still worth showing, so it is reported to the caller
        # rather than treated as a failure (main.py stores it as a 'truncated'
        # story the user can retry). The exception is a reasoning model that
        # spent the whole budget thinking and emitted almost no visible text:
        # there is nothing worth keeping, so that fails outright.
        truncated = finish == "length"
        truncate_source = "finish_reason" if finish != "?" else None
        if not truncated and finish == "?" and not reasoning_tokens:
            # Some providers (x-ai/grok-4.7 observed) send no finish_reason in
            # the stream at all, so a capped response is indistinguishable from
            # a complete one. For a non-reasoning model completion_tokens is the
            # visible output and sits exactly at max_tokens when capped, which
            # makes this a reliable fallback. Guarded on reasoning_tokens == 0
            # because a reasoning model's completion_tokens bundles reasoning,
            # which is NOT counted against max_tokens and can exceed it freely.
            if completion_tokens >= max_tokens:
                truncated = True
                truncate_source = "token-cap"
        # Too little actual story to be worth keeping -- and this is NOT the same
        # test as "truncated": a reasoning model can spend the whole budget
        # thinking and emit a sentence without ever hitting the cap. Only checked
        # when the provider reported usage at all, because otherwise
        # completion_tokens is 0 for every story and this would reject them all.
        if usage and visible_tokens < _STORY_MIN_VISIBLE_TOKENS:
            if truncated:
                detail = ("the model used the whole token budget reasoning and "
                          "left almost no story text")
            else:
                detail = ("the model returned almost no story text for this word "
                          "list")
            return {
                "ok": False,
                "error": (
                    f"Error generating story: {detail}. Try a different model, or "
                    f"generate with fewer words."
                ),
            }
        actual_model = data.get("model", model)
        chars = len(content)
        native_reason = data.get("native_tokens_reasoning")
        native_part = f" native_reason={native_reason}" if native_reason is not None else ""
        id_part = f"id={story_id} " if story_id is not None else ""

        # Sanitize first, then measure: the health figures must describe what
        # actually gets stored, not what the model sent.
        cleaned = _strip_stray_bold(content, words)

        # Health metrics. Without these the log only showed token counts, which
        # made it impossible to tell a good run from a degenerate one: a 286-word
        # story once came back "ready" after 405s and 26.5k reasoning tokens.
        used = _count_words_used(cleaned, words)
        stray_bold = _count_stray_bold(cleaned, words)
        # Denominator for the density stat shown in the UI. Same helper the
        # word-salad guard uses, so the displayed percentage and the threshold
        # that can fail a generation can never drift apart.
        prose_words = _prose_word_count(_story_body(cleaned))
        health = f"used={used}/{len(words)} stray_bold={stray_bold}"
        if prose_words:
            health += f" density={used / prose_words:.0%}"
        if visible_tokens:
            health += f" reason/vis={reasoning_tokens / visible_tokens:.1f}"

        degeneracy = _detect_degeneracy(cleaned, words)
        gen_part = f" gen={generation_id}" if generation_id else ""
        trunc_part = f" truncated({truncate_source})" if truncated else ""
        print(
            f"[story] {id_part}ok | model={actual_model} via {provider_name}{gen_part} | words={len(words)} | "
            f"in={prompt_tokens} out={completion_tokens} (reason={reasoning_tokens}, visible={visible_tokens}) "
            f"total={total_tokens}{native_part} | reason_cfg={_reason_cfg_label(reasoning_config)} | "
            f"{elapsed:.1f}s | {finish}{trunc_part} | ${cost:.4f} | chars={chars} | {health}",
            flush=True,
        )
        if degeneracy:
            # ADVISORY ONLY. This used to fail the generation and throw the prose
            # away, but the text was readable in every measured case -- and the
            # tokens were already spent, so discarding bought nothing and cost the
            # user a retry. The reasons are stored on the story and shown as a
            # warning pill instead. The stderr line stays because the health log is
            # how a bad run gets noticed at all.
            print(f"[story] {id_part}flagged output: {'; '.join(degeneracy)}", file=sys.stderr, flush=True)

        return {
            "ok": True,
            "content": cleaned,
            "elapsed": elapsed,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "reasoning_tokens": reasoning_tokens,
            # True when the response was cut off by max_tokens (either reported
            # as finish_reason "length" or inferred from the token cap).
            # main.py publishes it anyway, as a 'truncated' story, so the user
            # can read it and retry.
            "truncated": truncated,
            "truncated_source": truncate_source,
            "generation_id": generation_id,
            # Advisory only -- reasons from _detect_degeneracy, empty when the run
            # was clean. main.py stores them for display; nothing is discarded on
            # account of them. NOT the same as 'truncated', which is a real status.
            "warnings": degeneracy,
            # Target-word coverage. `words_used` counts a listed word as used
            # when it appears at least once, counting inflections and phrase
            # variants (_token_forms); `words_total` is the submitted list, which
            # can be larger than the story_words rows if a word had no dictionary
            # entry, so the denominator is stored rather than derived.
            "words_used": used,
            "words_total": len(words),
            # Prose word tokens, the denominator of the density the UI shows.
            "prose_words": prose_words,
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
