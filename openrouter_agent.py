import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
MODELS_URL = "https://openrouter.ai/api/v1/models"

# Lazily-loaded {model_id: {prompt, completion}} price-per-token map (best-effort).
_pricing_cache = None


def _get_pricing():
    global _pricing_cache
    if _pricing_cache is None:
        try:
            r = requests.get(MODELS_URL, timeout=15)
            data = r.json().get("data", []) if r.ok else []
            _pricing_cache = {}
            for m in data:
                p = m.get("pricing") or {}
                _pricing_cache[m["id"]] = {
                    "prompt": float(p.get("prompt") or 0),
                    "completion": float(p.get("completion") or 0),
                }
        except Exception:
            _pricing_cache = {}
    return _pricing_cache


def is_ready() -> bool:
    return bool(OPENROUTER_API_KEY)


def _build_prompt(words, title):
    word_list = ", ".join(words)
    title_instruction = (
        f"Use exactly this title: '{title}'." if title
        else "Invent a short, fitting title."
    )
    return f"""Write a coherent, engaging story that naturally incorporates ALL of the following vocabulary words: {word_list}.

Use each word correctly in context so its meaning is clear from context. Make the story flow naturally and be enjoyable to read.

Emphasize every occurrence of the vocabulary words by wrapping them in Markdown bold (e.g. **word**), so they stand out for a language learner.

{title_instruction}

Output format (exactly two labelled sections):
Title: <the title>
Story: <the story text>
"""


def sync_generate_story(words, title=None, model=None):
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
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a creative writing assistant that weaves vocabulary words into short, coherent stories.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.9,
    }

    started = time.monotonic()
    try:
        print(f"story: requesting OpenRouter model '{model}' for {len(words)} words")
        try:
            response = requests.post(
                OPENROUTER_BASE_URL, headers=headers, json=payload, timeout=(10, 240)
            )
        except requests.exceptions.Timeout:
            return {"ok": False, "error": (
                "Error generating story: request timed out. The model may be slow or rate-limited "
                "for this many words. Try a smaller word set or a different model.")}
        except requests.exceptions.RequestException as e:
            return {"ok": False, "error": f"Error generating story: {e}"}

        if response.status_code != 200:
            detail = response.text[:400].replace("\n", " ")
            return {"ok": False, "error": f"Error generating story ({response.status_code}): {detail}"}

        try:
            data = response.json()
        except ValueError:
            return {"ok": False, "error": f"Error generating story: invalid JSON response ({response.status_code})."}
        print("story: received response")

        elapsed = time.monotonic() - started
        msg = (data.get("choices") or [{}])[0].get("message") or {}
        content = msg.get("content") or msg.get("reasoning") or ""
        if not content.strip():
            return {"ok": False, "error": "Error generating story: model returned an empty response (it may have produced only reasoning). Try again or use a different model."}

        usage = data.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens") or 0
        completion_tokens = usage.get("completion_tokens") or 0
        cost = usage.get("cost")
        if cost is None:
            pricing = _get_pricing().get(model) or {}
            cost = prompt_tokens * pricing.get("prompt", 0) + completion_tokens * pricing.get("completion", 0)

        return {
            "ok": True,
            "content": content,
            "elapsed": elapsed,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost": round(cost, 6),
        }
    except Exception as e:
        return {"ok": False, "error": f"Error generating story: {e}"}
