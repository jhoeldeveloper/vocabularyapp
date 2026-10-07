import os
from groq import Groq
from dotenv import load_dotenv
# Shared meaning post-processing (sense format, full stops, literal `\n`
# repairs) -- see gemini_agent.py for why this import is unconditional.
import openrouter_agent

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "openai/gpt-oss-120b")

client = None
if not GROQ_API_KEY:
    print("CRITICAL: GROQ_API_KEY is not set in your .env file. Groq functions will fail.")
else:
    try:
        client = Groq(api_key=GROQ_API_KEY)
        print(f"Groq model '{GROQ_MODEL_NAME}' initialized successfully.")
    except Exception as e:
        print(f"Error initializing Groq client: {e}")


def is_ready() -> bool:
    return client is not None


_MEANINGS_PROMPT = (
    "For the English word or phrase '{word}', write the 2 most common senses as a "
    "numbered Markdown list. Each item is `1. <concise gloss.>` -- ending the "
    "gloss with a full stop -- and, on the next "
    "line indented by three spaces, one short example sentence using that sense "
    "with the word highlighted in **bold**. The senses must be genuinely "
    "different. Do not give a frequency score, word class or inflected forms "
    "-- those are measured elsewhere."
)


def sync_get_meanings_of(word: str) -> str:
    if not client:
        return "Groq client not initialized."
    try:
        print("sending prompt for meanings...")
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful dictionary assistant."},
                {"role": "user", "content": _MEANINGS_PROMPT.format(word=word)},
            ],
            model=GROQ_MODEL_NAME,
        )
        print("received response for meanings")
        return openrouter_agent.normalize_meaning(response.choices[0].message.content)
    except Exception as e:
        return f"Error fetching meaning: {e}"


def sync_get_use_of(word: str) -> str:
    """American / British / Both, matching the OpenRouter agent's field set.

    Kept in step deliberately: main.py fans out to the same two functions for
    whichever agent is configured, and its fail-closed check inspects both
    results. An agent that returns a third field, or a different one, would
    write a word the others cannot read.
    """
    if not client:
        return "Groq client not initialized."
    try:
        print("sending prompt for dialect use...")
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful dictionary assistant."},
                {"role": "user", "content": (
                    f"Is the English word '{word}' distinctly American, distinctly "
                    f"British, or both? Answer with exactly one word: American, "
                    f"British or Both. If the word has no dialect distinction, "
                    f"answer Both."
                )},
            ],
            model=GROQ_MODEL_NAME,
        )
        print("received response for dialect use")
        answer = (response.choices[0].message.content or "").strip().lower()
        for option in ("american", "british", "both"):
            if option in answer:
                return option.capitalize() if option != "both" else "Both"
        return "Both"
    except Exception as e:
        return f"Error fetching use: {e}"



_SYNONYMS_PROMPT = (
    "List up to 5 true synonyms for the most common sense of the English word "
    "or phrase '{word}', as a single comma-separated line. Prefer everyday words "
    "a learner would actually use and leave out obscure or archaic ones. If it "
    "genuinely has no synonym, reply with nothing. Output only the list."
)


def sync_get_synonyms_of(word: str) -> str:
    """Stored once at add time. Not measured and not derived: the model's own
    judgement, which is why it round-trips through the DB unlike freq/family."""
    if not client:
        return "Groq client not initialized."
    try:
        print("sending prompt for synonyms...")
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful dictionary assistant."},
                {"role": "user", "content": _SYNONYMS_PROMPT.format(word=word)},
            ],
            model=GROQ_MODEL_NAME,
        )
        print("received response for synonyms")
        return response.choices[0].message.content
    except Exception as e:
        return f"Error fetching synonyms: {e}"
