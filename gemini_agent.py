import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2-flash-preview")

model = None
if not GEMINI_API_KEY:
    print("CRITICAL: GEMINI_API_KEY is not set in your .env file. Gemini functions will fail.")
else:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        print(f"Gemini model '{GEMINI_MODEL_NAME}' initialized successfully.")
    except Exception as e:
        print(f"Error initializing Gemini model ({GEMINI_MODEL_NAME}): {e}")


def is_ready() -> bool:
    return model is not None


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
    if not model:
        return "Gemini client not initialized."
    try:
        prompt = _MEANINGS_PROMPT.format(word=word)
        print("sending prompt for meanings...")
        response = model.generate_content(prompt)
        print("received response for meanings")
        return response.text
    except Exception as e:
        return f"Error fetching meaning: {e}"


def sync_get_use_of(word: str) -> str:
    """American / British / Both, matching the OpenRouter agent's field set.

    Kept in step deliberately: main.py fans out to the same two functions for
    whichever agent is configured, and its fail-closed check inspects both
    results. An agent that returns a third field, or a different one, would
    write a word the others cannot read.
    """
    if not model:
        return "Gemini client not initialized."
    try:
        prompt = (
            f"Is the English word '{word}' distinctly American, distinctly "
            f"British, or both? Answer with exactly one word: American, British "
            f"or Both. If the word has no dialect distinction, answer Both."
        )
        print("sending prompt for dialect use...")
        response = model.generate_content(prompt)
        print("received response for dialect use")
        answer = (response.text or "").strip().lower()
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
    if not model:
        return "Gemini client not initialized."
    try:
        print("sending prompt for synonyms...")
        response = model.generate_content(_SYNONYMS_PROMPT.format(word=word))
        print("received response for synonyms")
        return response.text
    except Exception as e:
        return f"Error fetching synonyms: {e}"
