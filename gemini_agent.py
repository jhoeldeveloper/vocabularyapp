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


def sync_get_meanings_of(word: str) -> str:
    if not model:
        return "Gemini client not initialized."
    try:
        prompt = f"Define the word '{word}'. Provide a concise, clear definition. Do not use lists or bullet points. Use smart highlighting for emphasis."
        print("sending prompt for meanings...")
        response = model.generate_content(prompt)
        print("received response for meanings")
        return response.text
    except Exception as e:
        return f"Error fetching meaning: {e}"


def sync_get_sentences_with(word: str) -> str:
    if not model:
        return "Gemini client not initialized."
    try:
        prompt = f"Create 5 example sentences using the word '{word}'. Do not use lists or numbered formats. Use line breaks between sentences and always highlight the word '{word}'."
        print("sending prompt for sentences...")
        response = model.generate_content(prompt)
        print("received response for sentences")
        return response.text
    except Exception as e:
        return f"Error fetching sentences: {e}"


def sync_get_synonyms_of(word: str) -> str:
    if not model:
        return "Gemini client not initialized."
    try:
        prompt = f"List 5 synonyms for the word '{word}'. Provide them as a comma-separated list. If the word has multiple meanings, include synonyms for each. Only output the synonyms, no extra text."
        print("sending prompt for synonyms...")
        response = model.generate_content(prompt)
        print("received response for synonyms")
        return response.text
    except Exception as e:
        return f"Error fetching synonyms: {e}"
