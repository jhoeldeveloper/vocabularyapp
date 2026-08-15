import os
from groq import Groq
from dotenv import load_dotenv

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


def sync_get_meanings_of(word: str) -> str:
    if not client:
        return "Groq client not initialized."
    try:
        prompt = f"Define the word/phrase '{word}'. Provide a concise, natural and clear definition. Do not use dictionary format. Do not use lists or bullet points. Use smart highlighting for emphasis."
        print("sending prompt for meanings...")
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful dictionary assistant."},
                {"role": "user", "content": prompt},
            ],
            model=GROQ_MODEL_NAME,
        )
        print("received response for meanings")
        return response.choices[0].message.content
    except Exception as e:
        return f"Error fetching meaning: {e}"


def sync_get_sentences_with(word: str) -> str:
    if not client:
        return "Groq client not initialized."
    try:
        prompt = f"Create 5 example sentences using the word '{word}'. Do not use lists or numbered formats. Use line breaks between sentences and always highlight the word '{word}'."
        print("sending prompt for sentences...")
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful dictionary assistant."},
                {"role": "user", "content": prompt},
            ],
            model=GROQ_MODEL_NAME,
        )
        print("received response for sentences")
        return response.choices[0].message.content
    except Exception as e:
        return f"Error fetching sentences: {e}"


def sync_get_synonyms_of(word: str) -> str:
    if not client:
        return "Groq client not initialized."
    try:
        prompt = f"List 5 synonyms for the word '{word}'. Provide them as a comma-separated list. If the word has multiple meanings, include synonyms for each. Only output the synonyms, no extra text."
        print("sending prompt for synonyms...")
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful dictionary assistant."},
                {"role": "user", "content": prompt},
            ],
            model=GROQ_MODEL_NAME,
        )
        print("received response for synonyms")
        return response.choices[0].message.content
    except Exception as e:
        return f"Error fetching synonyms: {e}"
