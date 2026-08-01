import os

import requests
from dotenv import load_dotenv

load_dotenv()


def get_api_key() -> str | None:
    return os.getenv("GROQ_API_KEY")



def get_base_url() -> str:
    return os.getenv(
        "GROQ_API_BASE_URL",
        "https://api.groq.com/openai/v1"
    )

def configure_api_key() -> str:
    api_key = get_api_key()
    if not api_key:
        raise EnvironmentError("Missing GROQ_API_KEY in environment")
    return api_key


def validate_api_key() -> dict:
    api_key = configure_api_key()
    url = f"{get_base_url()}/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.get(url, headers=headers, timeout=20)
    response.raise_for_status()
    return response.json()
