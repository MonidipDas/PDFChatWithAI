import os
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

def get_api_key():
    return (
        os.getenv("GROQ_API_KEY")
        or st.secrets.get("GROQ_API_KEY")
    )

def get_base_url():
    return (
        os.getenv("GROQ_API_BASE_URL")
        or st.secrets.get("GROQ_API_BASE_URL")
        or "https://api.groq.com/openai/v1"
    )

def configure_api_key():
    api_key = get_api_key()
    if not api_key:
        raise EnvironmentError("Missing GROQ_API_KEY")
    return api_key

def validate_api_key():
    api_key = configure_api_key()
    url = f"{get_base_url()}/models"
    headers = {"Authorization": f"Bearer {api_key}"}

    response = requests.get(url, headers=headers, timeout=20)
    response.raise_for_status()
    return response.json()