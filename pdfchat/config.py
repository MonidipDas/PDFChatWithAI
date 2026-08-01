import os
import time
import requests
import streamlit as st
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from threading import Lock
from urllib3.util.retry import Retry

load_dotenv()

DEFAULT_RATE_LIMIT_REQUESTS = 5
DEFAULT_RATE_LIMIT_PERIOD = 1.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_FACTOR = 1.0


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


def _get_rate_limit_settings():
    return (
        int(os.getenv("GROQ_API_RATE_LIMIT_REQUESTS", DEFAULT_RATE_LIMIT_REQUESTS)),
        float(os.getenv("GROQ_API_RATE_LIMIT_PERIOD", DEFAULT_RATE_LIMIT_PERIOD)),
    )


class _RateLimiter:
    def __init__(self, capacity: int, period: float):
        self.capacity = capacity
        self.period = period
        self.lock = Lock()
        self.calls = []

    def acquire(self) -> None:
        while True:
            with self.lock:
                now = time.monotonic()
                self.calls = [timestamp for timestamp in self.calls if timestamp > now - self.period]
                if len(self.calls) < self.capacity:
                    self.calls.append(now)
                    return
                wait = self.period - (now - self.calls[0])
            time.sleep(max(wait, 0.0))


def _make_session() -> requests.Session:
    retries = Retry(
        total=int(os.getenv("GROQ_API_MAX_RETRIES", DEFAULT_MAX_RETRIES)),
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS", "POST", "PUT", "DELETE", "PATCH"],
        backoff_factor=DEFAULT_BACKOFF_FACTOR,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


_RATE_LIMITER = _RateLimiter(*_get_rate_limit_settings())
_SESSION = _make_session()

def make_api_request(
    method: str,
    path: str,
    headers: dict | None = None,
    timeout: int = 20,
    json: dict | None = None,
    params: dict | None = None,
) -> requests.Response:
    url = f"{get_base_url().rstrip('/')}/{path.lstrip('/')}"
    request_headers = {
        "Authorization": f"Bearer {get_api_key()}",
    }
    if headers:
        request_headers.update(headers)

    max_attempts = int(os.getenv("GROQ_API_MAX_RETRIES", DEFAULT_MAX_RETRIES)) + 1
    response = None
    for attempt in range(1, max_attempts + 1):
        _RATE_LIMITER.acquire()
        response = _SESSION.request(
            method,
            url,
            headers=request_headers,
            timeout=timeout,
            json=json,
            params=params,
        )

        if response.status_code != 429:
            response.raise_for_status()
            return response

        retry_after = response.headers.get("Retry-After")
        if retry_after and retry_after.isdigit():
            wait = float(retry_after)
        else:
            wait = min(2 ** (attempt - 1), 10)
        time.sleep(wait)

    if response is not None:
        response.raise_for_status()
    raise RuntimeError("Failed to perform Groq API request")

def validate_api_key():
    response = make_api_request("GET", "/models", timeout=20)
    return response.json()