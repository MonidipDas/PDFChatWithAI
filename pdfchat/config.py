import logging
import os
import random
import time
import requests
import streamlit as st
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from threading import Lock
from urllib3.util.retry import Retry

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_RATE_LIMIT_REQUESTS = 5
DEFAULT_RATE_LIMIT_PERIOD = 1.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_FACTOR = 1.0
DEFAULT_MAX_BACKOFF = 30.0
DEFAULT_JITTER_MIN = 0.5
DEFAULT_JITTER_MAX = 1.0


def _safe_st_secret(key):
    """Read a Streamlit secret, returning None if unavailable."""
    try:
        return st.secrets.get(key)
    except Exception:
        return None


def get_api_key():
    return (
        os.getenv("GROQ_API_KEY")
        or _safe_st_secret("GROQ_API_KEY")
    )


def get_fallback_api_key():
    """Read the fallback Groq API key from env or Streamlit secrets.

    Returns ``None`` when no fallback key is configured.
    """
    return (
        os.getenv("GROQ_FALLBACK_API_KEY")
        or _safe_st_secret("GROQ_FALLBACK_API_KEY")
    )


def get_base_url():
    return (
        os.getenv("GROQ_API_BASE_URL")
        or _safe_st_secret("GROQ_API_BASE_URL")
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
        #maximum capacity requests each period
        
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

def _compute_backoff(attempt: int) -> float:
    """Return a sleep duration using exponential backoff with jitter.

    The base delay doubles each attempt and is capped at ``DEFAULT_MAX_BACKOFF``.
    A random jitter factor between ``DEFAULT_JITTER_MIN`` and
    ``DEFAULT_JITTER_MAX`` is applied to prevent thundering-herd problems.
    """
    base_delay = min(
        DEFAULT_BACKOFF_FACTOR * (2 ** (attempt - 1)),
        DEFAULT_MAX_BACKOFF,
    )
    jitter = random.uniform(DEFAULT_JITTER_MIN, DEFAULT_JITTER_MAX)
    return base_delay * jitter


def _get_api_keys() -> list[str]:
    """Return a list of API keys to try, primary first, fallback second."""
    primary = get_api_key()
    fallback = get_fallback_api_key()
    keys: list[str] = []
    if primary:
        keys.append(primary)
    if fallback and fallback != primary:
        keys.append(fallback)
    return keys


def make_api_request(
    method: str,
    path: str,
    headers: dict | None = None,
    timeout: int = 20,
    json: dict | None = None,
    params: dict | None = None,
) -> requests.Response:
    url = f"{get_base_url().rstrip('/')}/{path.lstrip('/')}"
    api_keys = _get_api_keys()
    if not api_keys:
        raise EnvironmentError(
            "Missing GROQ_API_KEY (and no GROQ_FALLBACK_API_KEY configured)"
        )

    max_attempts = int(os.getenv("GROQ_API_MAX_RETRIES", DEFAULT_MAX_RETRIES)) + 1
    last_response: requests.Response | None = None
    last_error: Exception | None = None

    for key_index, api_key in enumerate(api_keys):
        key_label = "primary" if key_index == 0 else "fallback"
        request_headers = {"Authorization": f"Bearer {api_key}"}
        if headers:
            request_headers.update(headers)

        for attempt in range(1, max_attempts + 1):
            _RATE_LIMITER.acquire()
            try:
                response = _SESSION.request(
                    method,
                    url,
                    headers=request_headers,
                    timeout=timeout,
                    json=json,
                    params=params,
                )
            except requests.RequestException as exc:
                last_error = exc
                logger.warning(
                    "Request error with %s key (attempt %d/%d): %s",
                    key_label, attempt, max_attempts, exc,
                )
                wait = _compute_backoff(attempt)
                logger.debug("Backing off %.2fs before next attempt", wait)
                time.sleep(wait)
                continue

            last_response = response

            # ── Success ──────────────────────────────────────────
            if response.status_code not in (429, 401, 403):
                response.raise_for_status()
                return response

            # ── Auth failure → skip remaining retries, try next key ──
            if response.status_code in (401, 403):
                logger.warning(
                    "Auth error %d with %s key; switching to next key",
                    response.status_code, key_label,
                )
                break

            # ── Rate-limited (429) → backoff + retry ─────────────
            retry_after = response.headers.get("Retry-After")
            if retry_after and retry_after.isdigit():
                wait = float(retry_after)
            else:
                wait = _compute_backoff(attempt)

            logger.info(
                "Rate-limited with %s key (attempt %d/%d); "
                "backing off %.2fs",
                key_label, attempt, max_attempts, wait,
            )
            time.sleep(wait)
        else:
            # All retries exhausted for this key due to 429s — try next key
            if len(api_keys) > 1 and key_index < len(api_keys) - 1:
                logger.info(
                    "All retries exhausted for %s key; "
                    "falling back to next key",
                    key_label,
                )

    # ── All keys exhausted ───────────────────────────────────────
    if last_response is not None:
        last_response.raise_for_status()
    if last_error is not None:
        raise last_error
    raise RuntimeError("Failed to perform Groq API request")


def validate_api_key():
    response = make_api_request("GET", "/models", timeout=20)
    data = response.json()

    # Optionally validate the fallback key as well
    fallback = get_fallback_api_key()
    if fallback and fallback != get_api_key():
        try:
            fb_headers = {"Authorization": f"Bearer {fallback}"}
            url = f"{get_base_url().rstrip('/')}/models"
            fb_resp = _SESSION.request("GET", url, headers=fb_headers, timeout=20)
            fb_resp.raise_for_status()
            logger.info("Fallback API key validated successfully")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Fallback API key validation failed: %s", exc)

    return data