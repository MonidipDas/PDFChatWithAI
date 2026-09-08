import streamlit as st
from typing import List, Dict


def _get_user_id() -> str:
    """Return the current user's ID from session state, or 'default'."""
    return st.session_state.get("current_user_id", "default")


def _session_key() -> str:
    """Return the user-scoped session key for chat history."""
    return f"chat_history_{_get_user_id()}"


def _get_history() -> List[Dict[str, str]]:
    """Returns the conversation history list from session state for the current user."""
    key = _session_key()
    if key not in st.session_state:
        st.session_state[key] = []
    return st.session_state[key]

def load_memory() -> List[Dict[str, str]]:
    """Loads chat history from the current session for the current user."""
    return list(_get_history())

def save_memory(history: List[Dict[str, str]]) -> None:
    """Replaces the session chat history for the current user."""
    st.session_state[_session_key()] = history

def add_interaction(question: str, answer: str) -> None:
    """Adds a single Q&A interaction to the current user's session memory."""
    _get_history().append({"question": question, "answer": answer})

def get_conversation_context(limit: int = 3) -> str:
    """Gets the last few interactions as a formatted string for the current user."""
    history = _get_history()
    if not history:
        return ""

    recent_history = history[-limit:]
    context_lines = []
    for h in recent_history:
        context_lines.append(f"User: {h['question']}")
        context_lines.append(f"AI: {h['answer']}")

    return "\n".join(context_lines)

def clear_memory() -> None:
    """Clears the chat history for the current user's session."""
    st.session_state[_session_key()] = []
