import streamlit as st
from typing import List, Dict

SESSION_KEY = "chat_history"

def _get_history() -> List[Dict[str, str]]:
    """Returns the conversation history list from session state."""
    if SESSION_KEY not in st.session_state:
        st.session_state[SESSION_KEY] = []
    return st.session_state[SESSION_KEY]

def load_memory() -> List[Dict[str, str]]:
    """Loads chat history from the current session."""
    return list(_get_history())

def save_memory(history: List[Dict[str, str]]) -> None:
    """Replaces the session chat history."""
    st.session_state[SESSION_KEY] = history

def add_interaction(question: str, answer: str) -> None:
    """Adds a single Q&A interaction to session memory."""
    _get_history().append({"question": question, "answer": answer})

def get_conversation_context(limit: int = 3) -> str:
    """Gets the last few interactions as a formatted string."""
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
    """Clears the chat history for the current session."""
    st.session_state[SESSION_KEY] = []
