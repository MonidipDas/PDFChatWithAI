import json
import os
from typing import List, Dict

MEMORY_FILE = "chat_history.json"

def load_memory() -> List[Dict[str, str]]:
    """Loads chat history from a JSON file."""
    if not os.path.exists(MEMORY_FILE):
        return []
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def save_memory(history: List[Dict[str, str]]) -> None:
    """Saves chat history to a JSON file."""
    try:
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=4)
    except Exception as e:
        print(f"Error saving memory: {e}")

def add_interaction(question: str, answer: str) -> None:
    """Adds a single Q&A interaction to memory."""
    history = load_memory()
    history.append({"question": question, "answer": answer})
    save_memory(history)

def get_conversation_context(limit: int = 3) -> str:
    """Gets the last few interactions as a formatted string."""
    history = load_memory()
    if not history:
        return ""
    
    recent_history = history[-limit:]
    context_lines = []
    for h in recent_history:
        context_lines.append(f"User: {h['question']}")
        context_lines.append(f"AI: {h['answer']}")
    
    return "\n".join(context_lines)

def clear_memory() -> None:
    """Clears the chat history."""
    if os.path.exists(MEMORY_FILE):
        os.remove(MEMORY_FILE)
