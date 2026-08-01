from .config import make_api_request

PROMPT_TEMPLATE = """
You are a helpful assistant. Use the following context to answer the user's question.
Only use information from the context and respond clearly.

Context:
{context}

Question: {question}

Answer:"""


def format_context(docs):
    return "\n\n".join(getattr(doc, "page_content", str(doc)) for doc in docs)


def get_answer(question: str, vector_store) -> str:
    docs = vector_store.similarity_search(question)
    prompt_text = PROMPT_TEMPLATE.format(context=format_context(docs), question=question)

    headers = {
        "Content-Type": "application/json",
    }
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "user", "content": prompt_text}
        ],
        "temperature": 0.2,
        "max_tokens": 512,
    }

    try:
        response = make_api_request(
            "POST",
            "/chat/completions",
            headers=headers,
            json=payload,
            timeout=60,
        )
        data = response.json()

        # Standard Chat Completions-style response handling
        choices = data.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            content = message.get("content")
            if isinstance(content, str):
                return content.strip()

        # If the response format doesn't match, return raw JSON as fallback
        return str(data)

    except Exception as exc:
        if isinstance(exc, Exception):
            ctx = format_context(docs[:3])
            return f"(Groq API error: {exc}). Returning extracted context instead:\n\n{ctx}"
        raise
