import os

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


def get_model_candidates():
    configured = os.getenv("GROQ_MODEL") or os.getenv("GROQ_MODELS") or ""
    candidates = []
    if configured:
        candidates.extend(
            model.strip() for model in configured.split(",") if model.strip()
        )

    candidates.extend([
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
    ])

    return list(dict.fromkeys(candidates))


def _get_error_status_code(exc):
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None)


def get_answer(question: str, vector_store) -> str:
    docs = vector_store.similarity_search(question)
    prompt_text = PROMPT_TEMPLATE.format(context=format_context(docs), question=question)

    headers = {
        "Content-Type": "application/json",
    }

    last_error = None
    for model_name in get_model_candidates():
        payload = {
            "model": model_name,
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

            choices = data.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                content = message.get("content")
                if isinstance(content, str):
                    return content.strip()

            return str(data)
        except Exception as exc:
            last_error = exc
            if _get_error_status_code(exc) not in (404,):
                break

    ctx = format_context(docs[:3])
    if last_error is not None:
        return f"(Groq API error: {last_error}). Returning extracted context instead:\n\n{ctx}"

    return ctx
