import requests

from langchain.prompts import PromptTemplate

from .config import get_base_url, get_api_key

PROMPT_TEMPLATE = """
You are a helpful assistant. Use the following context to answer the user's question.
Only use information from the context and respond clearly.

Context:
{context}

Question: {question}

Answer:"""

prompt = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"],
)


def format_context(docs):
    return "\n\n".join(getattr(doc, "page_content", str(doc)) for doc in docs)


def get_answer(question: str, vector_store) -> str:
    docs = vector_store.similarity_search(question)
    prompt_text = prompt.format(context=format_context(docs), question=question)

    base = get_base_url().rstrip('/')
    url = f"{base}/chat/completions"
    headers = {
        "Authorization": f"Bearer {get_api_key()}",
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
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=60,
        )

        response.raise_for_status()
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

    except requests.HTTPError as http_err:
        # Likely a 404 or other API-level error — fall back to returning extracted
        # context from the retrieved documents so the app still responds.
        ctx = format_context(docs[:3])
        return f"(Groq API error: {http_err}). Returning extracted context instead:\n\n{ctx}"
    except Exception:
        # Re-raise unexpected exceptions so they can be handled/logged upstream
        raise
