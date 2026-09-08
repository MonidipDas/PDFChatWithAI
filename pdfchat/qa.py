import logging
import os

from .config import make_api_request

try:
    from .guardrails import (
        check_input_guardrails,
        check_output_guardrails,
        FALLBACK_INJECTION,
        FALLBACK_HALLUCINATION,
        FALLBACK_UNSAFE,
    )
    _GUARDRAILS_AVAILABLE = True
except Exception:  # noqa: BLE001
    _GUARDRAILS_AVAILABLE = False

    # No-op stubs so the app still works without guardrails
    def check_input_guardrails(question):
        return []

    def check_output_guardrails(answer, context, question):
        return []

    FALLBACK_INJECTION = "Could not process your question."
    FALLBACK_HALLUCINATION = "The answer could not be verified."
    FALLBACK_UNSAFE = "The response was blocked."

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """
You are a helpful assistant. Use the following context to answer the user's question.
Only use information from the context and respond clearly.
If previous conversation history is provided, you may use it for context but prioritize the PDF context.

Previous Conversation:
{history}

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
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
    ])

    return list(dict.fromkeys(candidates))


def _get_error_status_code(exc):
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None)


from pdfchat.memory import get_conversation_context, add_interaction

def get_answer(question: str, retriever, *, _skip_guardrails: bool = False) -> str:
    """Get an answer for *question* using the retriever.

    Runs input and output guardrails unless *_skip_guardrails* is ``True``.
    Returns just the answer string for backward compatibility.
    Use :func:`get_answer_with_guardrails` to also receive guardrail details.
    """
    answer, _results = get_answer_with_guardrails(
        question, retriever, _skip_guardrails=_skip_guardrails,
    )
    return answer


def get_answer_with_guardrails(
    question: str,
    retriever,
    *,
    _skip_guardrails: bool = False,
) -> tuple:
    """Get an answer for *question* and return ``(answer, guardrail_results)``.

    *guardrail_results* is a dict with keys ``"input"`` and ``"output"``,
    each containing a list of guardrail result objects.
    """
    guardrail_results = {"input": [], "output": []}

    # ── Input guardrails ─────────────────────────────────────────────
    if not _skip_guardrails:
        input_results = check_input_guardrails(question)
        guardrail_results["input"] = input_results

        for gr in input_results:
            if not gr.passed:
                logger.warning(
                    "Input guardrail BLOCKED [%s]: %s", gr.category, gr.details,
                )
                add_interaction(question, FALLBACK_INJECTION)
                return FALLBACK_INJECTION, guardrail_results

    # ── Retrieve context & build prompt ──────────────────────────────
    docs = retriever.invoke(question)
    context = format_context(docs)
    history = get_conversation_context()
    prompt_text = PROMPT_TEMPLATE.format(
        history=history, context=context, question=question,
    )

    headers = {"Content-Type": "application/json"}

    # ── Call LLM ─────────────────────────────────────────────────────
    last_error = None
    answer = None

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
                    answer = content.strip()
                    break

            answer = str(data)
            break
        except Exception as exc:
            last_error = exc
            if _get_error_status_code(exc) not in (404,):
                break

    # Fallback when LLM call fails
    if answer is None:
        ctx = format_context(docs[:3])
        if last_error is not None:
            answer = f"(Groq API error: {last_error}). Returning extracted context instead:\n\n{ctx}"
        else:
            answer = ctx

    # ── Output guardrails ────────────────────────────────────────────
    if not _skip_guardrails:
        output_results = check_output_guardrails(answer, context, question)
        guardrail_results["output"] = output_results

        for gr in output_results:
            if not gr.passed:
                logger.warning(
                    "Output guardrail BLOCKED [%s]: %s", gr.category, gr.details,
                )
                if gr.category == "hallucination":
                    fallback = FALLBACK_HALLUCINATION
                elif gr.category == "unsafe_output":
                    fallback = FALLBACK_UNSAFE
                else:
                    fallback = FALLBACK_UNSAFE

                add_interaction(question, fallback)
                return fallback, guardrail_results

    add_interaction(question, answer)
    return answer, guardrail_results

