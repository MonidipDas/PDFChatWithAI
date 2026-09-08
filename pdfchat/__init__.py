"""PDF Chat package."""

from .config import configure_api_key, validate_api_key
from .pdf_processing import extract_text_from_pdf
from .guardrails import (
    check_input_guardrails,
    check_output_guardrails,
    GuardrailResult,
    InputGuardrail,
    HallucinationGuardrail,
    UnsafeOutputGuardrail,
)

__all__ = [
    "configure_api_key",
    "validate_api_key",
    "extract_text_from_pdf",
    "check_input_guardrails",
    "check_output_guardrails",
    "GuardrailResult",
    "InputGuardrail",
    "HallucinationGuardrail",
    "UnsafeOutputGuardrail",
]

