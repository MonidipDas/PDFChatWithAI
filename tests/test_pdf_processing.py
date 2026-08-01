import io

import pytest

from pdfchat.pdf_processing import extract_text_from_pdf


def test_extract_text_from_pdf_invalid_pdf_raises_value_error():
    fake_pdf = io.BytesIO(b"not a pdf")

    with pytest.raises(ValueError):
        extract_text_from_pdf(fake_pdf)
