from typing import BinaryIO

from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError


def extract_text_from_pdf(pdf_file: BinaryIO) -> str:
    if not pdf_file:
        return ""

    try:
        reader = PdfReader(pdf_file)
        return "".join(page.extract_text() or "" for page in reader.pages)
    except PdfReadError as exc:
        raise ValueError("Invalid or corrupted PDF file") from exc
