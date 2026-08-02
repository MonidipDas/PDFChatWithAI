import asyncio
import sys
# asyncio event loops help in avoiding runtime errors related to asynchronous operations.

import streamlit as st

from pdfchat.config import configure_api_key, validate_api_key
from pdfchat.embeddings import create_vector_store
from pdfchat.pdf_processing import extract_text_from_pdf
from pdfchat.qa import get_answer


def ensure_event_loop() -> None:
    if sys.platform.startswith("win"):
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        except Exception:
            pass

    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())


def main() -> None:
    ensure_event_loop()
    st.set_page_config(page_title="PDF Chat with Groq", layout="wide")
    st.title("📚 Ask Queries from your pdf")

    try:
        configure_api_key()
        validate_api_key()
        st.success("✅ Groq API key is valid and authenticated!")
    except Exception as exc:
        st.error(f"❌ Groq API key authentication failed: {exc}")
        return

    uploaded_pdf = st.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded_pdf is None:
        st.info("Upload a PDF to begin.")
        return

    with st.spinner("Reading and processing PDF..."):
        try:
            text = extract_text_from_pdf(uploaded_pdf)
            vector_store = create_vector_store(text)
            st.success("PDF processed successfully! Ask your questions below.")
        except Exception as exc:
            st.error(f"Error processing PDF: {exc}")
            return

    question = st.text_input("❓ Ask a question about the PDF:")
    if question:
        with st.spinner("Thinking..."):
            answer = get_answer(question, vector_store)
        st.write("💬 Answer:", answer)


if __name__ == "__main__":
    main()
