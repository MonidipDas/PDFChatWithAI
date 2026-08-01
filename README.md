# PDFChatWithAI

A small Streamlit project to upload a PDF, create embeddings with Groq, and answer questions from the document.

## Structure

- `app.py`: Streamlit entry point
- `pdfchat/`: application package
  - `config.py`: environment and API configuration
  - `pdf_processing.py`: PDF text extraction
  - `embeddings.py`: vector store creation
  - `qa.py`: prompt and QA chain logic
- `requirements.txt`: Python dependencies

## Setup

1. Create a `.env` file with `GROQ_API_KEY=your_api_key`.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```
