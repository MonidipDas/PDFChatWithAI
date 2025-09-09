import asyncio
import sys

# Ensure an event loop exists for async libraries (fixes Streamlit thread error)
if sys.platform.startswith("win"):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

import os
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores.faiss import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Load environment variables and configure Google API key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

# Test API key authentication
try:
    models = genai.list_models()
    st.success("✅ Google API key is valid and authenticated!")
except Exception as e:
    st.error(f"❌ Google API key authentication failed: {e}")

# 🧠 Custom prompt template
prompt_template = """
You are a helpful assistant. Use the following context to answer the user's question.
Only use information from the context and respond clearly.

Context:
{context}

Question: {question}

Answer:"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# 🧠 Load QA chain
def get_qa_chain():
    llm = ChatGoogleGenerativeAI(model="models/chat-bison-001", temperature=0.2)
    return load_qa_chain(llm=llm, chain_type="stuff", prompt=prompt)

# 📄 Extract text from PDF
from PyPDF2.errors import PdfReadError

def extract_text_from_pdf(pdf_file):
    try:
        reader = PdfReader(pdf_file)
        raw_text = ""
        for page in reader.pages:
            raw_text += page.extract_text() or ""
        return raw_text
    except PdfReadError:
        st.warning(f"Could not read {getattr(pdf_file, 'name', 'uploaded file')}. It may be corrupted or not a valid PDF.")
        return ""
    except Exception as e:
        st.error(f"Error reading {getattr(pdf_file, 'name', 'uploaded file')}: {e}")
        return ""

# 🔍 Create FAISS vectorstore
def create_vector_store(text):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-gecko-001")
    return FAISS.from_texts(chunks, embedding=embeddings)

# 💬 Handle user question
def handle_question(question, vector_store, qa_chain):
    docs = vector_store.similarity_search(question)
    return qa_chain.run(input_documents=docs, question=question)

# 🖼️ Streamlit UI
def main():
    st.set_page_config(page_title="PDF Chat with GenAI", layout="wide")
    st.title("📚 Chat with your PDF using Google Generative AI")

    uploaded_pdf = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_pdf is not None:
        with st.spinner("Reading and processing PDF..."):
            text = extract_text_from_pdf(uploaded_pdf)
            vector_store = create_vector_store(text)
            qa_chain = get_qa_chain()
        st.success("PDF processed successfully! Ask your questions below.")

        question = st.text_input("❓ Ask a question about the PDF:")
        if question:
            with st.spinner("Thinking..."):
                answer = handle_question(question, vector_store, qa_chain)
            st.write("💬 Answer:", answer)

if __name__ == "__main__":
    main()
