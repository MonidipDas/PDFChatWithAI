import asyncio
import sys
import pandas as pd
import plotly.express as px
import streamlit as st

from pdfchat.config import configure_api_key, validate_api_key
from pdfchat.embeddings import create_vector_store
from pdfchat.pdf_processing import extract_text_from_pdf
from pdfchat.qa import get_answer
from pdfchat.memory import load_memory, clear_memory


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


def apply_custom_css():
    st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(to bottom right, #1e1e2f, #252542);
            color: #ffffff;
            font-family: 'Inter', sans-serif;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: transparent;
            border-radius: 4px 4px 0px 0px;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #3b3b58;
            color: #fca311 !important;
        }
        .stButton>button {
            background-color: #fca311;
            color: #14213d;
            border: none;
            border-radius: 8px;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #e5980b;
            transform: scale(1.05);
        }
        .stTextInput>div>div>input {
            border-radius: 8px;
            border: 1px solid #4a4a6a;
            background-color: #2e2e48;
            color: #ffffff;
        }
        h1, h2, h3 {
            color: #fca311;
        }
        </style>
    """, unsafe_allow_html=True)


def tab_chat(retriever):
    st.subheader("💬 Chat with PDF")
    question = st.text_input("❓ Ask a question about the PDF:", key="chat_input")
    if question:
        with st.spinner("Thinking..."):
            answer = get_answer(question, retriever)
        
        st.markdown(f"**You:** {question}")
        st.info(f"**AI:** {answer}")


def tab_memory():
    st.subheader("🧠 Agentic Memory (Context)")
    st.write("This tab displays the conversation history that the AI is actively using to retain context.")
    history = load_memory()
    
    if not history:
        st.write("No memory stored yet.")
    else:
        for idx, item in enumerate(history):
            with st.expander(f"Interaction {idx + 1}: {item['question']}", expanded=False):
                st.write("**Answer:**")
                st.write(item['answer'])
                
    if st.button("Clear Memory"):
        clear_memory()
        st.success("Memory cleared!")
        st.rerun()


def tab_dashboard():
    st.subheader("📊 Semantic Chunking & Reranking Accuracy")
    st.write("Simulated graphical representation of retrieval accuracy improvements.")
    
    # Create sample accuracy data
    data = {
        "Strategy": ["Base FAISS (Dense)", "BM25 (Sparse)", "Ensemble (Hybrid)", "Cross-Encoder (Reranked)"],
        "Accuracy Score": [0.65, 0.58, 0.78, 0.92]
    }
    df = pd.DataFrame(data)
    
    fig = px.bar(
        df, 
        x="Strategy", 
        y="Accuracy Score", 
        color="Strategy",
        title="Retrieval Accuracy by Strategy",
        text_auto=True,
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white")
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.caption("* Note: These are simulated scores to demonstrate the accuracy impact of hybrid search and reranking on semantic chunks.")


def main() -> None:
    ensure_event_loop()
    st.set_page_config(page_title="Agentic PDF Chat", layout="wide", initial_sidebar_state="expanded")
    apply_custom_css()
    
    st.title("📚 Agentic PDF Explorer")

    with st.sidebar:
        st.header("⚙️ Configuration")
        try:
            configure_api_key()
            validate_api_key()
            st.success("✅ Groq API Authenticated")
        except Exception as exc:
            st.error(f"❌ API Error: {exc}")
            return
            
        st.divider()
        st.subheader("📄 Document Upload")
        uploaded_pdf = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_pdf is None:
        st.info("👈 Please upload a PDF in the sidebar to begin.")
        return

    with st.spinner("Processing PDF with Hybrid Search and Reranking..."):
        try:
            # We would ideally cache this retriever creation in Streamlit session state
            # but for simplicity we keep it here.
            text = extract_text_from_pdf(uploaded_pdf)
            retriever = create_vector_store(text)
        except Exception as exc:
            st.error(f"Error processing PDF: {exc}")
            return

    # Create Tabs
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "🧠 Agentic Memory", "📊 Accuracy Dashboard"])
    
    with tab1:
        tab_chat(retriever)
        
    with tab2:
        tab_memory()
        
    with tab3:
        tab_dashboard()


if __name__ == "__main__":
    main()
