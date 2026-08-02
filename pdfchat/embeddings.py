
# CharacterTextSplitter will help me to split the text into chunks.
# HuggingFaceEmbeddings will help me to create numerical vector embeddings for the text chunks.
# FAISS will help me to store and search(similarity search) the vector embeddings efficiently.
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def create_vector_store(text: str) -> FAISS:
    if not text:
        raise ValueError("No text available to embed.")
    
    # Max No of characters in a chunk is 1000 and two consecutive have 200 characters common.
    # By overlapping the chunks, we can ensure that the context is preserved across chunks.
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    
    # Every chunk is embedded into a vector representation using the HuggingFaceEmbeddings model.
    # Vectors are stored in FAISS Vector Store for efficient similarity search.
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(chunks, embedding=embeddings)
