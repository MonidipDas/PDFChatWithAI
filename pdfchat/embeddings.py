
# CharacterTextSplitter will help me to split the text into chunks.
# HuggingFaceEmbeddings will help me to create numerical vector embeddings for the text chunks.
# FAISS will help me to store and search(similarity search) the vector embeddings efficiently.
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker


def create_vector_store(text: str):
    if not text:
        raise ValueError("No text available to embed.")
    
    # Max No of characters in a chunk is 1000 and two consecutive have 200 characters common.
    # By overlapping the chunks, we can ensure that the context is preserved across chunks.
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    
    # Every chunk is embedded into a vector representation using the HuggingFaceEmbeddings model.
    # Vectors are stored in FAISS Vector Store for efficient similarity search.
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    faiss_retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    # BM25 Retriever for sparse hybrid search
    bm25_retriever = BM25Retriever.from_texts(chunks)
    bm25_retriever.k = 5

    # Ensemble Retriever combining dense and sparse search
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5]
    )

    # Cross Encoder Reranker to improve search results relevance
    model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    compressor = CrossEncoderReranker(model=model, top_n=3)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=ensemble_retriever
    )

    return compression_retriever
