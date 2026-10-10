# embedding and FAISS vectorstore module
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

_embedder = None


def get_embedder(model_name: str = "models/gemini-embedding-001", api_key: str = None):
    """
    Get or initialize the Google Generative AI embeddings model (cached singleton).
    Zero local RAM overhead — embeddings are computed via Google's cloud API.
    """
    global _embedder
    if _embedder is None:
        key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY not found in environment or .env file.")
        _embedder = GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=key)
    return _embedder


def create_vectorstore(chunks, embedder=None, save_dir: str = None):
    """
    Create a FAISS vectorstore from document chunks.
    Optionally save to local disk.
    """
    if embedder is None:
        embedder = get_embedder()

    vectorstore = FAISS.from_documents(chunks, embedder)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        vectorstore.save_local(save_dir)

    return vectorstore


def load_vectorstore(db_dir: str, embedder=None):
    """
    Load an existing FAISS vectorstore from disk.
    """
    if embedder is None:
        embedder = get_embedder()

    if not os.path.exists(db_dir):
        raise FileNotFoundError(f"Vector store directory not found at: {db_dir}")

    vectorstore = FAISS.load_local(
        folder_path=db_dir,
        embeddings=embedder,
        allow_dangerous_deserialization=True
    )
    return vectorstore


def get_retriever(vectorstore, k: int = 5):
    """
    Get a retriever from the vectorstore with top-k search kwargs.
    """
    return vectorstore.as_retriever(search_kwargs={"k": k})