# embedding and FAISS vectorstore module
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

_embedder = None


def get_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Get or initialize the HuggingFace embeddings model (cached).
    """
    global _embedder
    if _embedder is None:
        _embedder = HuggingFaceEmbeddings(model=model_name)
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