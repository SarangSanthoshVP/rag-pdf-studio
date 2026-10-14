import os
from typing import List, Optional
from langchain_core.documents import Document

_cross_encoder_model = None
_reranker = None
_cross_encoder_available = None


def is_cross_encoder_available() -> bool:
    """
    Check if sentence-transformers and PyTorch are installed and usable.
    """
    global _cross_encoder_available
    if _cross_encoder_available is None:
        try:
            import sentence_transformers
            _cross_encoder_available = True
        except ImportError:
            _cross_encoder_available = False
            print("[Reranker Notice] sentence-transformers not installed. Reranker will gracefully fall back to base retriever.", flush=True)
        except Exception as e:
            _cross_encoder_available = False
            print(f"[Reranker Notice] Could not initialize sentence-transformers ({e}). Falling back.", flush=True)
    return _cross_encoder_available


def get_reranker(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", top_n: int = 3):
    """
    Get or initialize cached HuggingFace CrossEncoder and CrossEncoderReranker.
    Returns None if sentence-transformers is not available.
    """
    global _cross_encoder_model, _reranker
    if not is_cross_encoder_available():
        return None

    try:
        from langchain_community.cross_encoders import HuggingFaceCrossEncoder
        from langchain_classic.retrievers.document_compressors import CrossEncoderReranker

        if _cross_encoder_model is None:
            _cross_encoder_model = HuggingFaceCrossEncoder(model_name=model_name)
        if _reranker is None or _reranker.top_n != top_n:
            _reranker = CrossEncoderReranker(model=_cross_encoder_model, top_n=top_n)
        return _reranker
    except Exception as e:
        print(f"[Reranker Notice] Failed to load CrossEncoder ({e}). Falling back to base retriever.", flush=True)
        return None


def get_cross_encoder_retriever(base_retriever, top_n: int = 3):
    """
    Wrap a base retriever (e.g. FAISS retriever) with ContextualCompressionRetriever
    using CrossEncoderReranker. If unavailable, safely returns the base retriever.
    """
    compressor = get_reranker(top_n=top_n)
    if compressor is None:
        return base_retriever

    try:
        from langchain_classic.retrievers import ContextualCompressionRetriever
        return ContextualCompressionRetriever(
            base_retriever=base_retriever,
            base_compressor=compressor
        )
    except Exception as e:
        print(f"[Reranker Notice] Failed to create compression retriever ({e}). Returning base retriever.", flush=True)
        return base_retriever


def cross_encoder_retriever(base_retriever, top_n: int = 3):
    """
    Alias for get_cross_encoder_retriever.
    """
    return get_cross_encoder_retriever(base_retriever=base_retriever, top_n=top_n)


def rerank_documents(docs: List[Document], query: str, top_n: int = 3) -> List[Document]:
    """
    Directly score and rerank a pre-retrieved list of Document objects using the cross-encoder.
    If unavailable, safely returns the top_n documents.
    """
    if not docs:
        return []
    compressor = get_reranker(top_n=top_n)
    if compressor is None:
        return docs[:top_n]

    try:
        return compressor.compress_documents(docs, query)
    except Exception as e:
        print(f"[Reranker Notice] Compression error ({e}). Returning base docs.", flush=True)
        return docs[:top_n]