from typing import List, Optional
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.documents import Document

_cross_encoder_model = None
_reranker = None


def get_reranker(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", top_n: int = 3):
    """
    Get or initialize cached HuggingFace CrossEncoder and CrossEncoderReranker.
    """
    global _cross_encoder_model, _reranker
    if _cross_encoder_model is None:
        _cross_encoder_model = HuggingFaceCrossEncoder(model_name=model_name)
    if _reranker is None or _reranker.top_n != top_n:
        _reranker = CrossEncoderReranker(model=_cross_encoder_model, top_n=top_n)
    return _reranker


def get_cross_encoder_retriever(base_retriever, top_n: int = 3):
    """
    Wrap a base retriever (e.g. FAISS retriever) with a ContextualCompressionRetriever
    that uses CrossEncoderReranker to score and rank top_n most relevant chunks.
    """
    compressor = get_reranker(top_n=top_n)
    compression_retriever = ContextualCompressionRetriever(
        base_retriever=base_retriever,
        base_compressor=compressor
    )
    return compression_retriever


def cross_encoder_retriever(base_retriever, top_n: int = 3):
    """
    Alias for get_cross_encoder_retriever.
    """
    return get_cross_encoder_retriever(base_retriever=base_retriever, top_n=top_n)


def rerank_documents(docs: List[Document], query: str, top_n: int = 3) -> List[Document]:
    """
    Directly score and rerank a pre-retrieved list of Document objects using the cross-encoder.
    Particularly useful for multi-part decomposed queries after merging context.
    """
    if not docs:
        return []
    compressor = get_reranker(top_n=top_n)
    return compressor.compress_documents(docs, query)