# loading and chunking module
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_pdf(file_path: str):
    """
    Load a PDF document from file_path and return page documents.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found at: {file_path}")
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    return pages


def split_documents(pages, chunk_size: int = 1000, chunk_overlap: int = 150):
    """
    Split loaded document pages into chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ", ", " ", ""]
    )
    chunks = splitter.split_documents(pages)
    return chunks


def load_and_chunk_pdf(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 150):
    """
    Loads PDF and splits it into chunks.
    Returns (chunks, pages).
    """
    pages = load_pdf(file_path)
    chunks = split_documents(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return chunks, pages
