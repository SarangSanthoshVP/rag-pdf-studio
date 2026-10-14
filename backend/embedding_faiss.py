# embedding and FAISS vectorstore module
import os
import re
import time
import random
from typing import List
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

_embedder = None


class ResilientGoogleEmbeddings(Embeddings):
    """
    Wrapper around GoogleGenerativeAIEmbeddings providing:
    1. Batching to stay well under Google API limits (e.g. 25 chunks per batch).
    2. Exponential backoff with jitter on 429 / RESOURCE_EXHAUSTED errors.
    3. Intelligent extraction of Google's suggested cooldown duration (e.g. 'retry in 22.8s').
    """

    def __init__(
        self,
        base_embedder: GoogleGenerativeAIEmbeddings,
        batch_size: int = 25,
        max_retries: int = 6,
        initial_delay: float = 3.0,
        max_delay: float = 65.0,
        inter_batch_pause: float = 0.5
    ):
        self.base_embedder = base_embedder
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.inter_batch_pause = inter_batch_pause

    def _execute_with_backoff(self, operation, desc: str = "Embedding"):
        attempt = 0
        while True:
            try:
                return operation()
            except Exception as err:
                attempt += 1
                err_str = str(err)
                is_quota_error = (
                    "429" in err_str
                    or "RESOURCE_EXHAUSTED" in err_str
                    or "quota" in err_str.lower()
                    or "rate" in err_str.lower()
                )

                if attempt > self.max_retries or not is_quota_error:
                    print(f"[{desc} Error] Permanent failure or max retries ({self.max_retries}) reached: {err}", flush=True)
                    raise err

                # Check if Google provided a specific retry delay
                match = re.search(r"retry in (\d+(?:\.\d+)?)s", err_str, re.IGNORECASE)
                if match:
                    wait_time = float(match.group(1)) + 1.5
                else:
                    # Exponential backoff with random jitter
                    wait_time = min(
                        self.initial_delay * (2 ** (attempt - 1)) + random.uniform(0.5, 1.5),
                        self.max_delay
                    )

                print(
                    f"[RateLimit Backoff] {desc} encountered 429/Quota limit on attempt {attempt}/{self.max_retries}. "
                    f"Waiting {wait_time:.1f}s before retrying...",
                    flush=True
                )
                time.sleep(wait_time)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds texts in batches of self.batch_size with exponential backoff on rate limits.
        """
        all_embeddings: List[List[float]] = []
        total_texts = len(texts)
        total_batches = (total_texts + self.batch_size - 1) // self.batch_size

        print(f"[Embedding] Starting embedding of {total_texts} chunks across {total_batches} batches...", flush=True)

        for batch_idx in range(total_batches):
            start = batch_idx * self.batch_size
            end = min(start + self.batch_size, total_texts)
            batch = texts[start:end]
            batch_num = batch_idx + 1

            batch_embs = self._execute_with_backoff(
                lambda: self.base_embedder.embed_documents(batch),
                desc=f"Batch {batch_num}/{total_batches}"
            )
            all_embeddings.extend(batch_embs)
            print(f"[Embedding] Batch {batch_num}/{total_batches} ({len(batch)} chunks) embedded successfully.", flush=True)

            # Slight pause between batches to avoid bursting free-tier RPM
            if batch_idx < total_batches - 1 and self.inter_batch_pause > 0:
                time.sleep(self.inter_batch_pause)

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        Embeds a single query string with retry on rate limit.
        """
        return self._execute_with_backoff(
            lambda: self.base_embedder.embed_query(text),
            desc="Query embedding"
        )


def get_embedder(model_name: str = "models/gemini-embedding-001", api_key: str = None):
    """
    Get or initialize the Google Generative AI embeddings model (cached singleton)
    wrapped with ResilientGoogleEmbeddings for automatic backoff & retry.
    """
    global _embedder
    if _embedder is None:
        key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY not found in environment or .env file.")
        base = GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=key)
        _embedder = ResilientGoogleEmbeddings(base_embedder=base, batch_size=25, max_retries=6)
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