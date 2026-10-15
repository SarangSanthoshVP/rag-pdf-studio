import os
import shutil
import json
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.loading_chunking import load_and_chunk_pdf
from backend.embedding_faiss import create_vectorstore, load_vectorstore, get_retriever, get_embedder
from backend.reranker import get_cross_encoder_retriever
from backend.rag_chain import create_rag_chain, generate_answer_from_docs
from backend.query_router import route_and_retrieve
from backend.stt_tts import transcribe_audio, text_to_speech

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")
FAISS_DIR = os.path.join(DATA_DIR, "faiss_index")

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(FAISS_DIR, exist_ok=True)

# In-memory per-session pipeline state: session_id -> state dict
# Each session has isolated vectorstore, retrievers, chain, and metadata
user_sessions: Dict[str, Dict[str, Any]] = {}


def sanitize_session_id(session_id: Optional[str]) -> str:
    """Sanitize session_id to alphanumeric/dash/underscore to prevent path traversal."""
    if not session_id:
        return "default"
    clean = "".join(c for c in session_id if c.isalnum() or c in ("-", "_")).strip()
    return clean or "default"


def get_session_uploads_dir(clean_session_id: str) -> str:
    path = os.path.join(UPLOADS_DIR, clean_session_id)
    os.makedirs(path, exist_ok=True)
    return path


def get_or_load_session(session_id: Optional[str]) -> Dict[str, Any]:
    """
    Retrieve in-memory session state or restore it from local FAISS disk if present.
    Ensures full session isolation so one user never sees another's active document.
    """
    clean_id = sanitize_session_id(session_id)

    # Return cached in-memory state if active
    if clean_id in user_sessions and user_sessions[clean_id].get("chain") is not None:
        return user_sessions[clean_id]

    session_faiss_dir = os.path.join(FAISS_DIR, clean_id)
    meta_path = os.path.join(session_faiss_dir, "metadata.json")

    # If this specific session has a saved FAISS index on disk, restore it
    if os.path.exists(os.path.join(session_faiss_dir, "index.faiss")):
        try:
            vectorstore = load_vectorstore(session_faiss_dir)
            base_retriever = get_retriever(vectorstore, k=5)
            reranked_retriever = get_cross_encoder_retriever(base_retriever, top_n=3)
            chain = create_rag_chain(reranked_retriever)

            doc_name = "Indexed Document"
            total_pages = None
            total_chunks = None

            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                        doc_name = meta.get("document_name", doc_name)
                        total_pages = meta.get("total_pages")
                        total_chunks = meta.get("total_chunks")
                except Exception:
                    pass

            user_sessions[clean_id] = {
                "vectorstore": vectorstore,
                "base_retriever": base_retriever,
                "retriever": reranked_retriever,
                "chain": chain,
                "document_name": doc_name,
                "total_pages": total_pages,
                "total_chunks": total_chunks,
            }
            print(f"[Session Restored] Session '{clean_id}' successfully loaded '{doc_name}' from disk.", flush=True)
            return user_sessions[clean_id]
        except Exception as e:
            print(f"[Session Warning] Failed to restore session '{clean_id}' from disk: {e}", flush=True)

    # Empty session default
    empty_state = {
        "vectorstore": None,
        "base_retriever": None,
        "retriever": None,
        "chain": None,
        "document_name": None,
        "total_pages": None,
        "total_chunks": None,
    }
    user_sessions[clean_id] = empty_state
    return empty_state


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm embedder on startup
    try:
        get_embedder()
        print("Embedder pre-warmed successfully with resilient backoff.", flush=True)
    except Exception as e:
        print(f"Startup embedder notice: {e}", flush=True)
    yield


app = FastAPI(
    title="RAG Project API",
    description="FastAPI backend for multi-user session-isolated PDF Q&A using LangChain, FAISS, and Gemini",
    version="2.0.0",
    lifespan=lifespan
)

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


class SourceItem(BaseModel):
    content: str
    metadata: dict


class QueryResponse(BaseModel):
    query: str
    answer: str
    document_name: Optional[str] = None
    query_type: Optional[str] = None
    sub_queries: Optional[List[str]] = []
    sub_query_evidence: Optional[Dict[str, Any]] = {}
    sources: List[SourceItem] = []


@app.get("/")
def read_root(session_id: Optional[str] = None):
    clean_id = sanitize_session_id(session_id)
    state = get_or_load_session(clean_id)
    return {
        "status": "online",
        "session_id": clean_id,
        "has_document": state["chain"] is not None,
        "active_document": state["document_name"]
    }


@app.get("/status")
def get_status(session_id: Optional[str] = None):
    """
    Returns active document status isolated strictly to the requesting session_id.
    New visitors without an uploaded document receive has_document=false.
    """
    clean_id = sanitize_session_id(session_id)
    state = get_or_load_session(clean_id)
    return {
        "session_id": clean_id,
        "has_document": state["chain"] is not None,
        "active_document": state["document_name"],
        "total_pages": state.get("total_pages"),
        "total_chunks": state.get("total_chunks")
    }


@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None)
):
    """
    1. Isolates upload to session-specific directory.
    2. Chunks and computes Gemini embeddings.
    3. Saves per-session FAISS index to disk.
    4. Configures session-isolated retriever and RAG chain.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    clean_id = sanitize_session_id(session_id)
    user_uploads_dir = get_session_uploads_dir(clean_id)
    session_faiss_dir = os.path.join(FAISS_DIR, clean_id)
    os.makedirs(session_faiss_dir, exist_ok=True)

    file_path = os.path.join(user_uploads_dir, file.filename)

    # Save uploaded file in session folder
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # 1. Load and chunk
        chunks, pages = load_and_chunk_pdf(file_path)
        if not chunks:
            raise HTTPException(status_code=400, detail="No readable text extracted from the PDF.")

        # 2. Embed & store into isolated session directory
        vectorstore = create_vectorstore(chunks, save_dir=session_faiss_dir)

        # 3. Save metadata to disk for persistence across restarts/reloads
        meta_path = os.path.join(session_faiss_dir, "metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "document_name": file.filename,
                "total_pages": len(pages),
                "total_chunks": len(chunks)
            }, f)

        # 4. Create base retriever and wrap with CrossEncoder reranker (with safe fallback)
        base_retriever = get_retriever(vectorstore, k=5)
        reranked_retriever = get_cross_encoder_retriever(base_retriever, top_n=3)
        chain = create_rag_chain(reranked_retriever)

        # Update session state
        user_sessions[clean_id] = {
            "vectorstore": vectorstore,
            "base_retriever": base_retriever,
            "retriever": reranked_retriever,
            "chain": chain,
            "document_name": file.filename,
            "total_pages": len(pages),
            "total_chunks": len(chunks)
        }

        return {
            "status": "success",
            "message": f"Successfully processed '{file.filename}' for your session",
            "filename": file.filename,
            "total_pages": len(pages),
            "total_chunks": len(chunks),
            "session_id": clean_id
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")


@app.post("/ask", response_model=QueryResponse)
def ask(payload: QueryRequest):
    """
    Accepts user question:
    1. Looks up the requesting user's isolated session.
    2. Classifies query type (single_fact, multi_part, summarization).
    3. Decomposes multi-part queries if needed and retrieves session evidence.
    4. Merges context, applies reranking, and generates grounded answer.
    """
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    clean_id = sanitize_session_id(payload.session_id)
    state = get_or_load_session(clean_id)

    if state.get("retriever") is None:
        raise HTTPException(
            status_code=400,
            detail="No document has been uploaded for your session yet. Please upload a PDF first."
        )

    try:
        # Route, decompose (if multi_part), retrieve, and rerank using session's retriever
        routed_result = route_and_retrieve(query, state["retriever"], use_reranker=True, top_n=3)

        merged_docs = routed_result["merged_docs"]
        query_type = routed_result["query_type"]
        sub_queries = routed_result["sub_queries"]
        evidence = routed_result["evidence_by_sub_query"]

        # Generate answer from merged context
        answer = generate_answer_from_docs(merged_docs, query)

        sources = [
            SourceItem(content=doc.page_content, metadata=doc.metadata)
            for doc in merged_docs
        ]

        return QueryResponse(
            query=query,
            answer=answer,
            document_name=state["document_name"],
            query_type=query_type,
            sub_queries=sub_queries,
            sub_query_evidence=evidence,
            sources=sources
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")


@app.post("/reset_session")
def reset_session(session_id: Optional[str] = None):
    """
    Clears the active session index and files, returning to an empty knowledge base.
    """
    clean_id = sanitize_session_id(session_id)
    if clean_id in user_sessions:
        del user_sessions[clean_id]

    session_faiss_dir = os.path.join(FAISS_DIR, clean_id)
    if os.path.exists(session_faiss_dir):
        try:
            shutil.rmtree(session_faiss_dir)
        except Exception:
            pass

    return {
        "status": "success",
        "message": "Session reset successfully. You can now upload a new document.",
        "session_id": clean_id
    }


class TTSRequest(BaseModel):
    text: str
    lang: Optional[str] = "en"


@app.post("/stt")
async def speech_to_text(file: UploadFile = File(...)):
    """
    Speech-to-Text: Converts user spoken voice audio into text query using Gemini multimodal STT.
    """
    try:
        audio_bytes = await file.read()
        mime_type = file.content_type or "audio/webm"
        transcribed_text = transcribe_audio(audio_bytes, mime_type=mime_type)
        return {"text": transcribed_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Speech transcription failed: {str(e)}")


@app.post("/tts")
def text_to_speech_endpoint(payload: TTSRequest):
    """
    Text-to-Speech: Converts generated AI response into spoken audio (MP3).
    Only invoked when user clicks the Listen button below a response.
    """
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    try:
        audio_bytes = text_to_speech(text, lang=payload.lang or "en")
        return Response(content=audio_bytes, media_type="audio/mp3")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS conversion failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)