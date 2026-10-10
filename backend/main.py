import os
import shutil
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.loading_chunking import load_and_chunk_pdf
from backend.embedding_faiss import create_vectorstore, load_vectorstore, get_retriever, get_embedder
from backend.rag_chain import create_rag_chain, ask_question

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")
FAISS_DIR = os.path.join(DATA_DIR, "faiss_index")

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(FAISS_DIR, exist_ok=True)

# In-memory pipeline state
app_state = {
    "vectorstore": None,
    "retriever": None,
    "chain": None,
    "document_name": None
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm embedder and attempt loading existing index if present
    try:
        get_embedder()
        if os.path.exists(os.path.join(FAISS_DIR, "index.faiss")):
            vectorstore = load_vectorstore(FAISS_DIR)
            retriever = get_retriever(vectorstore)
            chain = create_rag_chain(retriever)
            app_state["vectorstore"] = vectorstore
            app_state["retriever"] = retriever
            app_state["chain"] = chain
            app_state["document_name"] = "Restored previous index"
            print("Successfully loaded existing FAISS index from disk.")
    except Exception as e:
        print(f"Index warm-up notice: {e}")
    yield


app = FastAPI(
    title="RAG Project API",
    description="FastAPI backend for PDF Q&A using LangChain, FAISS, and Gemini",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for Next.js frontend running locally
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust or restrict to ["http://localhost:3000"] in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str


class SourceItem(BaseModel):
    content: str
    metadata: dict


class QueryResponse(BaseModel):
    query: str
    answer: str
    document_name: Optional[str] = None
    sources: List[SourceItem] = []


@app.get("/")
def read_root():
    return {
        "status": "online",
        "has_document": app_state["chain"] is not None,
        "active_document": app_state["document_name"]
    }


@app.get("/status")
def get_status():
    return {
        "has_document": app_state["chain"] is not None,
        "active_document": app_state["document_name"]
    }


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    1. Saves uploaded PDF
    2. Loads & Chunks it using loading_chunking module
    3. Creates FAISS vectorstore using embedding_faiss module
    4. Builds LCEL RAG chain using rag_chain module
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_path = os.path.join(UPLOADS_DIR, file.filename)

    # Save uploaded file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # 1. Load and chunk
        chunks, pages = load_and_chunk_pdf(file_path)
        if not chunks:
            raise HTTPException(status_code=400, detail="No readable text extracted from the PDF.")

        # 2. Embed & store into FAISS
        vectorstore = create_vectorstore(chunks, save_dir=FAISS_DIR)

        # 3. Create retriever
        retriever = get_retriever(vectorstore)

        # 4. Create RAG chain
        chain = create_rag_chain(retriever)

        # Update in-memory state
        app_state["vectorstore"] = vectorstore
        app_state["retriever"] = retriever
        app_state["chain"] = chain
        app_state["document_name"] = file.filename

        return {
            "status": "success",
            "message": f"Successfully processed '{file.filename}'",
            "filename": file.filename,
            "total_pages": len(pages),
            "total_chunks": len(chunks)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")


@app.post("/ask", response_model=QueryResponse)
def ask(payload: QueryRequest):
    """
    Accepts user question, retrieves relevant chunks, and returns LLM answer.
    """
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    if app_state["chain"] is None or app_state["retriever"] is None:
        raise HTTPException(
            status_code=400,
            detail="No document has been uploaded yet. Please upload a PDF first."
        )

    try:
        # Invoke RAG chain
        answer = ask_question(app_state["chain"], query)

        # Retrieve source documents for citation
        source_docs = app_state["retriever"].invoke(query)
        sources = [
            SourceItem(content=doc.page_content, metadata=doc.metadata)
            for doc in source_docs
        ]

        return QueryResponse(
            query=query,
            answer=answer,
            document_name=app_state["document_name"],
            sources=sources
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)


#uvicorn backend.main:app --reload --port 8000
#npm run dev -frontend