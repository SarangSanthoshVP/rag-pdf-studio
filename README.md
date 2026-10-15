# ⚡ RAG Studio — Full-Stack PDF Knowledge Intelligence System

A production-ready Retrieval-Augmented Generation (RAG) platform with **multi-user session isolation**, enabling users to upload PDF documents, vectorize their content using FAISS and Google Gemini cloud embeddings with exponential backoff, and perform grounded question-answering with verifiable citations.

🌐 **Live Demo (Frontend):** [https://rag-pdf-studio.vercel.app](https://rag-pdf-studio.vercel.app)  
🚀 **Live API & Swagger Docs (Backend):** [https://rag-pdf-studio.onrender.com/docs](https://rag-pdf-studio.onrender.com/docs)

---

## 📌 Architecture Overview

```
[User Browser (Alice)]            [User Browser (Bob)]
  • Session ID: abc-123             • Session ID: xyz-789
  • Clean fresh state               • Clean fresh state
           │                                 │
           ▼ (HTTPS + Session ID)            ▼ (HTTPS + Session ID)
┌────────────────────────────────────────────────────────┐
│              Next.js 16 Frontend (Vercel)              │
│  • Persistent browser session ID (localStorage)        │
│  • Drag & Drop PDF upload & real-time indexing stats   │
│  • Interactive Q&A chat interface                      │
│  • Source citation drawer with page references         │
│  • "Clear / New PDF" session reset                     │
└───────────────────────────┬────────────────────────────┘
                            │
                 REST API   │ POST /upload (file, session_id)
               Calls (JSON) │ POST /ask (query, session_id)
                            │ GET  /status?session_id=...
                            ▼
┌────────────────────────────────────────────────────────┐
│                FastAPI Backend (Render)                │
│  • Multi-user session router                           │
│  • Isolated in-memory cache per session_id             │
└─────────────┬────────────────────────────┬─────────────┘
              │                            │
   Per-Session Vector Store                │ Retrieval & Generation
              ▼                            ▼
┌──────────────────────────────┐  ┌──────────────────────────────────┐
│  FAISS Isolated Indices      │  │ Google Gemini Cloud API          │
│  • /data/faiss_index/abc-123 │  │ • Embeddings: gemini-embedding-001│
│  • /data/faiss_index/xyz-789 │  │   (with exponential backoff)     │
│  • Zero cross-user leakage   │  │ • Generation: gemini-3.5-flash   │
└──────────────────────────────┘  └──────────────────────────────────┘
```

---

## 🔒 Multi-User Session Isolation & Privacy

In standard RAG demos, a single global vector store causes **data leakage** (where user B sees user A's uploaded PDF) and **race condition conflicts**. RAG Studio implements **Session Isolation**:

1. **Client-Side UUID (`localStorage`):**
   - When a user opens the site for the first time, a unique `session_id` is created in `localStorage`.
   - **Fresh Demo Experience:** New visitors always see an empty, welcoming interface with no previous user's documents.
   - **Tab & Refresh Persistence:** If a user refreshes the page or switches tabs, their `session_id` is retained, automatically restoring their active document and statistics.

2. **Per-Session Vector Store & Metadata:**
   - The backend stores FAISS vector indices and documents inside isolated directories: `backend/data/faiss_index/{session_id}/`.
   - Every `/status`, `/upload`, and `/ask` request is strictly scoped to the requesting `session_id`. One user can **never** view, query, or overwrite another user's document.

3. **Session Reset (`POST /reset_session`):**
   - Users can click **"✕ Clear / New PDF"** at any time to clear their index and start a fresh session with a new document.

---

## 🧠 LangChain RAG Pipeline Breakdown

The backend is built around a modular architecture split into distinct, single-responsibility components:

### 1. Document Loading (`backend/loading_chunking.py`)
- Uses `PyPDFLoader` to extract raw text and page-level metadata from uploaded PDF files.
- Retains page numbers for grounding and citation in answers.

### 2. Recursive Text Chunking (`backend/loading_chunking.py`)
- Uses `RecursiveCharacterTextSplitter` with:
  - `chunk_size = 1000` characters
  - `chunk_overlap = 150` characters
  - `separators = ["\n\n", "\n", ", ", " ", ""]`
- Reduces chunk fragmentation and preserves cohesive context while drastically cutting down quota usage.

### 3. Cloud Embeddings & Exponential Backoff (`backend/embedding_faiss.py`)
- Uses `GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")`.
- **Zero Local RAM Overhead:** Embeddings are computed via Google's cloud API, generating 3,072-dimensional vector representations.
- **Resilient Batching & Exponential Backoff:** Chunks are sent in batches of 25. If Google's free-tier rate limit (`429 RESOURCE_EXHAUSTED`) is reached, the system automatically parses Google's recommended cooldown window (or computes jittered exponential backoff) and retries up to 6 times without failing.

### 4. Vector Storage & Retrieval (`backend/embedding_faiss.py`)
- Builds per-session `FAISS` indices (`faiss-cpu`) stored on disk and cached in memory.
- Returns similarity retrievers configured with `search_kwargs={"k": 5}`.

### 5. Cross-Encoder Reranking (`backend/reranker.py`)
- Employs `cross-encoder/ms-marco-MiniLM-L-6-v2` via `ContextualCompressionRetriever` to re-score and rank the top 3 most relevant context chunks.
- **Cloud Safe Fallback:** If `sentence-transformers` is unavailable (e.g. to stay strictly under Render's 512 MB Free Tier RAM ceiling), the reranker automatically and gracefully falls back to the standard FAISS retriever without throwing errors.

### 6. Query Routing & Decomposition (`backend/query_router.py`)
- **Classifier:** Automatically classifies incoming questions into:
  - `single_fact`: Direct factual questions looking for specific points or definitions.
  - `multi_part`: Questions comparing entities or asking multi-faceted questions (e.g., *"What is X and how does it compare to Y?"*).
  - `summarization`: Requests for general document synthesis or high-level overviews.
- **Sub-Query Decomposition & Evidence Tracking:**
  - Decomposes `multi_part` questions into 2–4 targeted sub-queries.
  - Retrieves dedicated context chunks for each sub-query separately.
  - Returns clear evidence showing which documents were retrieved for which part before merging context.

### 7. LCEL Generation Chain (`backend/rag_chain.py`)
- Built using **LangChain Expression Language (LCEL)**:
  ```python
  chain = (
      {
          "context": retriever | format_docs,
          "query": RunnablePassthrough()
      }
      | prompt
      | llm
      | StrOutputParser()
  )
  ```
- **Prompt:** Enforces strict grounding — answers only from retrieved context, otherwise politely informs the user that information is outside the document.
- **LLM:** `ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0.0)`.

---

## 🚀 Cloud Deployment Guide (Step-by-Step)

This project separates frontend and backend to leverage the best free cloud platforms for each ecosystem:

| Tier | Platform | Purpose | Free Quota |
| :--- | :--- | :--- | :--- |
| **Frontend** | [Vercel](https://vercel.com) | Next.js Serverless Edge CDN | Unlimited builds, automatic SSL |
| **Backend** | [Render](https://render.com) | Python FastAPI Web Service | 512 MB RAM, free web service |

---

### Part A: Deploying Backend on Render

1. Sign in to [Render.com](https://render.com) using your GitHub account.
2. Click **New +** $\rightarrow$ **Web Service**.
3. Connect your repository (`rag-pdf-studio`).
4. Configure service parameters:
   - **Name:** `rag-pdf-studio` (or your choice)
   - **Language:** `Python`
   - **Branch:** `main`
   - **Build Command:**
     ```bash
     pip install -r requirements.txt
     ```
   - **Start Command:**
     ```bash
     uvicorn backend.main:app --host 0.0.0.0 --port $PORT
     ```
5. Add **Environment Variables**:
   - `GEMINI_API_KEY`: *Your Google AI Studio API key*
   - `PYTHON_VERSION`: `3.11.9`
6. Click **Deploy Web Service**.
7. Once live, copy your backend URL (e.g. `https://rag-pdf-studio.onrender.com`).

---

### Part B: Deploying Frontend on Vercel

1. Sign in to [Vercel.com](https://vercel.com) with GitHub.
2. Click **Add New...** $\rightarrow$ **Project**.
3. Select your repository (`rag-pdf-studio`).
4. In project configuration:
   - Click **Edit** next to **Root Directory** and choose `frontend`.
   - Framework Preset will auto-detect **Next.js**.
5. Expand **Environment Variables** and add:
   - **Key:** `NEXT_PUBLIC_API_URL`
   - **Value:** `https://rag-pdf-studio.onrender.com` *(your live Render backend URL)*
6. Click **Deploy**.
7. *(Important)* If you add or modify `NEXT_PUBLIC_API_URL` after deployment, trigger a **Redeploy** on Vercel (under the **Deployments** tab) so the variable is embedded into the production bundle.

---

### Part C: Continuous Deployment (CI/CD)

Whenever you edit code locally, just push to GitHub:
```powershell
git add .
git commit -m "update feature"
git push origin main
```
- **Vercel** automatically rebuilds and deploys the frontend live within 45 seconds.
- **Render** automatically rebuilds and restarts the backend live within 1–2 minutes.

---

## 💻 Local Development Setup

### 1. Clone & Set Environment
```powershell
git clone https://github.com/SarangSanthoshVP/rag-pdf-studio.git
cd rag-pdf-studio

# Create .env file with your API key
echo "GEMINI_API_KEY=your_key_here" > .env
```

### 2. Backend Setup
```powershell
# Create and activate virtual environment
python -m venv .venvp
.\.venvp\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Run FastAPI server on port 8000
uvicorn backend.main:app --reload --port 8000
```
Swagger UI: `http://localhost:8000/docs`

### 3. Frontend Setup
```powershell
cd frontend
npm install
npm run dev
```
Next.js App: `http://localhost:3000`

---

## 📁 Repository Structure

```text
rag-pdf-studio/
├── .env                       # Local secrets (ignored by Git)
├── .gitignore                 # Excludes venv, node_modules, temp data
├── README.md                  # Project documentation & architecture
├── requirements.txt           # Lightweight production backend dependencies
├── backend/
│   ├── main.py                # FastAPI routes (/upload, /ask, /status, /reset_session) + CORS
│   ├── loading_chunking.py    # PyPDFLoader + RecursiveCharacterTextSplitter
│   ├── embedding_faiss.py     # GoogleGenerativeAIEmbeddings + Resilient backoff + FAISS
│   ├── reranker.py            # CrossEncoderReranker with safe cloud fallback
│   ├── query_router.py        # Query classification & sub-query decomposition
│   ├── rag_chain.py           # LCEL chain (Prompt + Gemini 3.5 Flash + Parser)
│   └── data/
│       ├── uploads/           # PDF storage directory (partitioned by session_id)
│       └── faiss_index/       # Serialized FAISS indices (isolated per session_id)
└── frontend/
    ├── package.json           # Next.js 16, React 19 dependencies
    └── app/
        ├── layout.js          # Google Fonts & global HTML wrapper
        ├── globals.css        # Vanilla CSS design system (Dark mode & animations)
        └── page.js            # Interactive RAG Dashboard UI & chat
```

---

## 🛡️ License
MIT License. Free for educational and commercial use.
