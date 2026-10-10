# ⚡ RAG Studio — Full-Stack PDF Knowledge Intelligence System

A production-ready Retrieval-Augmented Generation (RAG) platform that enables users to upload PDF documents, vectorize their content using FAISS and Google Gemini cloud embeddings, and perform question-answering with verifiable citations.

🌐 **Live Demo (Frontend):** [https://rag-pdf-studio.vercel.app](https://rag-pdf-studio.vercel.app)  
🚀 **Live API & Swagger Docs (Backend):** [https://rag-pdf-studio.onrender.com/docs](https://rag-pdf-studio.onrender.com/docs)

---

## 📌 Architecture Overview

```
[User Browser]
       │
       ▼ (HTTPS)
┌────────────────────────────────────────┐
│      Next.js 16 Frontend (Vercel)       │
│  • Drag & Drop PDF upload              │
│  • Real-time indexing stats            │
│  • Interactive Q&A chat interface      │
│  • Source citation drawer with pages   │
└───────────────────┬────────────────────┘
                    │
         REST API   │ POST /upload
       Calls (JSON) │ POST /ask
                    ▼
┌────────────────────────────────────────┐
│       FastAPI Backend (Render)         │
│  • CORS-enabled Python ASGI server     │
│  • In-memory session state             │
└─────────┬──────────────────┬───────────┘
          │                  │
   PDF Chunks & Vectorize    │ Retrieval & Generation
          ▼                  ▼
┌──────────────────┐  ┌──────────────────────────────────┐
│  FAISS Index     │  │ Google Gemini Cloud API          │
│  (Vector Store)  │  │ • Embeddings: gemini-embedding-001│
│  Top-k Search    │  │ • Generation: gemini-3.8-flash   │
└──────────────────┘  └──────────────────────────────────┘
```

---

## 🧠 LangChain RAG Pipeline Breakdown

The backend is built around a modular architecture split into distinct, single-responsibility components:

### 1. Document Loading (`backend/loading_chunking.py`)
- Uses `PyPDFLoader` to extract raw text and page-level metadata from uploaded PDF files.
- Each page retains its zero-indexed page number for grounding and citation.

### 2. Recursive Text Chunking (`backend/loading_chunking.py`)
- Uses `RecursiveCharacterTextSplitter` with:
  - `chunk_size = 600` characters
  - `chunk_overlap = 200` characters
  - `separators = ["\n\n", "\n", ", ", " ", ""]`
- Maintains semantic coherence across paragraph boundaries while preventing context fragmentation.

### 3. Cloud Embeddings (`backend/embedding_faiss.py`)
- Uses `GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")`.
- **Zero Local RAM Overhead:** Unlike local PyTorch models (`sentence-transformers`), embeddings are computed via Google's cloud API, generating 3,072-dimensional vector representations.
- **Why this was chosen:** Solved Render's 512 MB RAM limit (OOM crash) by eliminating 1.5 GB of PyTorch/Transformers dependencies.

### 4. Vector Storage & Retrieval (`backend/embedding_faiss.py`)
- Vectorizes chunks and builds an in-memory & on-disk `FAISS` index (`faiss-cpu`).
- Returns a similarity retriever configured with `search_kwargs={"k": 5}` to retrieve the top 5 most relevant context chunks.

### 5. LCEL Generation Chain (`backend/rag_chain.py`)
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
- **LLM:** `ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0.0)`.

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
│   ├── main.py                # FastAPI routes (/upload, /ask, /status) + CORS
│   ├── loading_chunking.py    # PyPDFLoader + RecursiveCharacterTextSplitter
│   ├── embedding_faiss.py     # GoogleGenerativeAIEmbeddings + FAISS vectorstore
│   ├── rag_chain.py           # LCEL chain (Prompt + Gemini 3.8 Flash + Parser)
│   └── data/
│       ├── uploads/           # PDF storage directory
│       └── faiss_index/       # Serialized FAISS index
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
