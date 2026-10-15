"use client";

import { useState, useEffect, useRef } from "react";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function RAGDashboard() {
  // State
  const [backendOnline, setBackendOnline] = useState(false);
  const [sessionId, setSessionId] = useState("");
  const [selectedFile, setSelectedFile] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState(null); // { type: 'success' | 'error', message: string, stats?: {} }
  const [activeDoc, setActiveDoc] = useState(null); // { filename: string, pages: number, chunks: number }

  const [messages, setMessages] = useState([]);
  const [query, setQuery] = useState("");
  const [asking, setAsking] = useState(false);
  const [expandedSources, setExpandedSources] = useState({});

  const fileInputRef = useRef(null);
  const messagesEndRef = useRef(null);

  // Helper to retrieve or initialize persistent session ID
  const getOrCreateSessionId = () => {
    if (typeof window === "undefined") return "";
    let sid = localStorage.getItem("rag_session_id");
    if (!sid) {
      sid = (typeof crypto !== "undefined" && crypto.randomUUID)
        ? crypto.randomUUID()
        : `user_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
      localStorage.setItem("rag_session_id", sid);
    }
    return sid;
  };

  // Check backend health and status isolated to current user's session
  const checkBackendStatus = async (currentSid) => {
    const sid = currentSid || sessionId || (typeof window !== "undefined" ? localStorage.getItem("rag_session_id") : "");
    try {
      const url = sid ? `${API_BASE_URL}/status?session_id=${encodeURIComponent(sid)}` : `${API_BASE_URL}/status`;
      const res = await fetch(url, { signal: AbortSignal.timeout(3000) });
      if (res.ok) {
        const data = await res.json();
        setBackendOnline(true);
        if (data.has_document && data.active_document) {
          setActiveDoc({
            filename: data.active_document,
            pages: data.total_pages,
            chunks: data.total_chunks,
          });
        } else {
          setActiveDoc(null);
        }
      } else {
        setBackendOnline(false);
      }
    } catch {
      setBackendOnline(false);
    }
  };

  // Initialize session on mount
  useEffect(() => {
    const sid = getOrCreateSessionId();
    setSessionId(sid);
    checkBackendStatus(sid);
    const interval = setInterval(() => checkBackendStatus(sid), 8000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, asking]);

  // Reset active session and start fresh
  const handleResetSession = async () => {
    const sid = sessionId || (typeof window !== "undefined" ? localStorage.getItem("rag_session_id") : "");
    try {
      if (sid) {
        await fetch(`${API_BASE_URL}/reset_session?session_id=${encodeURIComponent(sid)}`, { method: "POST" });
      }
    } catch {}

    const newSid = (typeof crypto !== "undefined" && crypto.randomUUID)
      ? crypto.randomUUID()
      : `user_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
    if (typeof window !== "undefined") {
      localStorage.setItem("rag_session_id", newSid);
    }
    setSessionId(newSid);
    setActiveDoc(null);
    setSelectedFile(null);
    setUploadStatus(null);
    setMessages([]);
  };

  // Drag & drop handlers
  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (file.type === "application/pdf" || file.name.endsWith(".pdf")) {
        setSelectedFile(file);
        setUploadStatus(null);
      } else {
        setUploadStatus({ type: "error", message: "Only PDF documents are supported." });
      }
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedFile(file);
      setUploadStatus(null);
    }
  };

  // Upload handler with session isolation
  const handleUpload = async () => {
    if (!selectedFile) return;

    setUploading(true);
    setUploadStatus({ type: "loading", message: `Processing and indexing "${selectedFile.name}"...` });

    const sid = sessionId || (typeof window !== "undefined" ? localStorage.getItem("rag_session_id") : "");
    const formData = new FormData();
    formData.append("file", selectedFile);
    if (sid) {
      formData.append("session_id", sid);
    }

    try {
      const res = await fetch(`${API_BASE_URL}/upload`, {
        method: "POST",
        body: formData,
      });

      const data = await res.json();

      if (res.ok) {
        setUploadStatus({
          type: "success",
          message: data.message,
          stats: { pages: data.total_pages, chunks: data.total_chunks },
        });
        setActiveDoc({
          filename: data.filename,
          pages: data.total_pages,
          chunks: data.total_chunks,
        });
        // Add welcome message in chat
        setMessages([
          {
            sender: "assistant",
            text: `Document "${data.filename}" has been successfully indexed into FAISS (${data.total_pages} pages, ${data.total_chunks} chunks). What would you like to know about it?`,
          },
        ]);
      } else {
        setUploadStatus({
          type: "error",
          message: data.detail || "Upload failed. Please check backend server.",
        });
      }
    } catch (err) {
      setUploadStatus({
        type: "error",
        message: "Failed to connect to backend. Is it running?",
      });
    } finally {
      setUploading(false);
    }
  };

  // Ask Question handler with session isolation
  const handleAsk = async (e) => {
    if (e) e.preventDefault();
    const currentQuery = query.trim();
    if (!currentQuery || asking) return;

    // Append user message
    const newMessages = [...messages, { sender: "user", text: currentQuery }];
    setMessages(newMessages);
    setQuery("");
    setAsking(true);

    const sid = sessionId || (typeof window !== "undefined" ? localStorage.getItem("rag_session_id") : "");

    try {
      const res = await fetch(`${API_BASE_URL}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: currentQuery, session_id: sid }),
      });

      const data = await res.json();

      if (res.ok) {
        setMessages([
          ...newMessages,
          {
            sender: "assistant",
            text: data.answer,
            query_type: data.query_type,
            sub_queries: data.sub_queries || [],
            sub_query_evidence: data.sub_query_evidence || {},
            sources: data.sources || [],
          },
        ]);
      } else {
        setMessages([
          ...newMessages,
          {
            sender: "assistant",
            text: `Error: ${data.detail || "Unable to get an answer."}`,
          },
        ]);
      }
    } catch {
      setMessages([
        ...newMessages,
        {
          sender: "assistant",
          text: "Connection failed. Please verify that the FastAPI backend is running.",
        },
      ]);
    } finally {
      setAsking(false);
    }
  };

  const toggleSources = (index) => {
    setExpandedSources((prev) => ({
      ...prev,
      [index]: !prev[index],
    }));
  };

  const formatFileSize = (bytes) => {
    if (!bytes) return "0 KB";
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  const sampleQuestions = [
    "Provide a detailed summary of this document.",
    "What are the key conclusions or takeaways?",
    "Explain the core methodology or topics covered.",
  ];

  return (
    <div className="app-container">
      {/* Top Header */}
      <header className="app-header">
        <div className="brand-section">
          <div className="brand-icon">⌛️</div>
          <div className="brand-text">
            <h1>RAG STUDIO DASHBOARD</h1>
            <p>PDF Knowledge Retrieval </p>
          </div>
        </div>

        <div className="header-status">
          <div
            id="backend-status-pill"
            className={`status-pill ${backendOnline ? "online" : "offline"}`}
          >
            <span className="status-dot"></span>
            <span>{backendOnline ? "Backend Live" : "Backend Offline"}</span>
          </div>
          <button
            id="refresh-status-btn"
            onClick={checkBackendStatus}
            style={{
              background: "transparent",
              border: "1px solid var(--border-subtle)",
              color: "var(--text-muted)",
              padding: "6px 12px",
              borderRadius: "999px",
              cursor: "pointer",
              fontSize: "12px",
            }}
          >
            Refresh
          </button>
        </div>
      </header>

      {/* Main Grid */}
      <main className="dashboard-grid">
        {/* Left: Upload & Document Controls */}
        <div className="sidebar-panel">
          {/* Upload Card */}
          <section className="card">
            <div className="card-title-group">
              <h2 className="card-title">
                <span>📄</span> UPLOAD & INDEX PDF
              </h2>
            </div>

            {/* Dropzone */}
            <div
              id="pdf-dropzone"
              className={`dropzone ${isDragging ? "drag-active" : ""}`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              <input
                id="pdf-file-input"
                type="file"
                accept=".pdf,application/pdf"
                ref={fileInputRef}
                className="file-input-hidden"
                onChange={handleFileChange}
              />
              <span className="dropzone-icon">📥</span>
              <p className="dropzone-label">Click to browse or drag PDF here</p>
              <p className="dropzone-sublabel">Supported format: PDF up to 50MB</p>
            </div>

            {/* Selected File Details */}
            {selectedFile && (
              <div className="file-preview">
                <div className="file-info">
                  <span className="file-badge-icon">📑</span>
                  <div>
                    <p className="file-meta-name" title={selectedFile.name}>
                      {selectedFile.name}
                    </p>
                    <p className="file-meta-size">{formatFileSize(selectedFile.size)}</p>
                  </div>
                </div>
                <button
                  id="remove-file-btn"
                  className="btn-remove-file"
                  onClick={() => {
                    setSelectedFile(null);
                    setUploadStatus(null);
                  }}
                  title="Remove file"
                >
                  ✕
                </button>
              </div>
            )}

            {/* Upload Button */}
            <button
              id="upload-submit-btn"
              className="btn-primary"
              disabled={!selectedFile || uploading}
              onClick={handleUpload}
            >
              {uploading ? (
                <>
                  <span className="spinner"></span>
                  <span>Splitting & Vectorizing...</span>
                </>
              ) : (
                <>
                  <span></span>
                  <span>Process & Index PDF</span>
                </>
              )}
            </button>

            {/* Upload Notification Box */}
            {uploadStatus && (
              <div className={`upload-status-box ${uploadStatus.type}`}>
                {uploadStatus.type === "loading" && <span className="spinner"></span>}
                {uploadStatus.type === "success" && <span>✓</span>}
                {uploadStatus.type === "error" && <span>⚠</span>}
                <span>{uploadStatus.message}</span>
              </div>
            )}
          </section>

          {/* Active Document Overview Card */}
          <section className="card">
            <div className="card-title-group">
              <h2 className="card-title">
                <span>📖</span> ACTIVE KNOWLEDGE BASE
              </h2>
              {activeDoc && (
                <button
                  id="reset-session-btn"
                  onClick={handleResetSession}
                  style={{
                    background: "rgba(244, 63, 94, 0.1)",
                    border: "1px solid rgba(244, 63, 94, 0.3)",
                    color: "#fb7185",
                    padding: "4px 10px",
                    borderRadius: "6px",
                    fontSize: "11px",
                    fontWeight: "500",
                    cursor: "pointer",
                    transition: "all 0.2s ease",
                  }}
                  title="Clear active document and start fresh"
                >
                  ✕ Clear / New PDF
                </button>
              )}
            </div>

            {activeDoc ? (
              <div>
                <p style={{ fontSize: "14px", fontWeight: "600", color: "#e2e8f0" }}>
                  {activeDoc.filename}
                </p>
                <div className="doc-stats-grid">
                  <div className="stat-chip">
                    <p className="stat-chip-label">Pages Loaded</p>
                    <p className="stat-chip-val">{activeDoc.pages || "—"}</p>
                  </div>
                  <div className="stat-chip">
                    <p className="stat-chip-label">Vector Chunks</p>
                    <p className="stat-chip-val">{activeDoc.chunks || "—"}</p>
                  </div>
                </div>
                <div
                  style={{
                    marginTop: "14px",
                    display: "flex",
                    alignItems: "center",
                    gap: "8px",
                    fontSize: "12px",
                    color: "var(--accent-emerald)",
                  }}
                >
                  <span
                    style={{
                      width: "8px",
                      height: "8px",
                      borderRadius: "50%",
                      background: "#10b981",
                    }}
                  ></span>
                  <span>FAISS vector index active in memory</span>
                </div>
              </div>
            ) : (
              <p style={{ fontSize: "13px", color: "var(--text-muted)", lineHeight: "1.6" }}>
                No PDF is currently loaded. Upload a PDF using the box above to extract text chunks and generate embeddings.
              </p>
            )}
          </section>
        </div>

        {/* Right: Interactive Chat Surface */}
        <section className="card chat-container">
          <div className="chat-header">
            <div className="chat-title-group">
              <h2>Document Assistant</h2>
              <p>
                {activeDoc
                  ? `Answering queries based strictly on "${activeDoc.filename}"`
                  : "Upload a document on the left to start asking questions"}
              </p>
            </div>
            {messages.length > 0 && (
              <button
                id="clear-chat-btn"
                onClick={() => setMessages([])}
                style={{
                  background: "transparent",
                  border: "1px solid var(--border-subtle)",
                  color: "var(--text-muted)",
                  padding: "5px 12px",
                  borderRadius: "6px",
                  fontSize: "12px",
                  cursor: "pointer",
                }}
              >
                Clear Chat
              </button>
            )}
          </div>

          {/* Messages Area */}
          <div className="chat-messages">
            {messages.length === 0 ? (
              <div className="empty-chat">
                <span className="empty-chat-icon">💬</span>
                <h3>Ask Anything About Your PDF</h3>
                <p>
                  The system will search through your document chunks using FAISS cosine similarity and formulate answers using Google Gemini.
                </p>

                {activeDoc && (
                  <div className="sample-queries">
                    {sampleQuestions.map((sq, i) => (
                      <button
                        key={i}
                        className="chip-query"
                        onClick={() => {
                          setQuery(sq);
                        }}
                      >
                        {sq}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            ) : (
              messages.map((msg, idx) => (
                <div key={idx} className="message-row">
                  {msg.sender === "user" ? (
                    <div className="message-user">{msg.text}</div>
                  ) : (
                    <div className="message-ai">
                      <div className="message-ai-header" style={{ justifyContent: "space-between", width: "100%" }}>
                        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                          <span></span>
                          <span>Response</span>
                        </div>
                        {msg.query_type && (
                          <span
                            style={{
                              fontSize: "11px",
                              padding: "2px 8px",
                              borderRadius: "999px",
                              fontWeight: 600,
                              background:
                                msg.query_type === "multi_part"
                                  ? "rgba(168, 85, 247, 0.2)"
                                  : msg.query_type === "summarization"
                                    ? "rgba(6, 182, 212, 0.2)"
                                    : "rgba(16, 185, 129, 0.2)",
                              color:
                                msg.query_type === "multi_part"
                                  ? "#c084fc"
                                  : msg.query_type === "summarization"
                                    ? "#22d3ee"
                                    : "#34d399",
                              border: "1px solid rgba(255, 255, 255, 0.1)",
                            }}
                          >
                            {msg.query_type === "multi_part"
                              ? " Multi-part Routed"
                              : msg.query_type === "summarization"
                                ? "Summarization"
                                : " Single-fact"}
                          </span>
                        )}
                      </div>

                      {/* Decomposed Sub-Queries (if multi_part) */}
                      {msg.sub_queries && msg.sub_queries.length > 1 && (
                        <div
                          style={{
                            background: "rgba(99, 102, 241, 0.08)",
                            border: "1px dashed rgba(99, 102, 241, 0.3)",
                            borderRadius: "8px",
                            padding: "8px 12px",
                            fontSize: "12px",
                          }}
                        >
                          <p style={{ fontWeight: 600, color: "#a5b4fc", marginBottom: "4px" }}>
                            🔍 Decomposed Sub-Queries:
                          </p>
                          <ul style={{ paddingLeft: "18px", color: "#cbd5e1", margin: 0 }}>
                            {msg.sub_queries.map((sq, sqIdx) => (
                              <li key={sqIdx}>{sq}</li>
                            ))}
                          </ul>
                        </div>
                      )}

                      <div className="message-ai-content">{msg.text}</div>

                      {/* Source Citations */}
                      {msg.sources && msg.sources.length > 0 && (
                        <div>
                          <button
                            className="sources-toggle"
                            onClick={() => toggleSources(idx)}
                          >
                            <span>{expandedSources[idx] ? "▼ Hide" : "▶ View"}</span>
                            <span>{msg.sources.length} Retrieved Source Chunks</span>
                          </button>

                          {expandedSources[idx] && (
                            <div className="sources-list">
                              {msg.sources.map((src, sIdx) => (
                                <div key={sIdx} className="source-item">
                                  <p className="source-meta">
                                    Chunk #{sIdx + 1} • Page {src.metadata?.page !== undefined ? src.metadata.page + 1 : "N/A"}
                                  </p>
                                  <p>{src.content}</p>
                                </div>
                              ))}
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ))
            )}

            {asking && (
              <div className="message-row">
                <div className="message-ai">
                  <div className="message-ai-header">
                    <span className="spinner" style={{ width: "12px", height: "12px" }}></span>
                    <span>Retrieving context & generating answer...</span>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Input Bar */}
          <form className="chat-input-bar" onSubmit={handleAsk}>
            <input
              id="query-input"
              type="text"
              placeholder={
                activeDoc
                  ? "Ask a question about the document..."
                  : "Please upload a PDF first to ask questions..."
              }
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              disabled={!activeDoc || asking}
              className="chat-input"
            />
            <button
              id="ask-submit-btn"
              type="submit"
              disabled={!query.trim() || !activeDoc || asking}
              className="btn-send"
            >
              <span>Send</span>
              <span>➤</span>
            </button>
          </form>
        </section>
      </main>
    </div>
  );
}
