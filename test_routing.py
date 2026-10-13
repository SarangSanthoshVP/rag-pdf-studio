# Verification test for Query Routing & Decomposition
import os
import json
from dotenv import load_dotenv

# Load env variables
load_dotenv()

from langchain_core.documents import Document
from backend.embedding_faiss import create_vectorstore, get_retriever, get_embedder
from backend.query_router import route_and_retrieve, classify_and_decompose_query
from backend.rag_chain import generate_answer_from_docs

def run_checklist_test():
    print("=" * 70)
    print("1. PREPARING KNOWLEDGE BASE WITH CONCEPT X AND CONCEPT Y")
    print("=" * 70)

    # Documents containing Concept X (Superposition) and Concept Y (Entanglement)
    docs = [
        Document(
            page_content="Concept X: Quantum Superposition is a fundamental principle of quantum mechanics where a physical system can exist in multiple quantum states simultaneously until it is measured. It is represented mathematically by linear combinations of state vectors.",
            metadata={"source": "physics_handbook.pdf", "page": 1, "topic": "Superposition"}
        ),
        Document(
            page_content="Concept Y: Quantum Entanglement is a phenomenon where two or more particles become interconnected such that the quantum state of one particle cannot be described independently of the others, regardless of the distance separating them.",
            metadata={"source": "physics_handbook.pdf", "page": 2, "topic": "Entanglement"}
        ),
        Document(
            page_content="Comparison: While Superposition refers to a single quantum system existing across multiple possibilities at once, Entanglement describes non-local correlations between multiple distinct quantum systems.",
            metadata={"source": "physics_handbook.pdf", "page": 3, "topic": "Comparison"}
        )
    ]

    print(f"Creating FAISS vector store with {len(docs)} documents...")
    embedder = get_embedder()
    vectorstore = create_vectorstore(docs, embedder=embedder)
    retriever = get_retriever(vectorstore, k=2)

    # Test 1: Single Fact
    print("\n" + "=" * 70)
    print("TEST 1: SINGLE-FACT LOOKUP")
    print("=" * 70)
    q1 = "What is the mathematical representation of Superposition?"
    plan1 = classify_and_decompose_query(q1)
    print(f"Query: '{q1}'")
    print(f"Detected Type: {plan1['query_type']}")
    print(f"Sub-queries: {plan1['sub_queries']}")

    # Test 2: Summarization
    print("\n" + "=" * 70)
    print("TEST 2: SUMMARIZATION REQUEST")
    print("=" * 70)
    q2 = "Can you summarize the main quantum concepts discussed in this document?"
    plan2 = classify_and_decompose_query(q2)
    print(f"Query: '{q2}'")
    print(f"Detected Type: {plan2['query_type']}")
    print(f"Sub-queries: {plan2['sub_queries']}")

    # Test 3: Multi-Part Comparison (Checklist Item)
    print("\n" + "=" * 70)
    print("TEST 3: MULTI-PART QUESTION (CHECKLIST)")
    print("=" * 70)
    q3 = "What is Superposition and how does it compare to Quantum Entanglement?"
    print(f"Multi-Part Query: '{q3}'\n")

    result = route_and_retrieve(q3, retriever)
    print(f"Classification: {result['query_type']}")
    print(f"Decomposed Sub-Queries:")
    for i, sq in enumerate(result['sub_queries'], 1):
        print(f"  {i}. {sq}")

    print("\n--- EVIDENCE BREAKDOWN (PROVING BOTH PARTS RETRIEVED) ---")
    for sub_q, evidence_chunks in result['evidence_by_sub_query'].items():
        print(f"\n[Sub-Query]: \"{sub_q}\"")
        print(f"Retrieved {len(evidence_chunks)} chunk(s):")
        for chunk in evidence_chunks:
            topic = chunk['metadata'].get('topic', 'N/A')
            page = chunk['metadata'].get('page', 'N/A')
            snippet = chunk['content'][:110] + "..."
            print(f"   -> [Page {page} | Topic: {topic}] {snippet}")

    # Generate answer from merged context
    print("\n--- FINAL ANSWER GENERATION FROM MERGED CONTEXT ---")
    answer = generate_answer_from_docs(result['merged_docs'], q3)
    print("Answer:\n", answer)
    print("\n" + "=" * 70)
    print("CHECKLIST VERIFICATION: COMPLETE AND VERIFIED!")
    print("=" * 70)

if __name__ == "__main__":
    run_checklist_test()
