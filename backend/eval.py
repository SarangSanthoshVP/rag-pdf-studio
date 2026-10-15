"""
Offline RAG Evaluation Script using Ragas metrics:
1. Faithfulness (Reference-free: checks for hallucination)
2. Answer Relevancy (Reference-free: checks pertinence to query)
3. Context Precision (Requires ground truth: checks ranking of context)
4. Context Recall (Requires ground truth: checks if ground truth was retrieved)
5. Answer Correctness (Requires ground truth: checks factual and semantic accuracy)

Run via:
    python backend/eval.py
"""
import os
import sys
import json
import time
from dotenv import load_dotenv

# Ensure backend directory is in path when run standalone
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 1. Apply compatibility bridge for Ragas on Python 3.14 & LangChain 0.3+
from backend.eval_compat import ensure_ragas_compatibility
ensure_ragas_compatibility()

from datasets import Dataset
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    AnswerCorrectness
)
import pandas as pd

from backend.rag_chain import get_llm, generate_answer_from_docs
from backend.embedding_faiss import get_embedder, load_vectorstore, create_vectorstore, get_retriever
from backend.reranker import get_cross_encoder_retriever
from langchain_core.documents import Document

# Load environment
load_dotenv()


# Built-in Golden Dataset for benchmarking
DEFAULT_GOLDEN_DATASET = [
    {
        "question": "What is FAISS and what is its main purpose?",
        "ground_truth": "FAISS (Facebook AI Similarity Search) is a library developed by Meta for fast, dense vector similarity search and clustering of high-dimensional embeddings."
    },
    {
        "question": "Why is cross-encoder reranking used after vector similarity search?",
        "ground_truth": "Cross-encoders jointly evaluate query and candidate document pairs using full cross-attention, providing higher precision relevance scores than bi-encoder cosine similarity alone."
    },
    {
        "question": "What is the difference between single-fact and multi-part queries?",
        "ground_truth": "Single-fact queries seek one specific piece of information from a single document section, whereas multi-part queries require decomposing the query into sub-questions and retrieving evidence across multiple document sections."
    }
]


def get_evaluation_retriever():
    """
    Locates an active session FAISS index or creates an in-memory sample
    vectorstore if no user index exists yet.
    """
    faiss_base_dir = os.path.join(PROJECT_ROOT, "data", "faiss_index")
    vectorstore = None

    if os.path.exists(faiss_base_dir):
        subdirs = [
            os.path.join(faiss_base_dir, d)
            for d in os.listdir(faiss_base_dir)
            if os.path.isdir(os.path.join(faiss_base_dir, d))
        ]
        for sdir in subdirs:
            try:
                vectorstore = load_vectorstore(sdir)
                if vectorstore:
                    print(f"Loaded existing session FAISS index from: {sdir}")
                    break
            except Exception:
                continue

    # Fallback to sample document if no uploaded index is available
    if vectorstore is None:
        print("No uploaded session index found. Creating demo benchmark vectorstore...")
        sample_chunks = [
            Document(
                page_content="FAISS (Facebook AI Similarity Search) is an open-source library developed by Meta AI. "
                             "Its main purpose is to enable fast, efficient similarity search and clustering of dense high-dimensional vectors.",
                metadata={"source": "benchmark_doc.pdf", "page": 0}
            ),
            Document(
                page_content="Cross-encoder reranking provides a two-stage retrieval pipeline. Bi-encoders retrieve initial candidates fast via cosine similarity, "
                             "and cross-encoders perform full joint cross-attention to score and rank the most relevant candidate passages.",
                metadata={"source": "benchmark_doc.pdf", "page": 1}
            ),
            Document(
                page_content="Query classification categorizes queries into single_fact, multi_part, or summarization. "
                             "Multi-part queries require decomposing the query into discrete sub-queries and retrieving distinct evidence for each sub-topic.",
                metadata={"source": "benchmark_doc.pdf", "page": 2}
            ),
        ]
        vectorstore = create_vectorstore(sample_chunks, save_dir=None)

    base_retriever = get_retriever(vectorstore, k=4)
    return get_cross_encoder_retriever(base_retriever, top_n=2)


def run_benchmark(golden_dataset=None, output_csv="backend/eval_results.csv"):
    """
    Executes the full 5-metric Ragas evaluation suite against the golden dataset.
    """
    if golden_dataset is None:
        golden_dataset = DEFAULT_GOLDEN_DATASET

    print(f"\n=======================================================")
    print(f"  Starting Ragas Evaluation Benchmark ({len(golden_dataset)} samples)")
    print(f"=======================================================\n")

    # 1. Initialize models and wrappers
    judge_llm = get_llm(model="gemini-3.5-flash-lite", temperature=0.0)
    embedder = get_embedder()
    wrapped_llm = LangchainLLMWrapper(judge_llm)
    wrapped_embedder = LangchainEmbeddingsWrapper(embedder)

    retriever = get_evaluation_retriever()

    # 2. Collect pipeline outputs
    evaluation_data = {
        "user_input": [],
        "response": [],
        "retrieved_contexts": [],
        "reference": []
    }

    print("Generating RAG responses for evaluation dataset...")
    for idx, item in enumerate(golden_dataset, 1):
        query = item["question"]
        ground_truth = item["ground_truth"]

        # Retrieve documents
        docs = retriever.invoke(query)
        context_list = [doc.page_content for doc in docs]

        # Generate response
        response = generate_answer_from_docs(docs, query, llm=judge_llm)

        evaluation_data["user_input"].append(query)
        evaluation_data["retrieved_contexts"].append(context_list)
        evaluation_data["response"].append(response)
        evaluation_data["reference"].append(ground_truth)

        print(f"  [{idx}/{len(golden_dataset)}] Query processed: {query[:50]}...")
        time.sleep(1)  # Throttling to respect free-tier RPM limits

    # 3. Build Ragas dataset
    ragas_dataset = Dataset.from_dict(evaluation_data)

    print("\nRunning Ragas evaluation metrics:")
    print("  • Faithfulness (Reference-free)")
    print("  • Answer Relevancy (Reference-free)")
    print("  • Context Precision (Reference-based)")
    print("  • Context Recall (Reference-based)")
    print("  • Answer Correctness (Reference-based)")

    # 4. Execute evaluation
    results = evaluate(
        dataset=ragas_dataset,
        metrics=[
            Faithfulness(),
            AnswerRelevancy(),
            ContextPrecision(),
            ContextRecall(),
            AnswerCorrectness()
        ],
        llm=wrapped_llm,
        embeddings=wrapped_embedder
    )

    # 5. Format and save results
    results_df = results.to_pandas()
    output_path = os.path.join(PROJECT_ROOT, output_csv)
    results_df.to_csv(output_path, index=False)

    print("\n=======================================================")
    print("  BENCHMARK RESULTS SCORECARD")
    print("=======================================================")
    cols_to_show = ["user_input", "faithfulness", "answer_relevancy", "context_precision", "context_recall", "answer_correctness"]
    display_cols = [c for c in cols_to_show if c in results_df.columns]
    print(results_df[display_cols].to_string(index=False))

    print("\n-------------------------------------------------------")
    print("  AVERAGE METRIC SCORES")
    print("-------------------------------------------------------")
    for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "answer_correctness"]:
        if metric in results_df:
            avg_score = results_df[metric].mean()
            print(f"  {metric.replace('_', ' ').title():<22}: {avg_score:.2%}")

    print(f"\nDetailed scorecard saved to: {output_path}")
    return results_df


if __name__ == "__main__":
    run_benchmark()