# Query Routing and Decomposition Module
import os
import json
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document

# Load environment variables
load_dotenv()


class QueryPlan(BaseModel):
    query_type: str = Field(
        description="One of 'single_fact', 'multi_part', or 'summarization'"
    )
    sub_queries: List[str] = Field(
        description="List of sub-queries. If multi_part, decomposed questions. Otherwise [query]."
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of why this query type was selected."
    )


def get_router_llm(model: str = "gemini-3.5-flash-lite", temperature: float = 0.0, api_key: str = None):
    """
    Initialize lightweight LLM for routing and query decomposition.
    """
    key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY not found in environment.")

    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        api_key=key
    )


def classify_and_decompose_query(query: str, llm=None) -> Dict[str, Any]:
    """
    Single lightweight LLM call that:
    1. Classifies the query as:
       - 'single_fact': Specific single lookup or direct definition.
       - 'multi_part': Complex multi-faceted question or comparison (e.g. 'What is X and how does it compare to Y?').
       - 'summarization': High-level overview or document synthesis.
    2. Decomposes multi_part queries into 2-4 search-friendly sub-questions.
    """
    if llm is None:
        llm = get_router_llm()

    parser = JsonOutputParser(pydantic_object=QueryPlan)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert query router and query decomposition engine for a RAG system.
Analyze the user's question and classify it into exactly one of:
1. 'single_fact': A targeted lookup for a single fact, definition, or specific piece of information.
2. 'multi_part': A question asking about multiple entities, comparison between concepts, or multi-step logic (e.g. 'What is X and how does it compare to Y?').
3. 'summarization': A request to summarize, provide key takeaways, or give an overall synthesis of the document.

Instructions:
- If 'multi_part': break down the question into 2 to 4 distinct sub-questions that each focus on retrieving a specific piece of the answer.
- If 'single_fact' or 'summarization': keep 'sub_queries' as a list containing only the original query.

Return strictly valid JSON matching this schema:
{format_instructions}"""
        ),
        ("human", "{query}")
    ]).partial(format_instructions=parser.get_format_instructions())

    chain = prompt | llm | parser

    try:
        result = chain.invoke({"query": query})
        return {
            "query_type": result.get("query_type", "single_fact"),
            "sub_queries": result.get("sub_queries", [query]),
            "reasoning": result.get("reasoning", "")
        }
    except Exception as e:
        # Fallback gracefully to single_fact if router encounters any error
        return {
            "query_type": "single_fact",
            "sub_queries": [query],
            "reasoning": f"Fallback due to router exception: {str(e)}"
        }


def retrieve_for_sub_queries(sub_queries: List[str], retriever) -> Tuple[List[Document], Dict[str, List[Dict[str, Any]]]]:
    """
    Executes retrieval for each decomposed sub-query.
    Returns:
    - merged_docs: Deduplicated list of Document objects preserving context richness.
    - evidence: Dict mapping each sub-query -> list of retrieved text chunks & metadata.
    """
    sub_query_evidence: Dict[str, List[Dict[str, Any]]] = {}
    merged_docs: List[Document] = []
    seen_contents = set()

    for sub_q in sub_queries:
        docs = retriever.invoke(sub_q)
        evidence_list = []

        for doc in docs:
            evidence_list.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })
            if doc.page_content not in seen_contents:
                seen_contents.add(doc.page_content)
                merged_docs.append(doc)

        sub_query_evidence[sub_q] = evidence_list

    return merged_docs, sub_query_evidence


def route_and_retrieve(query: str, retriever, llm=None, use_reranker: bool = True, top_n: int = 3) -> Dict[str, Any]:
    """
    High-level entrypoint:
    1. Classifies & decomposes the query.
    2. Retrieves context based on the classification.
    3. Merges context while tracking evidence for each sub-part.
    4. Optionally reranks merged documents using CrossEncoder.
    """
    plan = classify_and_decompose_query(query, llm=llm)
    query_type = plan["query_type"]
    sub_queries = plan["sub_queries"]

    if query_type == "multi_part" and len(sub_queries) > 1:
        merged_docs, evidence = retrieve_for_sub_queries(sub_queries, retriever)
        # Rerank merged documents across sub-queries against the overall query
        if use_reranker and merged_docs:
            from backend.reranker import rerank_documents
            merged_docs = rerank_documents(merged_docs, query, top_n=top_n)
    else:
        # single_fact or summarization
        docs = retriever.invoke(query)
        merged_docs = docs
        evidence = {
            query: [
                {"content": d.page_content, "metadata": d.metadata}
                for d in docs
            ]
        }

    return {
        "query": query,
        "query_type": query_type,
        "sub_queries": sub_queries,
        "reasoning": plan.get("reasoning", ""),
        "merged_docs": merged_docs,
        "evidence_by_sub_query": evidence
    }


if __name__ == "__main__":
    print("Testing Query Router and Decomposition independently...")
    test_q = "What is Superposition and how does it compare to Quantum Entanglement?"
    plan = classify_and_decompose_query(test_q)
    print("Query:", test_q)
    print("Plan:", json.dumps(plan, indent=2))
