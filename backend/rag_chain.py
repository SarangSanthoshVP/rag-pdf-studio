# RAG chain LCEL pipeline module
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables (.env)
load_dotenv()

DEFAULT_SYSTEM_PROMPT = """You are an expert in answering user questions using only the context given below:

{context}

If the user asks something outside of the context, then answer that it is outside of the context

question:
{query}
"""


def format_docs(docs):
    """
    Format list of Document chunks into a string context.
    """
    return "\n\n".join(doc.page_content for doc in docs)


def get_llm(model: str = "gemini-3.5-flash-lite", temperature: float = 0.0, api_key: str = None):
    """
    Initialize Google Generative AI LLM.
    """
    key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY is not set in environment or .env file.")

    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        api_key=key
    )


def create_rag_chain(retriever, llm=None, system_prompt: str = DEFAULT_SYSTEM_PROMPT, use_reranker: bool = False, top_n: int = 3):
    """
    Build the LCEL RAG chain:
    retriever -> (optional cross-encoder reranker) -> format_docs -> prompt -> llm -> StrOutputParser
    """
    if use_reranker:
        from backend.reranker import get_cross_encoder_retriever
        retriever = get_cross_encoder_retriever(retriever, top_n=top_n)

    if llm is None:
        llm = get_llm()

    prompt = ChatPromptTemplate.from_template(system_prompt)

    chain = (
        {
            "context": retriever | format_docs,
            "query": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def ask_question(chain, query: str):
    """
    Invoke the chain with the user query.
    """
    return chain.invoke(query)


def generate_answer_from_docs(docs, query: str, llm=None, system_prompt: str = DEFAULT_SYSTEM_PROMPT, rerank: bool = False, top_n: int = 3) -> str:
    """
    Generate answer directly from a pre-routed/merged list of retrieved Document objects.
    Optionally reranks docs using CrossEncoder if rerank=True.
    """
    if rerank and docs:
        from backend.reranker import rerank_documents
        docs = rerank_documents(docs, query, top_n=top_n)

    if llm is None:
        llm = get_llm()

    prompt = ChatPromptTemplate.from_template(system_prompt)
    context_str = format_docs(docs)

    generator_chain = prompt | llm | StrOutputParser()
    return generator_chain.invoke({"context": context_str, "query": query})