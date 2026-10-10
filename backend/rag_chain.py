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


def get_llm(model: str = "gemini-3.8-flash", temperature: float = 0.0, api_key: str = None):
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


def create_rag_chain(retriever, llm=None, system_prompt: str = DEFAULT_SYSTEM_PROMPT):
    """
    Build the LCEL RAG chain:
    retriever -> format_docs -> prompt -> llm -> StrOutputParser
    """
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