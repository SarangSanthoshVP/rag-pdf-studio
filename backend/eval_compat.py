"""
Ragas & Python 3.14 / LangChain compatibility bridge.
Provides stubs for deprecated langchain_community modules and patches nest_asyncio
for Python 3.14 task timeout compatibility.
"""
import sys
import types

def ensure_ragas_compatibility():
    """Ensure modules expected by Ragas can be imported cleanly without error."""
    # 1. Langchain-community compatibility stubs (removed in langchain-community 0.3+)
    if "langchain_community.chat_models.vertexai" not in sys.modules:
        stub_chat = types.ModuleType("langchain_community.chat_models.vertexai")
        stub_chat.ChatVertexAI = type("ChatVertexAI", (), {})
        sys.modules["langchain_community.chat_models.vertexai"] = stub_chat

    if "langchain_community.llms.vertexai" not in sys.modules:
        stub_llm = types.ModuleType("langchain_community.llms.vertexai")
        stub_llm.VertexAI = type("VertexAI", (), {})
        sys.modules["langchain_community.llms.vertexai"] = stub_llm

    # 2. nest_asyncio compatibility: In Python 3.14, nest_asyncio breaks asyncio.timeouts
    # Making nest_asyncio.apply a no-op restores native Python 3.14 async task handling.
    try:
        import nest_asyncio
        nest_asyncio.apply = lambda *args, **kwargs: None
    except ImportError:
        pass
