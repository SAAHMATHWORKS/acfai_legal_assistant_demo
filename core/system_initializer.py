# [file name]: core/system_initializer.py
import logging
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from core.graph_builder import GraphBuilder
from core.chat_manager import LegalChatManager
from core.router import CountryRouter
from database.mongodb_client import get_mongodb_client, get_vector_store
from database.postgres_checkpointer import get_postgres_checkpointer
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

async def setup_system():
    """Initialize the legal assistant system for API use"""
    
    # 1. Initialize MongoDB
    mongodb_client = await get_mongodb_client()
    vector_store_benin, collection_benin = await get_vector_store("benin")
    vector_store_madagascar, collection_madagascar = await get_vector_store("madagascar")
    
    # 2. Initialize retrievers
    from core.retriever import LegalRetriever
    benin_retriever = LegalRetriever(vector_store_benin, collection_benin)
    madagascar_retriever = LegalRetriever(vector_store_madagascar, collection_madagascar)
    
    country_retrievers = {
        "benin": benin_retriever,
        "madagascar": madagascar_retriever
    }
    
    # 3. Initialize LLM and router
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,
        max_tokens=2000,
        streaming=True  # Enable streaming for API
    )
    
    router = CountryRouter(llm)
    
    # 4. Initialize PostgreSQL checkpointer
    checkpointer = await get_postgres_checkpointer()
    logger.info("✅ PostgreSQL checkpointer initialized for API")
    
    # 5. Build graph
    graph_builder = GraphBuilder(
        router=router,
        llm=llm,
        checkpointer=checkpointer,
        country_retrievers=country_retrievers
    )
    
    workflow = graph_builder.build_graph()
    app = workflow.compile(checkpointer=checkpointer)
    
    # 6. Initialize chat manager
    chat_manager = LegalChatManager(app, checkpointer)
    
    logger.info("✅ API System initialized successfully")
    
    return {
        "chat_manager": chat_manager,
        "graph": app,
        "checkpointer": checkpointer
    }