# [file name]: core/system_initializer.py
import logging
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from core.graph_builder import GraphBuilder
from core.chat_manager import LegalChatManager
from core.router import CountryRouter
from database.mongodb_client import MongoDBClient
from database.postgres_checkpointer import PostgresCheckpointer
from langchain_openai import ChatOpenAI
from config import settings  # Make sure this import is correct

logger = logging.getLogger(__name__)

async def setup_system():
    """Initialize the legal assistant system for API use"""
    
    try:
        # 1. Initialize MongoDB using your existing class
        mongo_client = MongoDBClient()
        if not mongo_client.connect():
            raise Exception("MongoDB connection failed")
        
        logger.info("✅ MongoDB connected successfully")
        
        # 2. Use your existing vector stores directly from the client
        vector_store_benin = mongo_client.benin_vectorstore
        collection_benin = mongo_client.benin_collection
        vector_store_madagascar = mongo_client.madagascar_vectorstore
        collection_madagascar = mongo_client.madagascar_collection
        
        # 3. Initialize retrievers
        from core.retriever import LegalRetriever
        benin_retriever = LegalRetriever(vector_store_benin, collection_benin)
        madagascar_retriever = LegalRetriever(vector_store_madagascar, collection_madagascar)
        
        country_retrievers = {
            "benin": benin_retriever,
            "madagascar": madagascar_retriever
        }
        
        # 4. Initialize LLM and router
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            max_tokens=2000,
            streaming=True
        )
        
        router = CountryRouter()
        
        # 5. Initialize PostgreSQL checkpointer - FIXED DATABASE URL
        # Check what database URL setting you have
        database_url = getattr(settings, 'DATABASE_URL', None)
        
        if not database_url:
            # Try alternative setting names
            database_url = getattr(settings, 'POSTGRES_URL', None) or \
                          getattr(settings, 'POSTGRESQL_URL', None) or \
                          getattr(settings, 'DB_URL', None)
        
        if not database_url:
            raise Exception("No database URL found in settings")
            
        logger.info(f"🔗 Using database URL: {database_url.split('@')[-1] if '@' in database_url else 'local'}")  # Log safely
        
        postgres_checkpointer = PostgresCheckpointer(
            database_url=database_url,  # Use actual database URL
            max_connections=10,
            min_connections=2
        )
        
        if not await postgres_checkpointer.initialize():
            raise Exception("PostgreSQL checkpointer initialization failed")
            
        checkpointer = postgres_checkpointer.get_checkpointer()
        logger.info("✅ PostgreSQL checkpointer initialized for API")
        
        # 6. Build graph
        graph_builder = GraphBuilder(
            router=router,
            llm=llm,
            checkpointer=checkpointer,
            country_retrievers=country_retrievers
        )
        
        workflow = graph_builder.build_graph()
        app = workflow.compile(checkpointer=checkpointer)
        
        # 7. Initialize chat manager
        chat_manager = LegalChatManager(app, checkpointer)
        
        logger.info("✅ API System initialized successfully")
        
        return {
            "chat_manager": chat_manager,
            "graph": app,
            "checkpointer": checkpointer
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize system: {e}")
        raise