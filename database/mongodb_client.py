from pymongo import MongoClient, ReadPreference
from pymongo.errors import ServerSelectionTimeoutError, ConnectionFailure
from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch
from langchain_openai import OpenAIEmbeddings
from typing import Dict
import logging
from config.settings import settings

logger = logging.getLogger(__name__)

class MongoDBClient:
    def __init__(self):
        self.client = None
        self.db = None
        self.benin_collection = None
        self.madagascar_collection = None
        self.benin_vectorstore = None
        self.madagascar_vectorstore = None
        self.embedding_model = None

    def connect(self):
        """Connect to MongoDB and initialize collections"""
        try:
            # CRITICAL FIX: Add read preference to allow reading from secondary nodes
            self.client = MongoClient(
                settings.MONGO_URI,
                
                # Allow reading from secondary nodes when primary is unavailable
                read_preference=ReadPreference.SECONDARY_PREFERRED,
                
                # Reduce timeouts to fail faster (instead of 30s)
                serverSelectionTimeoutMS=10000,  # 10 seconds
                connectTimeoutMS=10000,
                socketTimeoutMS=10000,
                
                # Retry configuration
                retryWrites=True,
                retryReads=True,
                
                # Connection pool settings
                maxPoolSize=50,
                minPoolSize=10,
                
                # Write concern (for writes to still work)
                w='majority',
                journal=True
            )
            
            # Test the connection
            self.client.admin.command('ping')
            logger.info("✅ MongoDB connection test successful")
            
            self.db = self.client[settings.DATABASE_NAME]
            
            # Initialize collections
            self.benin_collection = self.db[settings.BENIN_COLLECTION]
            self.madagascar_collection = self.db[settings.MADAGASCAR_COLLECTION]
            
            # Verify collections exist and have data
            benin_count = self.benin_collection.count_documents({})
            madagascar_count = self.madagascar_collection.count_documents({})
            logger.info(f"📊 Bénin collection: {benin_count} documents")
            logger.info(f"📊 Madagascar collection: {madagascar_count} documents")
            
            # Initialize embedding model
            self.embedding_model = OpenAIEmbeddings(
                model=settings.EMBEDDING_MODEL,
                openai_api_key=settings.OPENAI_API_KEY
            )
            
            # Initialize vector stores with read preference
            self.benin_vectorstore = MongoDBAtlasVectorSearch(
                collection=self.benin_collection,
                embedding=self.embedding_model,
                index_name=settings.VECTOR_INDEX_NAME,
                text_key=settings.TEXT_KEY,
                embedding_key=settings.EMBEDDING_KEY,
            )
            
            self.madagascar_vectorstore = MongoDBAtlasVectorSearch(
                collection=self.madagascar_collection,
                embedding=self.embedding_model,
                index_name=settings.VECTOR_INDEX_NAME,
                text_key=settings.TEXT_KEY,
                embedding_key=settings.EMBEDDING_KEY,
            )
            
            print("✅ MongoDB connected successfully with SECONDARY_PREFERRED read preference")
            return True
            
        except (ServerSelectionTimeoutError, ConnectionFailure) as e:
            logger.error(f"❌ MongoDB connection failed: {e}")
            logger.error("🔍 Possible issues:")
            logger.error("   1. MongoDB Atlas cluster is paused")
            logger.error("   2. Network connectivity issues")
            logger.error("   3. IP address not whitelisted in Atlas")
            logger.error("   4. Cluster is undergoing maintenance")
            print(f"❌ MongoDB connection failed: {e}")
            return False
            
        except Exception as e:
            logger.error(f"❌ Unexpected error during MongoDB connection: {e}")
            print(f"❌ MongoDB connection failed: {e}")
            return False

    def get_collection_stats(self) -> Dict:
        """Get statistics for both collections"""
        if not self.client:
            return {}
        
        try:
            benin_count = self.benin_collection.count_documents({})
            madagascar_count = self.madagascar_collection.count_documents({})
            
            # Sample document to check schema
            benin_sample = self.benin_collection.find_one()
            madagascar_sample = self.madagascar_collection.find_one()
            
            # Check for documents by doc_type
            benin_case_study_count = self.benin_collection.count_documents({"doc_type": "case_study"})
            benin_articles_count = self.benin_collection.count_documents({"doc_type": "articles"})
            madagascar_case_study_count = self.madagascar_collection.count_documents({"doc_type": "case_study"})
            madagascar_articles_count = self.madagascar_collection.count_documents({"doc_type": "articles"})
            
            return {
                "benin": {
                    "total_documents": benin_count,
                    "case_study_count": benin_case_study_count,
                    "articles_count": benin_articles_count,
                    "has_embeddings": bool(benin_sample and 'vecteur_embedding' in benin_sample),
                    "sample_fields": list(benin_sample.keys()) if benin_sample else [],
                    "sample_doc_type": benin_sample.get('doc_type', 'NOT_SET') if benin_sample else None
                },
                "madagascar": {
                    "total_documents": madagascar_count,
                    "case_study_count": madagascar_case_study_count,
                    "articles_count": madagascar_articles_count,
                    "has_embeddings": bool(madagascar_sample and 'vecteur_embedding' in madagascar_sample),
                    "sample_fields": list(madagascar_sample.keys()) if madagascar_sample else [],
                    "sample_doc_type": madagascar_sample.get('doc_type', 'NOT_SET') if madagascar_sample else None
                }
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            print(f"Error getting collection stats: {e}")
            return {}

    def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("✅ MongoDB connection closed")
            print("✅ MongoDB connection closed")