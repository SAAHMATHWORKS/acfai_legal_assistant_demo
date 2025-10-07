# [file name]: core/nodes/retrieval_nodes.py
import logging
from typing import Dict, Any
from langchain_core.runnables import RunnableConfig

from models.state_models import MultiCountryLegalState
from core.retriever import LegalRetriever

logger = logging.getLogger(__name__)

class RetrievalNodes:
    """Scalable legal retrieval nodes for any number of countries"""
    
    def __init__(self, country_retrievers: Dict[str, LegalRetriever]):
        self.country_retrievers = country_retrievers
    
    async def country_retrieval_node(self, state: MultiCountryLegalState, config: RunnableConfig, country_code: str) -> Dict[str, Any]:
        """Generic country retrieval for any country"""
        try:
            if country_code not in self.country_retrievers:
                logger.error(f"❌ Country not configured: {country_code}")
                return {
                    "search_results": f"Country {country_code} not available",
                    "detected_articles": [],
                    "supplemental_message": f"Pays {country_code} non configuré dans le système."
                }
            
            retriever = self.country_retrievers[country_code]
            s = state.model_dump()
            last_human = self._get_last_human_message(s.get("messages", []))
            
            if not last_human:
                return {
                    "search_results": f"No query for {country_code} retrieval", 
                    "detected_articles": [],
                    "supplemental_message": "Aucune requête trouvée pour la recherche."
                }

            user_query = last_human.get("content", "").strip()
            if not user_query:
                return {
                    "search_results": f"Empty query for {country_code} retrieval", 
                    "detected_articles": [],
                    "supplemental_message": "Requête vide pour la recherche."
                }
            
            logger.info(f"🌍 Performing {country_code} retrieval for: '{user_query[:50]}...'")
            
            enhanced_docs, detected_articles, applied_filters, supplemental_message = await retriever.smart_legal_query(user_query, country_code)
            
            search_results = retriever.format_search_results(
                user_query, enhanced_docs, detected_articles, applied_filters, country_code, supplemental_message
            )
            
            logger.info(f"📚 Retrieved {len(enhanced_docs)} documents for {country_code}")
            
            return {
                "search_results": search_results,
                "detected_articles": detected_articles,
                "last_search_query": user_query,
                "supplemental_message": supplemental_message,  # Pass the supplemental message to state
                # Store complex data in search_metadata instead of legal_context
                "search_metadata": {
                    "applied_filters": applied_filters,
                    "documents_count": len(enhanced_docs),
                    "supplemental_message": supplemental_message
                }
            }
            
        except Exception as e:
            logger.error(f"Error in {country_code} retrieval: {str(e)}")
            return {
                "search_results": f"Erreur lors de la recherche {country_code}: {str(e)}",
                "detected_articles": [],
                "supplemental_message": f"Erreur lors de la recherche: {str(e)}"
            }
    
    def _get_last_human_message(self, messages: list) -> Dict[str, Any]:
        """Get the last human message"""
        for msg in reversed(messages):
            if msg.get("role") in ["user", "human"]:
                return msg
        return {}