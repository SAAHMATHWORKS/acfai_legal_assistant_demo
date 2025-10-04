# [file name]: core/nodes/base_node.py
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from langchain_core.runnables import RunnableConfig

from models.state_models import MultiCountryLegalState
from utils.helpers import dict_to_message_obj

logger = logging.getLogger(__name__)

class BaseNode:
    """Base class with common utilities for all nodes"""
    
    def _get_last_human_message(self, messages: List[Dict]) -> Optional[Dict]:
        """Get the last human message from conversation"""
        if not messages:
            return None
        for msg in reversed(messages):
            if msg.get("role", "").lower() in ("user", "human"):
                return msg
        return None
    
    def _has_complete_response(self, messages: List[Dict]) -> bool:
        """Check if there's already an assistant response in recent messages"""
        if not messages:
            return False
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and msg.get("content"):
                return True
        return False
    
    def _create_error_message(self, error: str) -> Dict[str, Any]:
        """Create standardized error message"""
        return {
            "role": "assistant",
            "content": f"Désolé, une erreur s'est produite lors du traitement de votre demande: {error}",
            "meta": {
                "is_error": True,
                "timestamp": self._get_timestamp()
            }
        }
    
    def _create_error_state(self, error: str) -> Dict[str, Any]:
        """Create error state with message"""
        return {
            "messages": [self._create_error_message(error)],
            "search_results": f"Error: {error}"
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for message metadata"""
        return datetime.now().isoformat()
    
    def _update_legal_context(self, legal_context: Dict, country: str) -> Dict:
        """Update legal context with country information"""
        updated = legal_context.copy() if legal_context else {}
        
        if country in ["benin", "madagascar"]:
            updated["detected_country"] = country
            if country == "benin":
                updated["jurisdiction"] = "Bénin"
            elif country == "madagascar":
                updated["jurisdiction"] = "Madagascar"
        else:
            updated["jurisdiction"] = "Unknown"
            updated["detected_country"] = "unknown"
            
        return updated
    
    def _create_router_response(self, country: str, explanation: str, legal_context: Dict) -> Dict[str, Any]:
        """Create standardized router response"""
        updated_context = self._update_legal_context(legal_context, country)
        return {
            "router_decision": country,
            "route_explanation": explanation,
            "legal_context": updated_context,
            "primary_intent": country
        }