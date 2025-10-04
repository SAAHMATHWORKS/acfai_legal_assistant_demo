# [file name]: models/state_models.py
from typing import List, Dict, Any, Optional, Annotated, Literal, Union
from pydantic import BaseModel, Field
import operator

class MultiCountryLegalState(BaseModel):
    messages: Annotated[List[Dict[str, Any]], operator.add] = Field(default_factory=list)
    legal_context: Dict[str, str] = Field(
        default_factory=lambda: {
            "jurisdiction": "Unknown",
            "user_type": "general", 
            "document_type": "legal",
            "detected_country": "unknown"
        }
    )
    session_id: Optional[str] = None
    last_search_query: Optional[str] = None
    detected_articles: Annotated[List[str], operator.add] = Field(default_factory=list)
    router_decision: Optional[str] = None
    search_results: Optional[str] = None
    route_explanation: Optional[str] = None
    country: Optional[str] = Field(default=None)  # Add explicit country field
    
    # Assistance email fields
    assistance_requested: bool = Field(default=False)
    user_email: Optional[str] = None
    assistance_description: Optional[str] = None
    email_status: Optional[str] = None  # "pending", "sent", "error"
    assistance_step: Optional[str] = Field(default=None)  # "collecting_email", "collecting_description", "confirming_send"
    pending_assistance_data: Dict[str, Any] = Field(default_factory=dict) 

    # Conversation repair tracking
    repair_type: Optional[str] = None
    original_query: Optional[str] = None
    misunderstanding_count: int = Field(default=0)
    
    # Enhanced routing support
    primary_intent: Optional[str] = Field(default=None)
    
    # NEW: Human approval fields
    approval_status: Optional[str] = Field(default=None)  # "pending", "approved", "rejected"
    approval_reason: Optional[str] = Field(default=None)
    approved_by: Optional[str] = Field(default=None)
    approval_timestamp: Optional[str] = Field(default=None)

    # Conversation summary fields
    summary_generated: bool = Field(default=False)
    last_summary_timestamp: Optional[str] = Field(default=None)

    @staticmethod
    def detect_country(text: str) -> str:
        """
        Detect country from text based on keywords.
        
        Args:
            text: User input text to analyze
            
        Returns:
            Country code: "benin", "madagascar", or "unknown"
        """
        if not text:
            return "unknown"
            
        text_lower = text.lower()
        
        # Benin keywords
        benin_keywords = [
            "bénin", "benin", "béninois", "béninoise",
            "cotonou", "porto-novo", "porto novo",
            "dahomey"  # Historical name
        ]
        
        # Madagascar keywords
        madagascar_keywords = [
            "madagascar", "malgache", "malagasy",
            "antananarivo", "tananarive", "tana",
            "toamasina", "tamatave"
        ]
        
        # Check for country mentions
        benin_score = sum(1 for keyword in benin_keywords if keyword in text_lower)
        madagascar_score = sum(1 for keyword in madagascar_keywords if keyword in text_lower)
        
        if benin_score > madagascar_score and benin_score > 0:
            return "benin"
        elif madagascar_score > benin_score and madagascar_score > 0:
            return "madagascar"
        
        return "unknown"


class RoutingResult(BaseModel):
    country: Literal["benin", "madagascar", "unclear", "greeting_small_talk",
                 "conversation_repair", "assistance_request", "conversation_summarization", "out_of_scope"]
    confidence: Literal["high", "medium", "low"] 
    method: str
    explanation: str

class SearchResult(BaseModel):
    documents: List[Any]
    detected_articles: List[str]
    applied_filters: Dict[str, Any]
    query: str
    country: str