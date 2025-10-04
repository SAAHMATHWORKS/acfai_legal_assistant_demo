# [file name]: core/conversation_repair.py
import logging
from typing import Dict, List, Optional, Any
import re

logger = logging.getLogger(__name__)

class ConversationRepair:
    def __init__(self):
        self.meta_keywords = [
            "pas compris", "mal compris", "reformuler", "autrement", 
            "différemment", "répéter", "redire", "expliquer autrement",
            "plus simple", "plus clair", "clarifier", "précisez",
            "explique mieux", "développe", "approfondis", "que veux-tu dire",
            "c'est-à-dire", "concrètement", "en pratique", "recommence",
            "ce n'est pas ça", "tu n'as pas compris", "erreur", "faux"
        ]
    
    def detect_repair_intent(self, query: str, conversation_history: List[Dict]) -> bool:
        """Simple detection - just check if this is a repair request"""
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in self.meta_keywords)

    async def generate_repair_response(self, query: str, conversation_history: List[Dict], llm) -> str:
        """Unified LLM-powered repair handling"""
        try:
            # Build conversation context
            context = self._build_conversation_context(conversation_history)
            
            repair_prompt = self._build_repair_prompt(query, context)
            
            # Use LLM for intelligent repair response
            from langchain_core.messages import HumanMessage
            response = await llm.ainvoke([HumanMessage(content=repair_prompt)])
            
            return response.content if hasattr(response, 'content') else str(response)
            
        except Exception as e:
            logger.error(f"LLM repair generation failed: {e}")
            return self._generate_fallback_response()

    def _build_conversation_context(self, conversation_history: List[Dict]) -> str:
        """Build conversation context for LLM"""
        if not conversation_history:
            return "Aucun contexte de conversation"
        
        # Get relevant conversation history
        relevant_messages = conversation_history[-6:]  # Last 6 messages
        
        context_lines = []
        for msg in relevant_messages:
            role = "Utilisateur" if msg.get("role") == "user" else "Assistant"
            content = msg.get("content", "")
            context_lines.append(f"{role}: {content}")
        
        return "\n".join(context_lines)

    def _build_repair_prompt(self, current_query: str, conversation_context: str) -> str:
        """Build intelligent repair prompt"""
        return f"""
Vous êtes un assistant juridique expert. L'utilisateur exprime un problème de compréhension ou demande une clarification.

**CONTEXTE DE LA CONVERSATION:**
{conversation_context}

**REQUÊTE ACTUELLE DE L'UTILISATEUR:**
"{current_query}"

**ANALYSE REQUISE:**
1. Identifiez le type de problème : incompréhension, besoin de clarification, reformulation, correction d'erreur
2. Analysez quel aspect précis pose problème dans la conversation
3. Adaptez votre réponse au contexte juridique si pertinent

**INSTRUCTIONS POUR LA RÉPONSE:**
- Accusez réception du problème de compréhension
- Fournissez une clarification adaptée et utile
- Si c'est juridique, simplifiez sans perdre la précision légale
- Utilisez des exemples concrets si pertinent
- Proposez des pistes pour avancer
- Gardez un ton professionnel et empathique
- Maximum 5-7 phrases

**RÉPONSE:**
"""

    def _generate_fallback_response(self) -> str:
        """Fallback if LLM fails"""
        return "Je m'excuse pour ce malentendu. Pouvez-vous reformuler votre demande ou préciser ce qui n'était pas clair ?"