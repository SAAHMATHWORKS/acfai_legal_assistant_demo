# [file name]: core/chat_manager.py
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import BaseMessage
from langgraph.types import Command

from config.settings import settings
from models.state_models import MultiCountryLegalState
from utils.helpers import dict_to_message_obj

logger = logging.getLogger(__name__)

class LegalChatManager:
    def __init__(self, graph, checkpointer):
        self.graph = graph
        self.checkpointer = checkpointer
        self.active_sessions = {}
        self.routing_stats = {
            "benin": 0,
            "madagascar": 0,
            "unclear": 0,
            "total_queries": 0
        }
        # Track pending interrupts by session
        self.pending_interrupts = {}

    async def chat(self, message: str, session_id: str, 
                  legal_context: Optional[Dict[str, str]] = None) -> str:
        """Process a chat message with session management and interrupt handling"""
        if not self.graph:
            raise RuntimeError("System not initialized. Call setup_system() first.")

        # Initialize or update session
        self._initialize_session(session_id)
        
        # Check if we have a pending interrupt for this session
        if session_id in self.pending_interrupts:
            return await self._handle_pending_interrupt(session_id, message)

        # Prepare input state
        input_state = self._prepare_input_state(message, session_id, legal_context)
        config = RunnableConfig(
            configurable={"thread_id": session_id},
            recursion_limit=100
        )

        try:
            # Track performance
            start_time = datetime.now()
            
            # Process through graph
            result = await self.graph.ainvoke(MultiCountryLegalState(**input_state), config)
            
            # Check for interrupt
            state_snapshot = await self.graph.aget_state(config)
            if state_snapshot and state_snapshot.next:
                # Graph is paused at an interrupt
                logger.info(f"⏸️ Graph interrupted at: {state_snapshot.next}")
                self.pending_interrupts[session_id] = {
                    "type": "human_approval",
                    "config": config,
                    "created_at": datetime.now(),
                    "paused_at": state_snapshot.next
                }
                return self._get_approval_prompt_message(result)
            
            # Track performance
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_session_stats(session_id, processing_time)
            
            # Extract and return response
            response = self._extract_response(result)
            self._update_routing_stats(response)
            
            return response
            
        except Exception as e:
            logger.exception(f"Chat error for session {session_id}")
            self._log_error(session_id, str(e))
            return f"Erreur lors du traitement: {str(e)}"

    async def _handle_pending_interrupt(self, session_id: str, message: str) -> str:
        """Handle user response to a pending interrupt using Command(resume=...)"""
        interrupt_data = self.pending_interrupts.get(session_id)
        if not interrupt_data:
            return "Erreur: Aucune interruption en attente."
        
        try:
            logger.info(f"📥 Resuming graph with moderator decision: {message}")
            
            config = interrupt_data["config"]
            
            # CRITICAL FIX: Use Command(resume=...) to properly resume from interrupt
            # This sends the user's message back to the interrupt() call
            result = await self.graph.ainvoke(
                Command(resume=message),
                config
            )
            
            # Clean up the pending interrupt
            del self.pending_interrupts[session_id]
            
            # Extract and return final response
            response = self._extract_response(result)
            self._update_routing_stats(response)
            
            logger.info(f"✅ Graph resumed successfully for session {session_id}")
            return response
            
        except Exception as e:
            logger.error(f"Error resuming from interrupt: {str(e)}")
            # Clean up on error
            if session_id in self.pending_interrupts:
                del self.pending_interrupts[session_id]
            return f"Erreur lors du traitement de la décision: {str(e)}"

    def _get_approval_prompt_message(self, state) -> str:
        """Generate message asking for human approval"""
        # Extract metadata from state
        if isinstance(state, MultiCountryLegalState):
            state_dict = state.model_dump()
        elif isinstance(state, dict):
            state_dict = state
        else:
            state_dict = {}
        
        user_email = state_dict.get("user_email", "Non spécifié")
        country = state_dict.get("legal_context", {}).get("detected_country", "Non spécifié")
        description = state_dict.get("assistance_description", "Non spécifié")
        
        return f"""
🔒 **APPROBATION HUMAINE REQUISE**

📧 **Utilisateur**: {user_email}
🌍 **Pays**: {country}
📝 **Description**: {description}

**Veuillez répondre avec:**
- "approve [raison]" pour approuver la demande
- "reject [raison]" pour rejeter la demande

**Exemples:**
- "approve Demande légitime de consultation"
- "reject Email invalide ou description trop vague"

**Votre décision:**
"""

    # === EXISTING METHODS (unchanged) ===

    async def get_conversation_history(self, session_id: str) -> List[BaseMessage]:
        """Get conversation history for a session"""
        if not self.graph:
            return []
        
        config = RunnableConfig(configurable={"thread_id": session_id})
        
        try:
            state = await self.graph.aget_state(config)
            if not state or not state.values:
                return []
            
            s = state.values
            if isinstance(s, MultiCountryLegalState):
                s = s.model_dump()
            elif isinstance(s, dict):
                pass
            else:
                s = {}

            raw_messages = s.get("messages", [])
            return [dict_to_message_obj(m) for m in raw_messages if isinstance(m, dict)]
            
        except Exception as e:
            logger.exception(f"Error getting conversation history for session {session_id}")
            return []

    def get_session_stats(self, session_id: str) -> Dict:
        """Get statistics for a specific session"""
        return self.active_sessions.get(session_id, {})

    def get_global_stats(self) -> Dict:
        """Get global system statistics"""
        return {
            "routing_stats": self.routing_stats,
            "active_sessions": len(self.active_sessions),
            "total_queries": self.routing_stats["total_queries"],
            "pending_interrupts": len(self.pending_interrupts)
        }

    def _initialize_session(self, session_id: str):
        """Initialize or update session tracking"""
        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = {
                "created": datetime.now(),
                "query_count": 0,
                "total_processing_time": 0,
                "average_processing_time": 0,
                "detected_countries": set(),
                "last_activity": datetime.now()
            }
        
        session_info = self.active_sessions[session_id]
        session_info["query_count"] += 1
        session_info["last_activity"] = datetime.now()

    def _prepare_input_state(self, message: str, session_id: str, 
                           legal_context: Optional[Dict[str, str]]) -> Dict:
        """Prepare input state for graph processing"""
        ctx = legal_context or {
            "jurisdiction": "Unknown", 
            "user_type": "general", 
            "document_type": "legal",
            "detected_country": "unknown"
        }
        
        if ctx.get("detected_country") is None:
            ctx["detected_country"] = "unknown"

        return {
            "messages": [{"role": "user", "content": message, "meta": {}}],
            "legal_context": ctx,
            "session_id": session_id,
            "router_decision": None,
            "search_results": None,
            "route_explanation": None,
            "last_search_query": None,
            "detected_articles": [],
        }

    def _extract_response(self, result) -> str:
        """Extract response text from graph result"""
        if isinstance(result, MultiCountryLegalState):
            r = result.model_dump()
        elif isinstance(result, dict):
            r = result
        else:
            r = {}

        msgs = r.get("messages", [])
        for m in reversed(msgs):
            if (m.get("role") or "").lower() in ("assistant", "ai"):
                return m.get("content", "")
        
        return "Désolé, je n'ai pas pu générer de réponse."

    def _update_session_stats(self, session_id: str, processing_time: float):
        """Update session statistics with processing time"""
        if session_id in self.active_sessions:
            session_info = self.active_sessions[session_id]
            session_info["total_processing_time"] += processing_time
            session_info["average_processing_time"] = (
                session_info["total_processing_time"] / session_info["query_count"]
            )

    def _update_routing_stats(self, response: str):
        """Update routing statistics based on response content"""
        self.routing_stats["total_queries"] += 1
        
        response_lower = response.lower()
        if any(keyword in response_lower for keyword in ["bénin", "béninois", "béninoise"]):
            self.routing_stats["benin"] += 1
        elif any(keyword in response_lower for keyword in ["madagascar", "malgache", "malagasy"]):
            self.routing_stats["madagascar"] += 1
        else:
            self.routing_stats["unclear"] += 1

    def _log_error(self, session_id: str, error: str):
        """Log error for monitoring"""
        logger.error(f"Session {session_id}: {error}")

    def cleanup_inactive_sessions(self, max_age_hours: int = 24):
        """Clean up sessions that have been inactive for too long"""
        cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)
        
        inactive_sessions = [
            session_id for session_id, info in self.active_sessions.items()
            if info["last_activity"].timestamp() < cutoff_time
        ]
        
        # Also clean up pending interrupts for inactive sessions
        for session_id in inactive_sessions:
            if session_id in self.pending_interrupts:
                del self.pending_interrupts[session_id]
            del self.active_sessions[session_id]
            logger.info(f"Cleaned up inactive session: {session_id}")