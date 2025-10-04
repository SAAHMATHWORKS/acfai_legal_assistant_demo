# [file name]: core/nodes/routing_nodes.py
import logging
from typing import Dict, Any
from langchain_core.runnables import RunnableConfig

from models.state_models import MultiCountryLegalState
from core.router import CountryRouter
from .base_node import BaseNode
from core.prompts.prompt_templates import PromptTemplates

logger = logging.getLogger(__name__)

class RoutingNodes(BaseNode):
    """Router, greeting, and conversation repair nodes"""
    
    def __init__(self, router: CountryRouter, conversation_repair, llm):
        self.router = router
        self.conversation_repair = conversation_repair
        self.llm = llm
        self.prompts = PromptTemplates()
    
    async def router_node(self, state: MultiCountryLegalState, config: RunnableConfig) -> Dict[str, Any]:
        """Enhanced router that detects primary intent with state awareness"""
        try:
            s = state.model_dump()
            
            # CRITICAL: Check if we're continuing an assistance workflow
            # This prevents the router from misclassifying continuation messages
            assistance_step = s.get("assistance_step")
            if assistance_step and assistance_step not in [None, "cancelled", "completed"]:
                logger.info(f"⏩ Bypassing router - continuing assistance at step: {assistance_step}")
                return {
                    "router_decision": "assistance_request",
                    "route_explanation": f"Continuing assistance workflow: {assistance_step}",
                    "assistance_step": assistance_step,  # Ensure step persists
                    "assistance_requested": True
                }
            
            # Normal routing for new messages
            return await self._perform_normal_routing(state, s)
            
        except Exception as e:
            logger.error(f"Router error: {str(e)}")
            legal_context = state.legal_context if hasattr(state, 'legal_context') else {}
            return self._create_router_response("unclear", f"Router error: {str(e)}", legal_context)
    
    async def _perform_normal_routing(self, state: MultiCountryLegalState, state_dict: Dict) -> Dict[str, Any]:
        """Perform normal routing for new user queries"""
        if not state_dict.get("messages"):
            logger.warning("No messages in state for router")
            return self._create_router_response("unclear", "No messages in state", state_dict.get("legal_context", {}))
        
        last_human = self._get_last_human_message(state_dict.get("messages", []))
        if not last_human:
            logger.warning("No user query found in router")
            return self._create_router_response("unclear", "No user query found", state_dict.get("legal_context", {}))

        user_query = last_human.get("content", "").strip()
        if not user_query:
            logger.warning("Empty user query in router")
            return self._create_router_response("unclear", "Empty user query", state_dict.get("legal_context", {}))
        
        logger.info(f"🔀 Routing query: '{user_query[:50]}...'")
        routing_result = await self.router.route_query(user_query, state_dict["messages"])
        
        primary_intent = routing_result.country
        logger.info(f"🎯 Router decision: {primary_intent} ({routing_result.confidence}) - {routing_result.method}")
        
        updated_context = self._update_legal_context(state_dict["legal_context"], primary_intent)
        
        response = {
            "router_decision": primary_intent,
            "route_explanation": f"{routing_result.method}: {routing_result.explanation}",
            "legal_context": updated_context,
            "primary_intent": primary_intent
        }
        
        # If this is an assistance request, initialize the workflow
        if primary_intent == "assistance_request":
            response.update({
                "assistance_step": "collecting_email",
                "assistance_requested": True
            })
        
        return response
    
    async def greeting_small_talk_node(self, state: MultiCountryLegalState, config: RunnableConfig) -> Dict[str, Any]:
        """Handle greetings and small talk"""
        try:
            s = state.model_dump()
            last_human = self._get_last_human_message(s.get("messages", []))
            user_query = last_human.get("content", "").lower() if last_human else ""
            
            logger.info(f"👋 Handling greeting/small_talk: '{user_query[:30]}...'")
            
            greeting_response = self.prompts.generate_greeting_response(user_query)
            
            return {
                "messages": [{
                    "role": "assistant",
                    "content": greeting_response,
                    "meta": {
                        "is_greeting": True,
                        "timestamp": self._get_timestamp()
                    }
                }],
                "search_results": "Greeting handled - no legal search performed"
            }
            
        except Exception as e:
            logger.error(f"Error in greeting node: {str(e)}")
            return self._create_error_state(f"Error in greeting: {str(e)}")
    
    async def conversation_repair_node(self, state: MultiCountryLegalState, config: RunnableConfig) -> Dict[str, Any]:
        """Unified repair handling with LLM"""
        try:
            s = state.model_dump()
            last_human = self._get_last_human_message(s.get("messages", []))
            user_query = last_human.get("content", "") if last_human else ""
            
            logger.info(f"🔧 Handling repair request: '{user_query[:30]}...'")
            
            repair_response = await self.conversation_repair.generate_repair_response(
                user_query, s.get("messages", []), self.llm
            )
            
            return {
                "messages": [{
                    "role": "assistant",
                    "content": repair_response,
                    "meta": {
                        "is_repair_response": True,
                        "timestamp": self._get_timestamp()
                    }
                }],
                "search_results": "Repair handled - no legal search performed"
            }
            
        except Exception as e:
            logger.error(f"Error in repair node: {str(e)}")
            return self._create_error_state(f"Error in repair: {str(e)}")
    
    def _create_router_response(self, decision: str, explanation: str, legal_context: Dict) -> Dict[str, Any]:
        """Create a standardized router response"""
        return {
            "router_decision": decision,
            "route_explanation": explanation,
            "legal_context": legal_context,
            "primary_intent": decision
        }
    
    def _get_last_human_message(self, messages: list) -> Dict[str, Any]:
        """Get the last human message from conversation history"""
        for msg in reversed(messages):
            if msg.get("role") in ["user", "human"]:
                return msg
        return {}
    
    def _update_legal_context(self, legal_context: Dict, primary_intent: str) -> Dict:
        """Update legal context based on routing decision"""
        updated_context = legal_context.copy()
        
        # Map router decisions to detected_country
        country_mapping = {
            "benin": "benin",
            "madagascar": "madagascar",
            "assistance_request": updated_context.get("detected_country", "unknown"),
            "greeting_small_talk": "unknown", 
            "conversation_repair": updated_context.get("detected_country", "unknown"),
            "conversation_summarization": updated_context.get("detected_country", "unknown"),
            "unclear": "unknown",
            "out_of_scope": "unknown"
        }
        
        updated_context["detected_country"] = country_mapping.get(primary_intent, "unknown")
        updated_context["primary_intent"] = primary_intent
        
        return updated_context
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def _create_error_state(self, error_message: str) -> Dict[str, Any]:
        """Create error state response"""
        return {
            "messages": [{
                "role": "assistant", 
                "content": f"❌ Désolé, une erreur s'est produite. Veuillez réessayer.",
                "meta": {"error": error_message}
            }]
        }