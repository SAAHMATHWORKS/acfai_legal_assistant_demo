# [file name]: core/nodes/helper_nodes.py
import logging
from typing import Dict, Any, List
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage

from models.state_models import MultiCountryLegalState
from .base_node import BaseNode
from core.prompts.prompt_templates import PromptTemplates

logger = logging.getLogger(__name__)

class HelperNodes(BaseNode):
    """Helper nodes for unclear routes and summarization"""
    
    def __init__(self, llm):
        self.llm = llm
        self.prompts = PromptTemplates()
    
    async def out_of_scope_node(self, state: MultiCountryLegalState, config: RunnableConfig) -> Dict[str, Any]:
        """Handle out-of-scope questions - redirect to legal domain"""
        try:
            logger.info("🚫 Out of scope question detected")
            
            redirect_message = {
                "role": "assistant",
                "content": (
                    "Je suis un assistant juridique spécialisé dans le droit du Bénin et de Madagascar. "
                    "Je ne peux répondre qu'aux questions relatives au droit et aux procédures juridiques.\n\n"
                    "Comment puis-je vous aider avec vos questions juridiques ?"
                ),
                "meta": {
                    "is_out_of_scope": True,
                    "timestamp": self._get_timestamp()
                }
            }
            
            return {
                "messages": [redirect_message],
                "current_country": "out_of_scope",
                "search_results": "Out of scope query - no legal search performed"
            }
            
        except Exception as e:
            logger.error(f"Error in out_of_scope handler: {str(e)}")
            return self._create_error_state(f"Error in out_of_scope: {str(e)}")
    
    async def unclear_route_node(self, state: MultiCountryLegalState, config: RunnableConfig) -> Dict[str, Any]:
        """Handle unclear routing cases - for ambiguous legal queries"""
        try:
            s = state.model_dump()
            route_explanation = s.get("route_explanation", "")
            
            # This is now only for unclear LEGAL queries
            clarification_msg = {
                "role": "assistant", 
                "content": self.prompts.get_clarification_message(),
                "meta": {
                    "requires_clarification": True,
                    "timestamp": self._get_timestamp()
                }
            }
            
            return {
                "messages": [clarification_msg],
                "search_results": "Country clarification needed"
            }
            
        except Exception as e:
            logger.error(f"Error in unclear route handling: {str(e)}")
            return self._create_error_state(f"Error in unclear route: {str(e)}")
    
    async def conversation_summarization_node(self, state: MultiCountryLegalState, config: RunnableConfig) -> Dict[str, Any]:
        """Generate summary of conversation history"""
        try:
            s = state.model_dump()
            messages = s.get("messages", [])
            
            logger.info(f"📋 Generating conversation summary for {len(messages)} messages")
            
            summary = await self._generate_conversation_summary(messages)
            
            return {
                "messages": [{
                    "role": "assistant",
                    "content": summary,
                    "meta": {
                        "is_summary": True,
                        "conversation_length": len(messages),
                        "timestamp": self._get_timestamp()
                    }
                }],
                "search_results": "Conversation summary generated - no legal search performed"
            }
            
        except Exception as e:
            logger.error(f"Error in conversation summarization: {str(e)}")
            return self._create_error_state(f"Error in summarization: {str(e)}")
    
    async def _generate_conversation_summary(self, messages: List[Dict]) -> str:
        """Use LLM to generate conversation summary"""
        conversation_messages = [
            msg for msg in messages 
            if msg.get("role") in ["user", "assistant"]
        ]
        
        if len(conversation_messages) <= 2:
            return "Notre conversation vient juste de commencer. Nous n'avons pas encore beaucoup échangé."
        
        conversation_text = ""
        for i, msg in enumerate(conversation_messages):
            role = "Utilisateur" if msg.get("role") == "user" else "Assistant"
            content = msg.get("content", "")
            conversation_text += f"{role}: {content}\n\n"
        
        summary_prompt = f"""
Vous êtes un assistant juridique. Résumez la conversation suivante entre l'utilisateur et vous-même.

**CONVERSATION:**
{conversation_text}

**INSTRUCTIONS:**
- Faites un résumé concis et clair
- Mettez en évidence les points juridiques principaux discutés
- Mentionnez les pays concernés (Bénin/Madagascar) si pertinents
- Gardez un ton professionnel mais accessible
- Maximum 5-7 phrases

**RÉSUMÉ:**
"""
        
        try:
            response = await self.llm.ainvoke([HumanMessage(content=summary_prompt)])
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            logger.error(f"LLM summarization failed: {e}")
            user_messages = [m for m in conversation_messages if m.get("role") == "user"]
            assistant_messages = [m for m in conversation_messages if m.get("role") == "assistant"]
            
            return f"""**Résumé de notre conversation:**

- **Échanges totaux**: {len(conversation_messages)} messages
- **Questions de l'utilisateur**: {len(user_messages)}
- **Réponses fournies**: {len(assistant_messages)}
- **Dernier échange**: {conversation_messages[-1].get('content', '')[:100]}...

*Pour un résumé détaillé, veuillez reposer votre question.*"""