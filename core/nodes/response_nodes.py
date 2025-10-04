# [file name]: core/nodes/response_nodes.py
import logging
import time
from datetime import datetime
from typing import Dict, Any
from langchain_core.runnables import RunnableConfig

from models.state_models import MultiCountryLegalState
from utils.helpers import dict_to_message_obj, message_obj_to_dict

logger = logging.getLogger(__name__)

class ResponseNodes:
    def __init__(self, llm):
        self.llm = llm

    async def response_generation_node(self, state: MultiCountryLegalState, config: RunnableConfig) -> Dict[str, Any]:
        """Generate appropriate responses based on current state"""
        
        assistance_step = state.assistance_step
        
        # Handle assistance workflow responses
        if assistance_step == "collecting_email":
            response_content = """
Je vois que vous souhaitez parler à un avocat. Pour vous aider, j'ai besoin de votre adresse email pour que notre équipe puisse vous contacter.

📧 **Veuillez me fournir votre adresse email :**
"""
            return {
                "messages": [{
                    "role": "assistant",
                    "content": response_content,
                    "meta": {"assistance_step": "collecting_email"}
                }]
            }
        
        elif assistance_step == "collecting_description":
            response_content = f"""
Merci ! Votre email ({state.user_email}) a été enregistré.

📝 **Veuillez maintenant décrire brièvement votre situation :**
- Quelle est votre question juridique ?
- De quel pays s'agit-il ?
- Quel type d'assistance recherchez-vous ?

Cette description aidera notre équipe à mieux vous orienter.
"""
            return {
                "messages": [{
                    "role": "assistant",
                    "content": response_content,
                    "meta": {"assistance_step": "collecting_description"}
                }]
            }
        
        elif assistance_step == "confirming_send":
            response_content = f"""
📋 **RÉCAPITULATIF DE VOTRE DEMANDE :**

📧 **Email :** {state.user_email}
📝 **Description :** {state.assistance_description}

✅ **Confirmez-vous l'envoi de cette demande à notre équipe juridique ?**

Répondez par :
- **"oui"** pour confirmer et envoyer
- **"non"** pour annuler et modifier
"""
            return {
                "messages": [{
                    "role": "assistant",
                    "content": response_content,
                    "meta": {"assistance_step": "confirming_send"}
                }]
            }
        
        else:
            # Default LLM response for non-assistance flows
            return await self._generate_llm_response(state, config)

    async def _generate_llm_response(self, state: MultiCountryLegalState, config: RunnableConfig) -> Dict[str, Any]:
        """Generate LLM-based response for normal conversation flows"""
        try:
            # Synthesize response using LLM
            response_content = await self._synthesize_response(state)
            
            return {
                "messages": [{
                    "role": "assistant",
                    "content": response_content,
                    "meta": {
                        "timestamp": datetime.now().isoformat(),
                        "generated_by": "llm"
                    }
                }]
            }
        except Exception as e:
            logger.error(f"Error generating LLM response: {str(e)}")
            return {
                "messages": [{
                    "role": "assistant",
                    "content": self._create_error_message(str(e)),
                    "meta": {"is_error": True}
                }]
            }

    async def _synthesize_response(self, state: MultiCountryLegalState) -> str:
        """Synthesize final response based on graph execution"""
        s = state.model_dump()
        
        # Build context-aware system prompt
        system_prompt = self._build_system_prompt(state)
        conversation_messages = self._build_conversation_messages(system_prompt, s.get("messages", []))
        
        # Always use LLM to generate final response
        logger.info("🧠 Generating final response with LLM")
        ai_resp = await self.llm.ainvoke(conversation_messages)
        
        return ai_resp.content if hasattr(ai_resp, 'content') else str(ai_resp)

    def _build_system_prompt(self, state: MultiCountryLegalState) -> str:
        """Build context-aware system prompt"""
        s = state.model_dump()
        
        base_prompt = """Vous êtes un assistant juridique expert spécialisé dans le droit du Bénin et de Madagascar.

TÂCHE: Fournir une réponse claire, précise et utile à l'utilisateur.
"""

        # Add legal context if available
        country_name = s.get("legal_context", {}).get("jurisdiction", "Unknown")
        if country_name != "Unknown":
            base_prompt += f"\nCONTEXTE JURIDIQUE: Vous répondez dans le cadre du droit {country_name}.\n"

        # Add search results if available
        search_results = s.get("search_results", "")
        if search_results and "RECHERCHE JURIDIQUE" in search_results:
            base_prompt += f"\nINFORMATIONS JURIDIQUES DISPONIBLES:\n{search_results}\n"
            base_prompt += """
INSTRUCTIONS POUR LA RÉPONSE JURIDIQUE:
- Basez-vous sur les informations juridiques disponibles
- Citez les articles de loi pertinents si possible  
- Soyez précis mais accessible aux non-juristes
- Indiquez si certaines informations manquent
"""
        else:
            base_prompt += """
INSTRUCTIONS GÉNÉRALES:
- Répondez de manière naturelle et utile
- Adaptez votre ton au contexte de la conversation
- Soyez empathique et professionnel
"""

        # Add assistance context if relevant
        if s.get("assistance_requested"):
            base_prompt += "\nCONTEXTE ASSISTANCE: L'utilisateur a demandé à parler à un avocat.\n"
        
        if s.get("approval_status") == "rejected":
            base_prompt += "\nCONTEXTE: La demande d'assistance a été rejetée. Expliquez poliment et proposez des alternatives.\n"
        elif s.get("approval_status") == "approved":
            base_prompt += "\nCONTEXTE: La demande d'assistance a été approuvée. Confirmez et donnez les prochaines étapes.\n"

        return base_prompt

    def _build_conversation_messages(self, system_prompt: str, messages: list) -> list:
        """Build conversation messages for LLM"""
        from langchain_core.messages import SystemMessage
        
        conversation_messages = [SystemMessage(content=system_prompt)]
        
        # Include recent conversation history (last 6 messages)
        recent_messages = messages[-6:] if len(messages) > 6 else messages
        
        # Convert to message objects
        conversation_messages.extend(dict_to_message_obj(m) for m in recent_messages)
        
        return conversation_messages

    async def human_approval_node(self, state: MultiCountryLegalState, config: RunnableConfig) -> Dict[str, Any]:
        """Handle human approval interrupts"""
        logger.info("👨‍⚖️ Human approval node - triggering interrupt")
        
        # For human approval, we still want a meaningful response
        return {
            "approval_status": "pending",
            "messages": [{
                "role": "assistant", 
                "content": "⏳ Votre demande d'assistance nécessite une approbation manuelle. Un modérateur va examiner votre demande.",
                "meta": {"requires_approval": True}
            }]
        }

    async def process_assistance_node(self, state: MultiCountryLegalState, config: RunnableConfig) -> Dict[str, Any]:
        """Process assistance after approval - let LLM generate final message"""
        logger.info("📧 Processing assistance request")
        
        # The LLM will generate the final success message in response_generation_node
        return {
            "email_status": "sent",
            "approval_status": "approved",
            "messages": []  # Empty messages so LLM generates the final response
        }

    def _create_error_message(self, error: str) -> str:
        """Create error message"""
        return f"❌ Désolé, une erreur s'est produite: {error}\n\nVeuillez réessayer ou reformuler votre demande."