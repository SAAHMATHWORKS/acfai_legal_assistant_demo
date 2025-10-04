# core/assistance/workflow_nodes.py
import logging
import re
from typing import Dict, Any, List
from langchain_core.runnables import RunnableConfig
from models.state_models import MultiCountryLegalState

logger = logging.getLogger(__name__)

class AssistanceWorkflowNodes:
    def __init__(self):
        self.email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')

    async def collect_assistance_info_node(self, state: MultiCountryLegalState, config: RunnableConfig) -> Dict[str, Any]:
        """Collect assistance information (email, description)"""
        s = state.model_dump()
        assistance_step = s.get("assistance_step", "collecting_email")
        user_input = s.get("messages", [{}])[-1].get("content", "") if s.get("messages") else ""
        
        logger.info(f"📝 Collecting assistance info - step: {assistance_step}")
        logger.debug(f"User input: {user_input}")
        
        if assistance_step == "collecting_email":
            if not user_input:
                logger.info(f"ℹ️  Waiting for email input")
                return {
                    "assistance_step": "collecting_email",
                    "messages": []  # Response node will generate the message
                }
            
            if self.email_pattern.match(user_input):
                logger.info(f"📧 Email collected: {user_input}")
                return {
                    "assistance_step": "collecting_description",
                    "user_email": user_input,
                    "assistance_requested": True,
                    "messages": []  # Response node will generate the message
                }
            else:
                logger.warning(f"Invalid email: {user_input}")
                return {
                    "assistance_step": "collecting_email",
                    "messages": [{
                        "role": "assistant",
                        "content": """⚠️ L'adresse email fournie semble invalide. Veuillez fournir une adresse email valide.

📧 **Veuillez me fournir votre adresse email :**""",
                        "meta": {"assistance_step": "collecting_email"}
                    }]
                }
        
        elif assistance_step == "collecting_description":
            if not user_input or len(user_input.strip()) < 10:
                logger.info(f"ℹ️  Waiting for description input")
                return {
                    "assistance_step": "collecting_description",
                    "messages": []  # Response node will generate the message
                }
            
            # Detect country from the description
            detected_country = MultiCountryLegalState.detect_country(user_input)
            
            logger.info(f"📝 Description collected: {user_input[:50]}...")
            logger.info(f"🌍 Detected country: {detected_country}")
            
            # Return the update - move to confirmation step
            return {
                "assistance_description": user_input,
                "assistance_step": "confirming_send",
                "country": detected_country,
                "legal_context": {
                    **state.legal_context,
                    "detected_country": detected_country
                },
                "messages": []  # Response node will generate the confirmation message
            }
        
        return {}

    async def confirm_assistance_send_node(self, state: MultiCountryLegalState, config: RunnableConfig) -> Dict[str, Any]:
        """Confirm assistance request before sending to legal team"""
        s = state.model_dump()
        user_input = s.get("messages", [{}])[-1].get("content", "").lower().strip() if s.get("messages") else ""
        
        logger.info(f"✅ Confirmation node - user input: {user_input[:30]}...")
        
        if user_input in ["oui", "yes", "ok", "confirmer"]:
            logger.info("✅ User confirmed assistance request")
            return {
                "assistance_step": "confirmed",
                "messages": []  # Let response node or approval node handle the message
            }
        elif user_input in ["non", "no", "cancel", "annuler"]:
            logger.info("❌ User cancelled assistance request")
            return {
                "assistance_step": "cancelled",
                "assistance_requested": False,
                "messages": [{
                    "role": "assistant",
                    "content": """❌ Votre demande a été annulée.

Si vous changez d'avis, vous pouvez relancer une demande en disant "Je veux parler à un avocat".""",
                    "meta": {"assistance_step": "cancelled"}
                }]
            }
        else:
            logger.info("ℹ️  Awaiting valid confirmation")
            return {
                "assistance_step": "confirming_send",
                "messages": [{
                    "role": "assistant",
                    "content": f"""⚠️ Veuillez confirmer avec "oui" ou "non".

📋 **RÉCAPITULATIF DE VOTRE DEMANDE :**

📧 **Email :** {s.get("user_email")}
📝 **Description :** {s.get("assistance_description")}

✅ **Confirmez-vous l'envoi de cette demande à notre équipe juridique ?**

Répondez par :
- **"oui"** pour confirmer et envoyer
- **"non"** pour annuler et modifier""",
                    "meta": {"assistance_step": "confirming_send"}
                }]
            }

    def route_assistance(self, state: MultiCountryLegalState) -> str:
        """Route assistance workflow based on current state"""
        s = state.model_dump()
        assistance_step = s.get("assistance_step", "collecting_email")
        
        logger.info(f"📋 Assistance step: {assistance_step}")
        logger.info(f"   - Has email: {s.get('user_email') is not None} ({s.get('user_email')})")
        logger.info(f"   - Has description: {s.get('assistance_description') is not None} ({s.get('assistance_description')})")
        
        if assistance_step == "collecting_email" and not s.get("user_email"):
            logger.info("→ Routing to: need_email (waiting for email)")
            return "need_email"
        elif assistance_step == "collecting_description" and not s.get("assistance_description"):
            logger.info("→ Routing to: need_description (waiting for description)")
            return "need_description"
        elif assistance_step == "confirming_send" and s.get("user_email") and s.get("assistance_description"):
            logger.info("→ Routing to: ready_to_confirm (awaiting user confirmation)")
            return "ready_to_confirm"
        elif assistance_step == "cancelled":
            logger.info("→ Routing to: cancelled")
            return "cancelled"
        
        logger.info("→ Routing to: need_email (default)")
        return "need_email"

    def route_after_confirmation(self, state: MultiCountryLegalState) -> str:
        """Route after confirmation step"""
        s = state.model_dump()
        assistance_step = s.get("assistance_step")
        last_message = s.get("messages", [{}])[-1] if s.get("messages") else {}
        user_input = last_message.get("content", "").lower().strip() if last_message.get("role") == "user" else ""
        
        logger.info(f"📋 Confirmation step: {assistance_step}")
        logger.info(f"   - Last user message: '{user_input}'")
        
        if assistance_step == "confirmed":
            logger.info("→ Routing to: confirmed (human approval)")
            return "confirmed"
        elif assistance_step == "cancelled":
            logger.info("→ Routing to: cancelled")
            return "cancelled"
        else:
            logger.info("→ Routing to: needs_correction (need clarification)")
            return "needs_correction"