# core/human_approval_node.py
import logging
from typing import Literal
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt, Command
from models.state_models import MultiCountryLegalState
from core.assistance.email_service import AssistanceEmailService
from datetime import datetime

logger = logging.getLogger(__name__)

class HumanApprovalNode:
    def __init__(self):
        self.email_service = AssistanceEmailService()
    
    async def process_approval(
        self,
        state: MultiCountryLegalState,
        config: RunnableConfig
    ) -> Command[Literal["response"]]:  # Updated: Removed "process_assistance"
        """Process human approval with interrupt"""
        try:
            # Validate required fields
            if not state.user_email or not state.assistance_description:
                logger.warning("Missing required fields for approval")
                return Command(
                    goto="response",
                    update={
                        "messages": [{
                            "role": "assistant",
                            "content": "❌ Données incomplètes pour l'approbation.",
                            "meta": {}
                        }]
                    }
                )
            
            logger.info(f"🔒 Human approval node triggered for {state.user_email}")
            
            # Prepare interrupt message
            interrupt_message = self._format_approval_request(state)
            
            # Trigger interrupt and wait for human input
            moderator_input = interrupt({
                "type": "human_approval",
                "user_email": state.user_email,
                "country": self._get_country_display(state),
                "description": state.assistance_description,
                "message": interrupt_message
            })
            
            logger.info(f"📥 Received moderator input: {moderator_input}")
            
            # Parse moderator decision
            decision = self._parse_decision(moderator_input)
            
            # Handle approval
            if decision["approved"]:
                return await self._handle_approval(state, decision)
            else:
                return await self._handle_rejection(state, decision)
                
        except Exception as e:
            logger.error(f"Error in approval node: {str(e)}", exc_info=True)
            return Command(
                goto="response",
                update={
                    "approval_status": "error",
                    "messages": [{
                        "role": "assistant",
                        "content": f"❌ Erreur lors de l'approbation: {str(e)}",
                        "meta": {}
                    }]
                }
            )
    
    async def _handle_approval(
        self, 
        state: MultiCountryLegalState, 
        decision: dict
    ) -> Command[Literal["response"]]:  # Updated: Removed "process_assistance"
        """Handle approved request (sends email and routes to response)"""
        logger.info(f"✅ Request APPROVED for {state.user_email}")
        
        # Send email
        email_result = self.email_service.send_assistance_request(
            user_email=state.user_email,
            user_query=state.last_search_query or "Demande d'assistance",
            assistance_description=state.assistance_description,
            country=self._get_country_display(state)
        )
        logger.info(f"✅ Emails envoyés avec succès pour {state.user_email}")
        
        # Build success message
        if email_result.get("success"):
            message_content = f"""✅ **DEMANDE APPROUVÉE ET ENVOYÉE**

📧 Un email de confirmation a été envoyé à: {state.user_email}
👨‍⚖️ Notre équipe juridique vous contactera sous 24-48 heures.

**Raison de l'approbation:** {decision['reason']}
**Approuvé par:** {decision['moderator_id']}
"""
        else:
            message_content = f"""⚠️ **DEMANDE APPROUVÉE MAIS ERREUR D'ENVOI**

La demande a été approuvée mais l'envoi d'email a échoué.
**Erreur:** {email_result.get('error', 'Unknown')}

Veuillez contacter directement: fitahiana@acfai.org
"""
        
        return Command(
            goto="response",
            update={
                "approval_status": "approved",
                "approval_reason": decision["reason"],
                "approved_by": decision["moderator_id"],
                "approval_timestamp": datetime.now().isoformat(),
                "email_status": "sent" if email_result.get("success") else "error",
                "messages": [{
                    "role": "assistant",
                    "content": message_content,
                    "meta": {"approval": "approved"}
                }]
            }
        )
    
    async def _handle_rejection(
        self, 
        state: MultiCountryLegalState, 
        decision: dict
    ) -> Command[Literal["response"]]:  # Updated: Removed "process_assistance"
        """Handle rejected request"""
        logger.info(f"❌ Request REJECTED for {state.user_email}")
        
        message_content = f"""❌ **DEMANDE REFUSÉE**

Votre demande d'assistance n'a pas été approuvée.

**Raison:** {decision['reason']}

Si vous pensez qu'il s'agit d'une erreur, veuillez reformuler votre demande avec plus de détails.
"""
        
        return Command(
            goto="response",
            update={
                "approval_status": "rejected",
                "approval_reason": decision["reason"],
                "approved_by": decision["moderator_id"],
                "approval_timestamp": datetime.now().isoformat(),
                "messages": [{
                    "role": "assistant",
                    "content": message_content,
                    "meta": {"approval": "rejected"}
                }]
            }
        )
    
    def _format_approval_request(self, state: MultiCountryLegalState) -> str:
        """Format the approval request message"""
        return f"""
🔒 **APPROBATION HUMAINE REQUISE**

📧 **Email:** {state.user_email}
🌍 **Pays:** {self._get_country_display(state)}
📝 **Description:** {state.assistance_description}
🔍 **Requête initiale:** {state.last_search_query or 'Non spécifiée'}

**Instructions:**
- Tapez "approve [raison]" pour approuver
- Tapez "reject [raison]" pour rejeter

**Exemples:**
- "approve Demande légitime"
- "reject Email invalide"
"""
    
    def _parse_decision(self, user_input: str) -> dict:
        """Parse moderator decision from input"""
        if not user_input or not isinstance(user_input, str):
            return {
                "approved": False,
                "reason": "Input invalide",
                "moderator_id": "system"
            }
        
        input_lower = user_input.lower().strip()
        
        # Check for approval keywords
        approve_keywords = ["approve", "approuver", "oui", "yes", "ok", "accept"]
        is_approved = any(kw in input_lower for kw in approve_keywords)
        
        # Extract reason (text after the decision keyword)
        reason = user_input.strip()
        for keyword in approve_keywords + ["reject", "rejeter", "non", "no"]:
            if keyword in input_lower:
                parts = user_input.split(keyword, 1)
                if len(parts) > 1 and parts[1].strip():
                    reason = parts[1].strip()
                    break
        
        if not reason or reason == user_input:
            reason = "Approuvé par modérateur" if is_approved else "Refusé par modérateur"
        
        return {
            "approved": is_approved,
            "reason": reason,
            "moderator_id": "human_moderator"
        }
    
    def _get_country_display(self, state: MultiCountryLegalState) -> str:
        """Get country display name"""
        country = state.country or state.legal_context.get("detected_country", "unknown")
        if country == "unknown" and state.assistance_description:
            country = MultiCountryLegalState.detect_country(state.assistance_description)
        country_map = {
            "benin": "Bénin",
            "madagascar": "Madagascar"
        }
        logger.debug(f"Country from state: {state.country}, legal_context: {state.legal_context.get('detected_country')}, description: {country}")
        return country_map.get(country, "Non spécifié")