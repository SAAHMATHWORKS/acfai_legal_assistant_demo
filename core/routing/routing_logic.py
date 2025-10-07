# [file name]: core/routing/routing_logic.py
import logging
from typing import Literal
from models.state_models import MultiCountryLegalState

logger = logging.getLogger(__name__)

class RoutingLogic:
    """Centralized routing logic for graph edges"""
    
    def route_after_info_collection(
        self, 
        state: MultiCountryLegalState
    ) -> Literal["need_email", "need_description", "ready_to_confirm", "cancelled"]:
        """Route based on current assistance step and collected data"""
        
        step = state.assistance_step
        has_email = bool(state.user_email)
        has_description = bool(state.assistance_description)
        
        logger.info(f"📋 Assistance step: {step}")
        logger.info(f"   - Has email: {has_email} ({state.user_email})")
        logger.info(f"   - Has description: {has_description} ({state.assistance_description})")
        
        # 🔥 NEW: Handle cancellation first
        if step == "cancelled":
            logger.info("🔄 Assistance workflow cancelled by user")
            return "cancelled"
        
        # Route based on current step progression
        if step == "collecting_email":
            if not has_email:
                logger.info("→ Routing to: need_email (waiting for email)")
                return "need_email"
            else:
                # Email collected, move to description
                logger.info("→ Routing to: need_description (email collected)")
                return "need_description"
        
        elif step == "collecting_description":
            if not has_description:
                logger.info("→ Routing to: need_description (waiting for description)")
                return "need_description"
            else:
                # Description collected, ready for confirmation
                logger.info("→ Routing to: ready_to_confirm (both collected)")
                return "ready_to_confirm"
        
        elif step == "confirming_send":
            # We're already in confirmation step - stay here until user confirms
            logger.info("→ Routing to: ready_to_confirm (awaiting user confirmation)")
            return "ready_to_confirm"
        
        else:
            # Default fallback logic
            if not has_email:
                logger.info("→ Routing to: need_email (default)")
                return "need_email"
            elif not has_description:
                logger.info("→ Routing to: need_description (default)")
                return "need_description"
            else:
                logger.info("→ Routing to: ready_to_confirm (default)")
                return "ready_to_confirm"
    
    def route_after_confirmation(
        self,
        state: MultiCountryLegalState
    ) -> Literal["confirmed", "cancelled", "needs_correction"]:
        """Route based on user's confirmation response and current step"""
        
        step = state.assistance_step
        last_message = self._get_last_user_message(state)
        
        logger.info(f"📋 Confirmation step: {step}")
        logger.info(f"   - Last user message: '{last_message}'")
        
        # 🔥 NEW: Handle cancellation from confirmation step
        if step == "cancelled":
            logger.info("→ Routing to: cancelled (workflow cancelled)")
            return "cancelled"
        
        elif step == "confirmed":
            logger.info("→ Routing to: confirmed (human approval)")
            return "confirmed"
        
        elif step == "confirming_send":
            # In confirmation step, check user response
            user_response = last_message.lower().strip() if last_message else ""
            
            if user_response in ["oui", "yes", "ok", "confirm", "confirmer", "c'est bon", "d'accord", "envoyer", "valider"]:
                logger.info("→ Routing to: confirmed (user confirmed)")
                return "confirmed"
            
            elif user_response in ["non", "no", "cancel", "annuler", "pas maintenant", "arrêter", "stop", "je ne veux plus"]:
                logger.info("→ Routing to: cancelled (user cancelled)")
                return "cancelled"
            
            else:
                # User provided description or unclear response - go to response to ask again
                logger.info("→ Routing to: needs_correction (need clarification)")
                return "needs_correction"
        
        else:
            logger.info("→ Routing to: needs_correction (invalid state)")
            return "needs_correction"
    
    def route_after_human_approval(
        self,
        state: MultiCountryLegalState
    ) -> Literal["approved", "rejected", "interrupt"]:
        """Route based on human approval status"""
        
        approval_status = state.approval_status
        logger.info(f"📋 Approval status: {approval_status}")
        
        if approval_status == "approved":
            logger.info("→ Routing to: approved (process assistance)")
            return "approved"
        
        elif approval_status == "rejected":
            logger.info("→ Routing to: rejected (response)")
            return "rejected"
        
        else:
            # Still waiting for approval or error state
            logger.info("→ Routing to: interrupt (waiting for decision)")
            return "interrupt"
    
    def _get_last_user_message(self, state: MultiCountryLegalState) -> str:
        """Extract the last user message from state"""
        if not state.messages:
            return ""
        
        for msg in reversed(state.messages):
            if hasattr(msg, 'role'):
                role = msg.role
            else:
                role = msg.get('role', '')
            
            if role in ['user', 'human']:
                if hasattr(msg, 'content'):
                    return msg.content
                else:
                    return msg.get('content', '')
        
        return ""
    
    def _looks_like_description(self, text: str) -> bool:
        """Check if text looks like a description rather than a confirmation"""
        description_indicators = [
            "j'ai besoin", "je veux", "je souhaite", "aide pour", "divorce", 
            "mariage", "héritage", "contrat", "travail", "familial", "bénin", "madagascar",
            "problème", "situation", "question", "demande"
        ]
        
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in description_indicators)