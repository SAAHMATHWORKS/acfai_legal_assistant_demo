# [file name]: core/assistance/email_service.py
"""
Wrapper for email functionality - provides a consistent interface
"""
import re
import logging
from typing import Optional, Dict
from core.email_tool import LegalAssistanceEmailer

logger = logging.getLogger(__name__)


class AssistanceEmailService:
    """Service wrapper for email operations"""
    
    def __init__(self):
        self.emailer = LegalAssistanceEmailer()
    
    def extract_email_from_text(self, text: str) -> Optional[str]:
        """Extract email from text"""
        return self.emailer.extract_email_from_text(text)
    
    def validate_email(self, email: str) -> bool:
        """Validate email format"""
        return self.emailer.validate_email(email)
    
    def send_assistance_request(
        self, 
        user_email: str, 
        user_query: str,
        assistance_description: str, 
        country: str
    ) -> Dict:
        """Send assistance request emails"""
        return self.emailer.send_assistance_request(
            user_email=user_email,
            user_query=user_query,
            assistance_description=assistance_description,
            country=country
        )