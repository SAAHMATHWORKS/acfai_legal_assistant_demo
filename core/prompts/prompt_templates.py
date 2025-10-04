# [file name]: core/prompts/prompt_templates.py
class PromptTemplates:
    """All prompt templates used in the graph"""
    
    @staticmethod
    def get_email_request_message() -> str:
        return """📧 **Demande d'assistance juridique**

Pour vous mettre en relation avec un avocat, j'ai besoin de votre adresse email.

**Votre email :**"""
    
    @staticmethod
    def get_description_prompt(email: str) -> str:
        return f"""📝 **Description de votre besoin**

Merci ! Email enregistré : {email}

Maintenant, décrivez-moi **comment vous souhaitez être assisté(e)** :

Exemples :
• "Consultation téléphonique de 30 minutes sur le droit de la famille"
• "Avis écrit sur un contrat de travail"  
• "Accompagnement pour une procédure de divorce"
• "Explication sur mes droits successoraux"

**Votre description :**"""
    
    @staticmethod
    def get_confirmation_prompt(data: dict) -> str:
        email = data.get("email", "Non fourni")
        description = data.get("description", "Non fournie")
        
        return f"""✅ **Confirmation d'envoi**

Veuillez confirmer l'envoi de votre demande d'assistance :

📧 **Email** : {email}
📋 **Description** : {description}

**L'avocat vous contactera directement dans les 24-48 heures.**

🔔 **Confirmez-vous l'envoi ?** (répondez par OUI/NON)"""
    
    @staticmethod
    def get_missing_info_prompt(current_step: str, has_email: bool) -> str:
        if current_step == "collecting_email":
            return "📧 **Email manquant** : Pourriez-vous me donner votre adresse email ?"
        else:
            return "📝 **Description manquante** : Pourriez-vous décrire comment vous souhaitez être assisté(e) ?"
    
    @staticmethod
    def get_non_legal_response() -> str:
        return """🔍 **Hors de mon domaine d'expertise**

Je suis un assistant juridique spécialisé pour le Bénin et Madagascar. 

**Je peux vous aider avec :**
⚖️ **Questions juridiques** : lois, droits, procédures
📚 **Textes de loi** : articles, codes, décrets  
🔧 **Assistance légale** : démarches, formalités
👨‍⚖️ **Connexion avocat** : assistance humaine

**Exemples de questions que je peux traiter :**
• "Procédure de divorce au Bénin"
• "Droits des enfants à Madagascar" 
• "Articles sur le droit du travail"
• "Comment contacter un avocat ?"

Posez-moi une question juridique !"""
    
    @staticmethod
    def get_clarification_message() -> str:
        return """Je ne peux pas déterminer de quel pays vous parlez. Pourriez-vous préciser si votre question concerne le droit du **Bénin** ou de **Madagascar** ?"""
    
    @staticmethod
    def generate_greeting_response(query: str) -> str:
        """Generate appropriate greeting responses"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["bonjour", "hello", "hi"]):
            return "👋 Bonjour ! Je suis votre assistant juridique spécialisé pour le Bénin et Madagascar. Comment puis-je vous aider aujourd'hui ?"
        elif any(word in query_lower for word in ["salut", "coucou"]):
            return "👋 Salut ! Je suis votre assistant juridique. Posez-moi vos questions sur le droit béninois ou malgache !"
        elif any(word in query_lower for word in ["comment ça va", "ça va", "comment vas-tu"]):
            return "😊 Je vais très bien, merci ! Je suis prêt à vous aider avec vos questions juridiques sur le Bénin ou Madagascar."
        elif any(word in query_lower for word in ["merci", "thanks"]):
            return "🤝 Je vous en prie ! N'hésitez pas si vous avez d'autres questions juridiques."
        elif any(word in query_lower for word in ["au revoir", "bye", "à bientôt"]):
            return "👋 Au revoir ! N'hésitez pas à revenir si vous avez besoin d'assistance juridique."
        elif any(word in query_lower for word in ["qui es-tu", "ton nom", "te présenter"]):
            return "⚖️ Je suis un assistant juridique IA spécialisé dans les droits du Bénin et de Madagascar. Je peux vous aider à trouver des informations sur les lois, articles, et procédures juridiques."
        else:
            return "👋 Bonjour ! Je suis votre assistant juridique. Posez-moi vos questions sur le droit béninois ou malgache !"