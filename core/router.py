# [file name]: core/router.py
import re
import logging
import json
from typing import Dict, List, Optional, Literal, Any 
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from config.settings import settings
from models.state_models import RoutingResult

logger = logging.getLogger(__name__)

class CountryRouter:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.CHAT_MODEL_2,
            temperature=0.1,
            max_tokens=200
        )

    async def route_query(self, query: str, conversation_history: List[Dict]) -> RoutingResult:
        """Unified LLM-powered routing"""
        try:
            # Build conversation context
            context = self._build_conversation_context(conversation_history)
            
            # LLM routing prompt
            routing_prompt = self._build_routing_prompt(query, context)
            
            logger.info(f"🔀 Routing query: '{query[:50]}...'")
            
            # Call LLM for routing decision
            response = await self.llm.ainvoke([SystemMessage(content=routing_prompt)])
            routing_result = self._parse_routing_response(response.content)
            
            logger.info(f"🎯 Router decision: {routing_result.country} ({routing_result.confidence})")
            
            return routing_result
            
        except Exception as e:
            logger.error(f"Router error: {e}")
            # Fallback to unclear
            return RoutingResult(
                country="unclear",
                confidence="low",
                method="error_fallback",
                explanation=f"Router error: {str(e)}"
            )

    def _build_routing_prompt(self, query: str, context: str) -> str:
        """Build comprehensive routing prompt"""
        return f"""
Vous êtes un routeur intelligent pour un assistant juridique spécialisé dans le droit du Bénin et de Madagascar.

**TÂCHE:** Analyser la requête utilisateur et déterminer la meilleure destination.

**DESTINATIONS POSSIBLES:**
- "benin": Questions juridiques concernant le Bénin (lois, procédures, droits)
- "madagascar": Questions juridiques concernant Madagascar (lois, procédures, droits)  
- "assistance_request": Demande pour parler à un avocat humain
- "greeting_small_talk": Salutations, présentations, remerciements (politesse uniquement)
- "conversation_repair": Incompréhension, demande de clarification
- "conversation_summarization": Demande de résumé de la conversation
- "out_of_scope": Questions NON juridiques (café, météo, sports, recettes, etc.)
- "unclear": Intention juridique incertaine

**REQUÊTE:** "{query}"

**CONTEXTE DE CONVERSATION:**
{context}

**RÈGLES DE CLASSIFICATION:**

1. **greeting_small_talk** - UNIQUEMENT pour politesse basique:
   - Salutations: "bonjour", "salut", "hello", "bonsoir", "au revoir"
   - Présentations brèves: "je m'appelle X", "mon nom est X"
   - Remerciements: "merci", "merci beaucoup"
   - Politesses simples: "comment ça va", "ça va bien"
   - Questions sur l'identité de l'assistant: "qui es-tu", "comment tu t'appelles"

2. **benin** - Pour questions juridiques sur le Bénin:
   - Mentions explicites: "bénin", "benin", "béninois"
   - Villes: "cotonou", "porto-novo"
   - Lois/procédures béninoises

3. **madagascar** - Pour questions juridiques sur Madagascar:
   - Mentions explicites: "madagascar", "malgache"
   - Villes: "antananarivo", "toamasina"
   - Lois/procédures malgaches

4. **assistance_request** - Demande d'aide humaine:
   - "parler à un avocat"
   - "contacter un avocat"
   - "assistance téléphonique"
   - "besoin d'aide juridique personnalisée"

5. **conversation_repair** - Problèmes de compréhension:
   - "je n'ai pas compris"
   - "répète s'il te plaît"
   - "explique autrement"
   - "qu'est-ce que tu veux dire"

6. **conversation_summarization** - Demande de résumé:
   - "résume notre conversation"
   - "récapitulatif"
   - "qu'avons-nous dit"

7. **out_of_scope** - Questions clairement NON juridiques:
   - Météo/Climat: "température à Douala", "il va pleuvoir?"
   - Nourriture: "recette de ndolé", "fais-moi un café"
   - Sport: "résultat du match", "qui a gagné?"
   - Technologie: "comment réparer mon téléphone", "meilleur ordinateur"
   - Divertissement: "raconte une blague", "parle-moi de musique"
   - Santé non-juridique: "symptômes grippe", "remèdes traditionnels"
   - **Règle clé**: AUCUN aspect juridique ou lien avec le droit

8. **unclear** - Questions juridiques MAIS pays/détails manquants:
   - "J'ai un problème de divorce" (quel pays?)
   - "Comment créer une entreprise" (Bénin ou Madagascar?)
   - "Besoin d'aide juridique" (trop vague)
   - "Question sur l'héritage" (juridiction non précisée)
   - **Règle clé**: Intention juridique évidente MAIS manque de précision sur le pays ou les détails

**EXEMPLES COMPLETS:**
- "Bonjour" → {{"destination": "greeting_small_talk", "confidence": "high", "reasoning": "Salutation simple"}}
- "je m'appelle Thibaut" → {{"destination": "greeting_small_talk", "confidence": "high", "reasoning": "Présentation personnelle"}}
- "comment est-ce que je m'appelle" → {{"destination": "greeting_small_talk", "confidence": "high", "reasoning": "Question personnelle de rappel"}}
- "salut comment ça va" → {{"destination": "greeting_small_talk", "confidence": "high", "reasoning": "Salutation et politesse"}}
- "merci beaucoup" → {{"destination": "greeting_small_talk", "confidence": "high", "reasoning": "Remerciement"}}
- "qui es-tu" → {{"destination": "greeting_small_talk", "confidence": "high", "reasoning": "Question sur l'identité de l'assistant"}}
- "procedure divorce Bénin" → {{"destination": "benin", "confidence": "high", "reasoning": "Question juridique explicite sur le Bénin"}}
- "loi foncière Madagascar" → {{"destination": "madagascar", "confidence": "high", "reasoning": "Question juridique sur Madagascar"}}
- "Je veux parler à un avocat" → {{"destination": "assistance_request", "confidence": "high", "reasoning": "Demande explicite d'assistance humaine"}}
- "Je n'ai pas compris" → {{"destination": "conversation_repair", "confidence": "high", "reasoning": "Demande de clarification"}}
- "résume notre conversation" → {{"destination": "conversation_summarization", "confidence": "high", "reasoning": "Demande de résumé"}}
- "fais-moi un café" → {{"destination": "out_of_scope", "confidence": "high", "reasoning": "Demande sans rapport avec le droit"}}
- "quelle est la météo" → {{"destination": "out_of_scope", "confidence": "high", "reasoning": "Question météorologique, non juridique"}}
- "température à Douala" → {{"destination": "out_of_scope", "confidence": "high", "reasoning": "Question climatique, hors domaine juridique"}}
- "raconte-moi une blague" → {{"destination": "out_of_scope", "confidence": "high", "reasoning": "Demande de divertissement, non juridique"}}
- "J'ai un problème de divorce" → {{"destination": "unclear", "confidence": "medium", "reasoning": "Question juridique mais pays non précisé"}}
- "Comment créer une entreprise" → {{"destination": "unclear", "confidence": "medium", "reasoning": "Question juridique mais juridiction manquante"}}

**IMPORTANT:** 
- **out_of_scope**: Questions SANS aucun aspect juridique (météo, sport, nourriture, etc.)
- **unclear**: Questions AVEC intention juridique MAIS manque de précision sur le pays
- Les présentations, salutations et remerciements sont "greeting_small_talk"
- Seules les questions JURIDIQUES avec pays identifié vont vers "benin" ou "madagascar"

**FORMAT DE RÉPONSE:**
Répondez UNIQUEMENT au format JSON valide:
{{
    "destination": "benin|madagascar|assistance_request|greeting_small_talk|conversation_repair|conversation_summarization|unclear",
    "confidence": "high|medium|low",
    "reasoning": "explication brève et claire"
}}

**RÉPONSE:**
"""

    def _parse_routing_response(self, response_text: str) -> RoutingResult:
        """Parse LLM routing response"""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON found in response")
            
            result = json.loads(json_match.group())
            
            # Validate required fields
            destination = result.get("destination", "unclear")
            confidence = result.get("confidence", "low")
            reasoning = result.get("reasoning", "No reasoning provided")
            
            # Map destination to RoutingResult country field
            valid_destinations = [
                "benin", "madagascar", "unclear", "greeting_small_talk", 
                "conversation_repair", "assistance_request", "conversation_summarization",
                "out_of_scope"
            ]
            
            if destination not in valid_destinations:
                logger.warning(f"Invalid destination from LLM: {destination}, defaulting to unclear")
                destination = "unclear"
                confidence = "low"
                reasoning = f"Destination invalide: {destination}"
            
            return RoutingResult(
                country=destination,
                confidence=confidence,
                method="llm_routing",
                explanation=reasoning
            )
            
        except Exception as e:
            logger.error(f"Error parsing routing response: {e}")
            logger.error(f"Raw response: {response_text}")
            
            return RoutingResult(
                country="unclear",
                confidence="low",
                method="parse_error",
                explanation=f"Parse error: {str(e)}"
            )

    def _build_conversation_context(self, conversation_history: List[Dict]) -> str:
        """Build conversation context"""
        if not conversation_history:
            return "Aucun contexte de conversation"
        
        # Get last 6 messages for context
        recent_messages = conversation_history[-6:]
        context_lines = []
        
        for msg in recent_messages:
            role = "Utilisateur" if msg.get("role") in ["user", "human"] else "Assistant"
            content = msg.get("content", "")
            context_lines.append(f"{role}: {content}")
        
        return "\n".join(context_lines)

    async def health_check(self) -> Dict[str, Any]:
        """Router health check"""
        try:
            # Test with a simple query
            test_result = await self.route_query("test", [])
            return {
                "status": "healthy",
                "llm_responding": True,
                "last_test_result": test_result.model_dump()
            }
        except Exception as e:
            return {
                "status": "unhealthy", 
                "llm_responding": False,
                "error": str(e)
            }