import re
import logging
import asyncio
from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document
from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch

from config.settings import settings
from config.constants import ARTICLE_PATTERNS, CATEGORY_KEYWORDS, DOCUMENT_TYPE_KEYWORDS

logger = logging.getLogger(__name__)

class LegalRetriever:
    def __init__(self, vectorstore: MongoDBAtlasVectorSearch, collection):
        self.vectorstore = vectorstore
        self.collection = collection

    async def smart_legal_query(self, user_query: str, country: str) -> Tuple[List[Document], List[str], Dict[str, Any], str]:
        """Perform smart legal search with automatic fallback and custom messages - ASYNC VERSION"""
        try:
            # Détection initiale du type de document
            initial_doc_type = self._detect_document_type(user_query.lower())
            pre_filter = self._build_pre_filters(user_query, country)
            
            logger.info(f"📋 Filtre doc_type initial: {initial_doc_type}")
            logger.info(f"🔍 Recherche {country} avec filtres: {pre_filter}")
            
            # Première recherche
            enhanced_docs, detected_articles, applied_filters = await self._perform_search_async(
                user_query, country, pre_filter
            )
            
            message_supplementaire = ""
            
            # Fallback automatique si aucun résultat pour case_study
            if not enhanced_docs and initial_doc_type == "case_study":
                logger.info("🔄 Fallback: Aucun case_study trouvé, recherche dans les articles")
                pre_filter["doc_type"] = "articles"
                pre_filter["fallback_used"] = True
                
                enhanced_docs, detected_articles, applied_filters = await self._perform_search_async(
                    user_query, country, pre_filter
                )
                applied_filters["original_search"] = "case_study"
                applied_filters["fallback_to"] = "articles"
                
                # Message personnalisé pour le fallback
                message_supplementaire = "La base sera enrichie avec des décisions de justice prochainement."
            
            logger.info(f"🔍 Search completed: {len(enhanced_docs)} documents found")
            return enhanced_docs, detected_articles, applied_filters, message_supplementaire
            
        except Exception as e:
            logger.error(f"Error in smart_legal_query: {str(e)}")
            # Return empty results on error
            return [], [], {"error": str(e)}, f"Erreur lors de la recherche: {str(e)}"


    async def _perform_search_async(self, user_query: str, country: str, pre_filter: Dict) -> Tuple[List[Document], List[str], Dict[str, Any]]:
        """Perform search with given filters - ASYNC VERSION"""
        try:
            detected_articles = self._detect_articles(user_query)
            enhanced_query = self._enhance_query(user_query, detected_articles)
            
            logger.info(f"🔢 Articles détectés: {detected_articles}")
            logger.info(f"🔍 Requête enrichie: {enhanced_query[:100]}...")
            
            # CRITICAL FIX: Run synchronous vectorstore operation in thread pool
            import asyncio
            docs = await asyncio.get_event_loop().run_in_executor(
                None,  # Use default thread pool
                lambda: self.vectorstore.similarity_search(
                    enhanced_query, 
                    k=settings.MAX_SEARCH_RESULTS, 
                    pre_filter=pre_filter
                )
            )
            
            logger.info(f"🎯 Vector search returned {len(docs)} raw documents")
            
            if docs:
                logger.info(f"📄 First result metadata: {docs[0].metadata}")
            else:
                logger.warning(f"⚠️ No documents found with filters: {pre_filter}")
                await self._debug_search_issue(pre_filter)
            
            enhanced_docs = self.enhance_with_article_context(docs)
            return enhanced_docs, detected_articles, pre_filter
            
        except Exception as e:
            logger.error(f"Error in _perform_search_async: {str(e)}")
            return [], [], {"error": str(e)}

    async def _debug_search_issue(self, pre_filter: Dict):
        """Debug why search returned no results"""
        try:
            # Check total document count
            total_count = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.collection.count_documents({})
            )
            logger.info(f"🔢 Total documents in collection: {total_count}")
            
            # Check documents matching country filter
            country_count = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.collection.count_documents({"pays": pre_filter.get("pays")})
            )
            logger.info(f"🌍 Documents for country {pre_filter.get('pays')}: {country_count}")
            
            # Check documents with embeddings
            embedding_count = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.collection.count_documents({
                    "pays": pre_filter.get("pays"),
                    "embedding": {"$exists": True, "$ne": None}
                })
            )
            logger.info(f"🎯 Documents with embeddings: {embedding_count}")
            
            # Sample document check
            sample_doc = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.collection.find_one({"pays": pre_filter.get("pays")})
            )
            
            if sample_doc:
                logger.info(f"📄 Sample document keys: {list(sample_doc.keys())}")
                logger.info(f"📄 Sample doc_type: {sample_doc.get('doc_type', 'NOT_SET')}")
            else:
                logger.error("❌ No sample document found!")
                
        except Exception as e:
            logger.error(f"Error in debug: {str(e)}")

    def _build_pre_filters(self, query: str, country: str) -> Dict[str, Any]:
        """Build search filters based on query and country"""
        # Filtre pays obligatoire - MAKE SURE EXACT MATCH
        country_mapping = {
            "benin": "Bénin",
            "madagascar": "Madagascar"
        }
        
        pre_filter = {"pays": country_mapping.get(country.lower(), country)}
        
        # Filtre doc_type pour différencier articles et études de cas
        query_lower = query.lower()
        detected_doc_type = self._detect_document_type(query_lower)
        pre_filter["doc_type"] = detected_doc_type
        
        logger.info(f"🏷️ Using country filter: {pre_filter['pays']}")
        logger.info(f"📋 Using doc_type filter: {detected_doc_type}")
        
        # Filtres par catégorie (optionnels)
        logger.info("ℹ️  No category filter applied - using all available family law documents")
        # for keyword, category in CATEGORY_KEYWORDS.items():
        #     if keyword in query_lower:
        #         pre_filter["categorie"] = category
        #         logger.info(f"🏷️ Filtre catégorie: {category}")
        #         break
                
        return pre_filter

    def _detect_document_type(self, query_lower: str) -> str:
        """Détecte le type de document basé sur les mots-clés de la requête"""
        # Mots-clés pour les études de cas
        case_study_indicators = [
            "jurisprudence", "arrêt", "décision", "tribunal", "cours", "jugement",
            "affaire", "procès", "litige", "contentieux", "précédent", "cas",
            "cour d'appel", "cour suprême", "conseil d'état", "juridiction"
        ]
        
        # Mots-clés pour les articles
        articles_indicators = [
            "article", "loi", "code", "décret", "texte", "disposition",
            "règlement", "ordonnance", "prescription", "norme", "chapitre", "titre"
        ]
        
        case_study_score = sum(1 for keyword in case_study_indicators if keyword in query_lower)
        articles_score = sum(1 for keyword in articles_indicators if keyword in query_lower)
        
        if case_study_score > articles_score and case_study_score > 0:
            return "case_study"
        elif articles_score > 0:
            return "articles"
        else:
            # Par défaut, on cherche les articles de loi
            return "articles"

    def _detect_articles(self, query: str) -> List[str]:
        """Detect article references in query"""
        detected_articles = []
        for pattern in ARTICLE_PATTERNS:
            matches = re.findall(pattern, query.lower())
            for match in matches:
                if isinstance(match, tuple):
                    nums = [n for n in match if n.isdigit()]
                    detected_articles.extend(nums)
                else:
                    nums = re.findall(r"\d+", match)
                    detected_articles.extend(nums)
        
        return sorted(list(set(detected_articles)))

    def _enhance_query(self, query: str, detected_articles: List[str]) -> str:
        """Enhance query with article context"""
        if detected_articles:
            enhanced = f"article {' '.join(detected_articles)} {query}"
            logger.info(f"🔢 Requête enrichie avec articles: {detected_articles}")
            return enhanced
        return query

    def enhance_with_article_context(self, results: List[Document]) -> List[Document]:
        """Enhance search results with referenced article context"""
        enhanced_results = []
        for result in results:
            enhanced_results.append(result)
            
            # Pour les documents de type "articles", on peut ajouter les références
            if result.metadata.get("doc_type") == "articles":
                article_refs = result.metadata.get("article_references", [])
                resolved_refs = result.metadata.get("resolved_references", {})
                
                for article_num in article_refs[:3]:
                    if article_num in resolved_refs:
                        ref_doc = Document(
                            page_content=f"Article {article_num} (Référencé): {resolved_refs[article_num][:500]}...",
                            metadata={
                                **result.metadata,
                                "is_reference": True,
                                "referenced_article": article_num,
                                "doc_type": "article_reference"
                            },
                        )
                        enhanced_results.append(ref_doc)
        
        return enhanced_results

    def format_search_results(self, query: str, enhanced_docs: List[Document], 
                            detected_articles: List[str], applied_filters: Dict[str, Any], 
                            country: str, supplemental_message: str = "") -> str:
        """Format search results for system prompt"""
        country_name = "Bénin" if country == "benin" else "Madagascar"
        
        if not enhanced_docs:
            doc_type = applied_filters.get("doc_type", "articles")
            
            # Check if this was an error case
            if "error" in applied_filters:
                return f"""
**🚨 ERREUR DE RECHERCHE - {country_name.upper()}**

Une erreur s'est produite lors de la recherche: {applied_filters['error']}

**Informations de débogage:**
- **Requête**: "{query}"
- **Pays**: {country_name}
- **Type de document recherché**: {doc_type}
- **Filtres**: {applied_filters}

Veuillez réessayer ou contacter le support technique.
"""
            
            if applied_filters.get("fallback_used"):
                # Cas où le fallback a été utilisé mais n'a rien trouvé non plus
                return f"""
**🔍 RECHERCHE JURIDIQUE - {country_name.upper()}**

Aucun document trouvé pour votre requête concernant la jurisprudence.

**💡 Informations :**
- Votre recherche portait sur des décisions de justice
- La base de données sera enrichie avec des décisions de justice prochainement
- En attendant, vous pouvez consulter les articles de loi pour des informations générales

**Filtres appliqués**: {applied_filters}
"""
            else:
                # Cas normal sans fallback
                return f"""
**🔍 RECHERCHE JURIDIQUE - {country_name.upper()}**

Aucun document trouvé avec les critères suivants:
- **Type de document**: {doc_type}
- **Catégorie**: {applied_filters.get('categorie', 'Toutes')}
- **Requête**: "{query}"

**Suggestion**: Essayez avec des termes plus généraux ou vérifiez l'orthographe.

**Filtres appliqués**: {applied_filters}
"""

        # Si des documents sont trouvés
        doc_type = applied_filters.get("doc_type", "articles")
        doc_type_fr = "articles de loi" if doc_type == "articles" else "études de cas/jurisprudence"
        
        fallback_note = ""
        if applied_filters.get("fallback_used"):
            fallback_note = f"""
**💡 Note importante :**
Votre requête concernait initialement des **décisions de justice**. 
Comme la base ne contient pas encore de jurisprudence, voici des informations issues des **textes de loi**.
{supplemental_message}
"""
        
        search_results = f"""
**🔍 RECHERCHE JURIDIQUE - {country_name.upper()}**
**Type de documents**: {doc_type_fr}
**Requête**: "{query}"
**Juridiction**: {country_name}
**Articles détectés**: {', '.join(detected_articles) if detected_articles else 'Aucun'}
**Filtres appliqués**: {applied_filters}
**Documents trouvés**: {len(enhanced_docs)}
{fallback_note}
"""

        # Formatage des documents trouvés
        main_docs = [doc for doc in enhanced_docs if not doc.metadata.get("is_reference", False)]
        
        for i, doc in enumerate(main_docs[:5]):
            doc_type = doc.metadata.get("doc_type", "inconnu")
            source = doc.metadata.get('source', 'Non spécifié')
            content = doc.page_content[:600]
            
            search_results += f"""
**📄 DOCUMENT {i+1}** (Type: {doc_type})
- **Source**: {source}
- **Contenu**: {content}...
"""
        
        return search_results

    # BACKWARD COMPATIBILITY: Keep sync version for any remaining sync calls
    def smart_legal_query_sync(self, user_query: str, country: str) -> Tuple[List[Document], List[str], Dict[str, Any], str]:
        """Synchronous version for backward compatibility"""
        logger.warning("Using sync version of smart_legal_query - consider migrating to async")
        return asyncio.run(self.smart_legal_query(user_query, country))