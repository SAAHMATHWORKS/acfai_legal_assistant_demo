#!/usr/bin/env python3
"""
Scalable Multi-Country Legal RAG System
Supports dynamic addition of new countries with clean architecture
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

from config.settings import settings
from database.mongodb_client import MongoDBClient
from database.postgres_checkpointer import PostgresCheckpointer
from core.router import CountryRouter
from core.retriever import LegalRetriever
from core.graph_builder import GraphBuilder
from core.chat_manager import LegalChatManager
from utils.logger import setup_logging

import uuid


class MultiCountryLegalRAGSystem:
    """Scalable system class supporting dynamic country addition"""
    
    def __init__(self):
        self.mongo_client = MongoDBClient()
        self.postgres_checkpointer = PostgresCheckpointer(
            database_url=settings.DATABASE_URL,
            max_connections=10,
            min_connections=2
        )
        self.router = None
        # Dynamic country retrievers dictionary - easily extensible!
        self.country_retrievers = {}
        self.llm = None
        self.graph = None
        self.chat_manager = None
        self.initialized = False

    async def initialize(self) -> bool:
        """Initialize the complete scalable system"""
        try:
            setup_logging()
            settings.validate()
            
            # Initialize databases
            if not self.mongo_client.connect():
                raise Exception("MongoDB connection failed")
                
            if not await self.postgres_checkpointer.initialize():
                logging.warning("PostgreSQL initialization failed")
            
            # Initialize core components
            self.router = CountryRouter()
            
            # Initialize default countries - easily extensible!
            self._initialize_default_countries()
            
            # Initialize LLM
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(
                model=settings.CHAT_MODEL,
                temperature=settings.CHAT_TEMPERATURE,
                max_tokens=settings.CHAT_MAX_TOKENS
            )
            
            # Build scalable graph with country dictionary
            graph_builder = GraphBuilder(
                router=self.router,
                llm=self.llm,
                checkpointer=self.postgres_checkpointer.get_checkpointer(),
                country_retrievers=self.country_retrievers  # Pass the dictionary
            )
            
            workflow = graph_builder.build_graph()
            
            # Compile with interrupt support
            self.graph = workflow.compile(
                checkpointer=self.postgres_checkpointer.get_checkpointer(),
                interrupt_before=["human_approval"]
            )
            
            # Initialize chat manager
            self.chat_manager = LegalChatManager(
                self.graph, 
                self.postgres_checkpointer.get_checkpointer()
            )
            
            await self._perform_health_check()
            
            self.initialized = True
            logging.info(f"✅ System initialized with {len(self.country_retrievers)} countries")
            self._print_system_info()
            
            return True
            
        except Exception as e:
            logging.error(f"❌ System initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _initialize_default_countries(self):
        """Initialize default countries - easily extensible!"""
        # Benin
        if hasattr(self.mongo_client, 'benin_vectorstore'):
            self.country_retrievers["benin"] = LegalRetriever(
                self.mongo_client.benin_vectorstore,
                self.mongo_client.benin_collection
            )
        
        # Madagascar  
        if hasattr(self.mongo_client, 'madagascar_vectorstore'):
            self.country_retrievers["madagascar"] = LegalRetriever(
                self.mongo_client.madagascar_vectorstore,
                self.mongo_client.madagascar_collection
            )
        
        logging.info(f"🌍 Initialized {len(self.country_retrievers)} default countries")

    def add_country(self, country_code: str, vectorstore, collection) -> bool:
        """Dynamically add a new country to the running system"""
        try:
            if country_code in self.country_retrievers:
                logging.warning(f"Country {country_code} already exists")
                return False
            
            new_retriever = LegalRetriever(vectorstore, collection)
            self.country_retrievers[country_code] = new_retriever
            
            # Rebuild graph if system is already initialized
            if self.initialized:
                graph_builder = GraphBuilder(
                    router=self.router,
                    llm=self.llm,
                    checkpointer=self.postgres_checkpointer.get_checkpointer(),
                    country_retrievers=self.country_retrievers
                )
                workflow = graph_builder.build_graph()
                self.graph = workflow.compile(
                    checkpointer=self.postgres_checkpointer.get_checkpointer(),
                    interrupt_before=["human_approval"]
                )
            
            logging.info(f"🎉 Successfully added country: {country_code}")
            return True
            
        except Exception as e:
            logging.error(f"❌ Failed to add country {country_code}: {e}")
            return False

    async def _perform_health_check(self):
        """Perform health check after initialization"""
        try:
            health_status = await self.health_check()
            
            unhealthy_components = [k for k, v in health_status.get('components', {}).items() if not v]
            if unhealthy_components:
                logging.warning(f"⚠️ Unhealthy components: {unhealthy_components}")
                
        except Exception as e:
            logging.warning(f"⚠️ Health check failed: {e}")

    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive system health check"""
        health_status = {
            "system_initialized": self.initialized,
            "mongodb_connected": self.mongo_client.client is not None,
            "postgres_healthy": {},
            "interrupt_enabled": True,
            "available_countries": list(self.country_retrievers.keys()),
            "components": {
                "router": self.router is not None,
                "llm": self.llm is not None,
                "graph": self.graph is not None,
                "chat_manager": self.chat_manager is not None,
                "country_retrievers": len(self.country_retrievers) > 0
            },
            "timestamp": datetime.now().isoformat(),
            "settings": {
                "chat_model": settings.CHAT_MODEL,
                "embedding_model": settings.EMBEDDING_MODEL,
                "max_search_results": settings.MAX_SEARCH_RESULTS
            }
        }
        
        # Test MongoDB connection
        if health_status["mongodb_connected"]:
            try:
                self.mongo_client.client.admin.command('ping')
                health_status["mongodb_ping"] = True
            except Exception as e:
                health_status["mongodb_ping"] = False
                health_status["mongodb_error"] = str(e)
        
        # Test PostgreSQL connection
        if hasattr(self.postgres_checkpointer, 'health_check'):
            postgres_health = await self.postgres_checkpointer.health_check()
            health_status["postgres_healthy"] = postgres_health
        
        return health_status

    async def chat(self, message: str, session_id: str = None, context: dict = None) -> str:
        """Public chat interface"""
        if not self.initialized:
            raise RuntimeError("System not initialized. Call initialize() first.")
        
        if not message or not message.strip():
            raise ValueError("Message cannot be empty")
        
        try:
            # Prepare context
            ctx = context or {}
            ctx.setdefault("jurisdiction", "Unknown")
            ctx.setdefault("user_type", "general")
            ctx.setdefault("document_type", "legal")
            ctx.setdefault("detected_country", "unknown")
            
            session_id = session_id or f"cli_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            return await self.chat_manager.chat(message, session_id, ctx)
        
        except Exception as e:
            logging.error(f"❌ Chat error for session {session_id}: {e}")
            return f"❌ Désolé, une erreur s'est produite lors du traitement de votre demande. Veuillez réessayer."

    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """Get information about a specific session"""
        if not self.initialized:
            raise RuntimeError("System not initialized")
        return self.chat_manager.get_session_stats(session_id)

    def get_global_stats(self) -> Dict[str, Any]:
        """Get global system statistics"""
        if not self.initialized:
            raise RuntimeError("System not initialized")
        return self.chat_manager.get_global_stats()

    def get_available_countries(self) -> List[str]:
        """Get list of available countries"""
        return list(self.country_retrievers.keys())

    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.mongo_client:
                self.mongo_client.close()
            if self.postgres_checkpointer:
                await self.postgres_checkpointer.close()
            logging.info("✅ System cleanup completed")
        except Exception as e:
            logging.error(f"❌ Error during cleanup: {e}")

    def _print_system_info(self):
        """Print system configuration information"""
        countries = list(self.country_retrievers.keys())
        print("\n" + "="*60)
        print("🚀 SCALABLE MULTI-COUNTRY LEGAL RAG SYSTEM")
        print("="*60)
        print(f"🌍 Available Countries: {', '.join(countries) if countries else 'None'}")
        print(f"🤖 AI Model: {settings.CHAT_MODEL}")
        print(f"💾 Database: MongoDB + PostgreSQL")
        print(f"🔍 Vector Search: {settings.EMBEDDING_MODEL}")
        print(f"⏸️  Interrupt Support: ENABLED")
        print(f"🌡️  Temperature: {settings.CHAT_TEMPERATURE}")
        print(f"📝 Max Tokens: {settings.CHAT_MAX_TOKENS}")
        print("="*60)


class InterruptTester:
    """Specialized tester for human approval interrupts"""
    
    def __init__(self, system: MultiCountryLegalRAGSystem):
        self.system = system
        self.test_results = []
    
    async def test_assistance_workflow(self, test_name: str, 
                                      user_query: str, 
                                      user_email: str,
                                      user_description: str,
                                      moderator_response: str) -> Dict[str, Any]:
        """Test the complete assistance workflow with interrupt"""
        print(f"\n🧪 Interrupt Test: {test_name}")
        print(f"📝 User Query: {user_query}")
        
        # session_id = f"test_{datetime.now().strftime('%H%M%S%f')}"
        session_id = f"interactive_{uuid.uuid4().hex[:8]}"
        current_response = ""
        
        try:
            # Step 1: Initial request
            print("1️⃣  Step 1: Initial assistance request...")
            current_response = await self.system.chat(user_query, session_id)
            print(f"🤖 Response: {current_response[:150]}...")
            
            # Step 2: Email collection
            if user_email and any(keyword in current_response.lower() for keyword in ["email", "adresse", "@"]):
                print(f"2️⃣  Step 2: Providing email: {user_email}")
                current_response = await self.system.chat(user_email, session_id)
                print(f"🤖 Response: {current_response[:150]}...")
            
            # Step 3: Description collection  
            if user_description and any(keyword in current_response.lower() for keyword in ["description", "décrire", "besoin"]):
                print(f"3️⃣  Step 3: Providing description: {user_description[:50]}...")
                current_response = await self.system.chat(user_description, session_id)
                print(f"🤖 Response: {current_response[:150]}...")
            
            # Step 4: Confirmation
            if any(keyword in current_response.lower() for keyword in ["confirmer", "confirmation", "oui/non"]):
                print("4️⃣  Step 4: Confirming request...")
                current_response = await self.system.chat("oui", session_id)
                print(f"🤖 Response: {current_response[:150]}...")
            
            # Step 5: Check for interrupt
            interrupt_detected = self._check_for_interrupt(current_response, session_id)
            
            if interrupt_detected:
                print("⏸️  INTERRUPT DETECTED! Waiting for moderator...")
                
                # Step 6: Moderator decision
                print(f"👨‍⚖️  Moderator: {moderator_response}")
                final_response = await self.system.chat(moderator_response, session_id)
                print(f"✅ Final Response: {final_response[:200]}...")
                
                result = {
                    "test_name": test_name,
                    "status": "PASS",
                    "interrupt_detected": True,
                    "moderator_decision": moderator_response,
                    "final_response": final_response,
                    "session_id": session_id
                }
            else:
                print("⚠️  No interrupt detected in workflow")
                result = {
                    "test_name": test_name,
                    "status": "FAIL",
                    "interrupt_detected": False,
                    "moderator_decision": None,
                    "final_response": current_response,
                    "error": "Interrupt not triggered",
                    "session_id": session_id
                }
            
            self.test_results.append(result)
            return result
            
        except Exception as e:
            logging.error(f"❌ Test error: {e}")
            error_result = {
                "test_name": test_name,
                "status": "ERROR",
                "interrupt_detected": False,
                "moderator_decision": None,
                "final_response": current_response,
                "error": str(e),
                "session_id": session_id
            }
            self.test_results.append(error_result)
            return error_result
    
    def _check_for_interrupt(self, response: str, session_id: str) -> bool:
        """Enhanced interrupt detection"""
        interrupt_indicators = [
            "APPROBATION", "APPROVAL", "HUMAN", "MODERATOR", 
            "DÉCISION", "DECISION", "APPROUVER", "REJETER"
        ]
        
        if any(indicator in response.upper() for indicator in interrupt_indicators):
            return True
        
        if (hasattr(self.system.chat_manager, 'pending_interrupts') and 
            session_id in self.system.chat_manager.pending_interrupts):
            return True
            
        return False
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*80)
        print("📊 INTERRUPT TEST SUMMARY")
        print("="*80)
        
        total = len(self.test_results)
        passed = len([r for r in self.test_results if r["status"] == "PASS"])
        failed = len([r for r in self.test_results if r["status"] == "FAIL"])
        errors = len([r for r in self.test_results if r["status"] == "ERROR"])
        
        print(f"📈 Total Tests: {total}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"🚨 Errors: {errors}")
        
        if passed > 0:
            print(f"\n🎉 Successful Tests:")
            for result in self.test_results:
                if result["status"] == "PASS":
                    print(f"  - {result['test_name']}")
        
        if failed > 0 or errors > 0:
            print(f"\n💥 Failed/Error Tests:")
            for result in self.test_results:
                if result["status"] in ["FAIL", "ERROR"]:
                    print(f"  - {result['test_name']}: {result.get('error', 'Unknown error')}")
        
        print("="*80)


async def run_interrupt_tests():
    """Run specialized tests for human approval interrupts"""
    system = MultiCountryLegalRAGSystem()
    tester = InterruptTester(system)
    
    try:
        print("🚀 Initializing system...")
        success = await system.initialize()
        if not success:
            print("❌ System initialization failed")
            return

        print("\n🧪 STARTING INTERRUPT TESTS")
        print("="*60)

        test_scenarios = [
            {
                "name": "Complete Workflow - Approve",
                "user_query": "Je veux parler a un avocat",
                "user_email": "test@example.com",
                "user_description": "Consultation pour divorce au Benin",
                "moderator_response": "approve Demande legitime"
            },
            {
                "name": "Complete Workflow - Reject", 
                "user_query": "Contactez-moi",
                "user_email": "test2@example.com",
                "user_description": "J'ai besoin d'aide",
                "moderator_response": "reject Description trop vague"
            }
        ]

        for scenario in test_scenarios:
            await tester.test_assistance_workflow(
                scenario["name"],
                scenario["user_query"],
                scenario["user_email"], 
                scenario["user_description"],
                scenario["moderator_response"]
            )
            await asyncio.sleep(1)

        tester.print_summary()

    except Exception as e:
        logging.error(f"❌ Error during testing: {e}")
    finally:
        await system.cleanup()


async def interactive_mode():
    """Run interactive chat mode"""
    system = MultiCountryLegalRAGSystem()
    
    try:
        print("🚀 Initializing system...")
        success = await system.initialize()
        if not success:
            print("❌ System initialization failed")
            return

        print("\n🎯 INTERACTIVE MODE - SCALABLE SYSTEM")
        print("="*60)
        print("Commands:")
        print("  'quit' - Exit")
        print("  'stats' - Show statistics") 
        print("  'health' - Health check")
        print("  'countries' - List available countries")
        print("  'session' - Session info")
        print("="*60)
        
        session_id = f"interactive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"Session ID: {session_id}")
        print(f"Available: {', '.join(system.get_available_countries())}\n")
        
        while True:
            try:
                user_input = input("👤 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                elif user_input.lower() == 'stats':
                    stats = system.get_global_stats()
                    print(f"\n📊 Statistics:")
                    print(f"  Total Queries: {stats.get('total_queries', 0)}")
                    print(f"  Active Sessions: {stats.get('active_sessions', 0)}")
                    print(f"  Pending Interrupts: {stats.get('pending_interrupts', 0)}")
                    continue
                elif user_input.lower() == 'health':
                    health = await system.health_check()
                    print(f"\n❤️  System Health:")
                    print(f"  Status: {'✅ HEALTHY' if health['system_initialized'] else '❌ UNHEALTHY'}")
                    print(f"  Countries: {len(health['available_countries'])} available")
                    print(f"  MongoDB: {'✅ Connected' if health['mongodb_connected'] else '❌ Disconnected'}")
                    continue
                elif user_input.lower() == 'countries':
                    countries = system.get_available_countries()
                    print(f"\n🌍 Available Countries: {', '.join(countries) if countries else 'None'}")
                    continue
                elif user_input.lower() == 'session':
                    info = system.get_session_info(session_id)
                    print(f"\n📋 Session Info:")
                    print(f"  Queries: {info.get('query_count', 0)}")
                    print(f"  Avg Time: {info.get('average_processing_time', 0):.2f}s")
                    continue
                elif not user_input:
                    continue
                
                start_time = time.time()
                response = await system.chat(user_input, session_id)
                response_time = time.time() - start_time
                
                print(f"🤖 Assistant ({response_time:.2f}s): {response}\n")
                
                # Check for interrupt
                if (hasattr(system.chat_manager, 'pending_interrupts') and 
                    session_id in system.chat_manager.pending_interrupts):
                    print("⏸️  💡 SYSTEM PAUSED - Next message treated as moderator decision\n")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}\n")
                
    finally:
        await system.cleanup()


async def health_check_mode():
    """Run system health check only"""
    system = MultiCountryLegalRAGSystem()
    
    try:
        print("🔍 Performing health check...")
        success = await system.initialize()
        
        if success:
            health = await system.health_check()
            print("\n" + "="*50)
            print("📋 SYSTEM HEALTH REPORT")
            print("="*50)
            print(f"✅ System Initialized: {health['system_initialized']}")
            print(f"🌍 Available Countries: {len(health['available_countries'])}")
            print(f"💾 MongoDB: {'✅ Connected' if health['mongodb_connected'] else '❌ Disconnected'}")
            print(f"⏸️  Interrupt Support: {'✅ Enabled' if health['interrupt_enabled'] else '❌ Disabled'}")
            
            print(f"\n🔧 Components:")
            for component, status in health['components'].items():
                print(f"  {component}: {'✅ OK' if status else '❌ Missing'}")
                
            all_healthy = (health['system_initialized'] and 
                          health['mongodb_connected'] and 
                          all(health['components'].values()))
            print(f"\n🎯 Overall Status: {'✅ HEALTHY' if all_healthy else '❌ UNHEALTHY'}")
            
        else:
            print("❌ System initialization failed")
            
    finally:
        await system.cleanup()


async def quick_test_mode():
    """Run a quick single test"""
    system = MultiCountryLegalRAGSystem()
    
    try:
        print("🚀 Quick Test Mode")
        print("Initializing system...")
        success = await system.initialize()
        if not success:
            print("❌ System initialization failed")
            return

        test_query = "Bonjour, quelle est la procedure pour un divorce au Benin?"
        session_id = "quick_test"
        
        print(f"\n🧪 Testing: {test_query}")
        start_time = time.time()
        response = await system.chat(test_query, session_id)
        response_time = time.time() - start_time
        
        print(f"✅ Response ({response_time:.2f}s): {response}")
        
        print(f"\n📊 System Info:")
        print(f"  Available Countries: {', '.join(system.get_available_countries())}")
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
    finally:
        await system.cleanup()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="🚀 Scalable Multi-Country Legal RAG System"
    )
    
    parser.add_argument(
        "--mode",
        choices=["interactive", "health", "interrupt", "quick"],
        default="interactive",
        help="Run mode (default: interactive)"
    )
    
    args = parser.parse_args()
    
    if args.mode == "interactive":
        asyncio.run(interactive_mode())
    elif args.mode == "health":
        asyncio.run(health_check_mode())
    elif args.mode == "interrupt":
        asyncio.run(run_interrupt_tests())
    elif args.mode == "quick":
        asyncio.run(quick_test_mode())