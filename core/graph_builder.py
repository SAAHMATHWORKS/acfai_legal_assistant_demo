# [file name]: core/graph_builder.py
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
import logging
from typing import Dict, List, Any
from langchain_core.runnables import RunnableConfig

from models.state_models import MultiCountryLegalState
from core.router import CountryRouter
from core.retriever import LegalRetriever
from core.conversation_repair import ConversationRepair 
from core.human_approval_node import HumanApprovalNode

# Import modular components
from core.nodes.routing_nodes import RoutingNodes
from core.assistance.workflow_nodes import AssistanceWorkflowNodes
from core.nodes.retrieval_nodes import RetrievalNodes
from core.nodes.response_nodes import ResponseNodes
from core.nodes.helper_nodes import HelperNodes
from core.routing.routing_logic import RoutingLogic

logger = logging.getLogger(__name__)

class GraphBuilder:
    def __init__(
        self, 
        router: CountryRouter,
        llm, 
        checkpointer: AsyncPostgresSaver,
        # Country retrievers as a dictionary for easy extension
        country_retrievers: Dict[str, LegalRetriever] = None
    ):
        self.router = router
        self.llm = llm
        self.checkpointer = checkpointer
        
        # Initialize country retrievers - easily extensible!
        self.country_retrievers = country_retrievers or {}
        
        # Initialize modular components
        self.conversation_repair = ConversationRepair()
        self.human_approval = HumanApprovalNode()
        self.routing_logic = RoutingLogic()
        
        # Initialize node groups
        self.routing_nodes = RoutingNodes(router, self.conversation_repair, llm)
        self.assistance_nodes = AssistanceWorkflowNodes()
        
        # Dynamic retrieval nodes based on available countries
        self.retrieval_nodes = RetrievalNodes(self.country_retrievers)
        
        self.response_nodes = ResponseNodes(llm)
        self.helper_nodes = HelperNodes(llm)
        
        logger.info(f"GraphBuilder initialized with countries: {list(self.country_retrievers.keys())}")

    def add_country(self, country_code: str, retriever: LegalRetriever):
        """Dynamically add a new country to the system"""
        self.country_retrievers[country_code] = retriever
        self.retrieval_nodes = RetrievalNodes(self.country_retrievers)  # Re-initialize
        logger.info(f"Added country: {country_code}")

    def build_graph(self) -> StateGraph:
        """Build simplified flow with all routing categories"""
        workflow = StateGraph(MultiCountryLegalState)
        
        # Core nodes
        workflow.add_node("router", self.routing_nodes.router_node)
        workflow.add_node("response", self.response_nodes.response_generation_node)
        
        # Country retrieval nodes - dynamically created
        country_nodes = {}
        for country_code in self.country_retrievers.keys():
            node_name = f"{country_code}_retrieval"
            workflow.add_node(node_name, self._create_country_retrieval_node(country_code))
            country_nodes[country_code] = node_name
        
        # Handler nodes
        workflow.add_node("greeting_handler", self.routing_nodes.greeting_small_talk_node)
        workflow.add_node("repair_handler", self.routing_nodes.conversation_repair_node)
        workflow.add_node("summary_handler", self.helper_nodes.conversation_summarization_node)
        workflow.add_node("unclear_handler", self.helper_nodes.unclear_route_node)
        workflow.add_node("out_of_scope_handler", self.helper_nodes.out_of_scope_node)
        
        # Assistance nodes - Using wrapper methods to ensure correct signatures
        workflow.add_node("assistance_collect_info", self._create_assistance_collect_wrapper())
        workflow.add_node("assistance_confirm", self._create_assistance_confirm_wrapper())
        workflow.add_node("human_approval", self.human_approval.process_approval)
        workflow.add_node("process_assistance", self._create_process_assistance_node)

        # Main flow
        workflow.add_edge(START, "router")
        
        # Router directly routes to appropriate nodes
        workflow.add_conditional_edges(
            "router",
            self._route_after_router,
            {
                **country_nodes,  # benin_retrieval, madagascar_retrieval, etc.
                "greeting_small_talk": "greeting_handler",
                "conversation_repair": "repair_handler",
                "conversation_summarization": "summary_handler", 
                "unclear": "unclear_handler",
                "out_of_scope": "out_of_scope_handler",
                "assistance_request": "assistance_collect_info"
            }
        )
        
        # All handlers go to response
        workflow.add_edge("greeting_handler", "response")
        workflow.add_edge("repair_handler", "response")
        workflow.add_edge("summary_handler", "response")
        workflow.add_edge("unclear_handler", "response")
        workflow.add_edge("out_of_scope_handler", "response")
        
        # Country nodes go to response
        for country_code in self.country_retrievers.keys():
            workflow.add_edge(f"{country_code}_retrieval", "response")
        
        # Assistance sub-flow
        workflow.add_conditional_edges(
            "assistance_collect_info",
            self.routing_logic.route_after_info_collection,
            {
                "need_email": "response",  # Ask for email
                "need_description": "response",  # Ask for description
                "ready_to_confirm": "assistance_confirm",
                "cancelled": "response" 
            }
        )
        
        # CRITICAL FIX: After response, only continue assistance if we have new user input
        workflow.add_conditional_edges(
            "response",
            self._route_after_response,
            {
                "continue_assistance": "assistance_collect_info",
                "end": END
            }
        )
        
        workflow.add_conditional_edges(
            "assistance_confirm",
            self.routing_logic.route_after_confirmation,
            {
                "confirmed": "human_approval",
                "cancelled": "response",
                "needs_correction": "response"
            }
        )
        
        workflow.add_conditional_edges(
            "human_approval",
            self.routing_logic.route_after_human_approval,
            {
                "approved": "process_assistance",
                "rejected": "response", 
                "interrupt": "response"
            }
        )
        
        workflow.add_edge("process_assistance", "response")
        
        logger.info(f"Scalable graph built for {len(self.country_retrievers)} countries: {list(self.country_retrievers.keys())}")
        return workflow
    
    def _create_assistance_collect_wrapper(self):
        """Wrapper to ensure proper method signature for assistance collection"""
        async def wrapper(state: MultiCountryLegalState, config: RunnableConfig) -> Dict[str, Any]:
            result = await self.assistance_nodes.collect_assistance_info_node(state, config)
            # Ensure supplemental_message is included if not present
            if "supplemental_message" not in result:
                result["supplemental_message"] = ""
            return result
        return wrapper
    
    def _create_assistance_confirm_wrapper(self):
        """Wrapper to ensure proper method signature for assistance confirmation"""
        async def wrapper(state: MultiCountryLegalState, config: RunnableConfig) -> Dict[str, Any]:
            result = await self.assistance_nodes.confirm_assistance_send_node(state, config)
            # Ensure supplemental_message is included if not present
            if "supplemental_message" not in result:
                result["supplemental_message"] = ""
            return result
        return wrapper
    
    def _route_after_router(self, state: MultiCountryLegalState) -> str:
        """Route directly from router - single source of truth"""
        router_decision = state.router_decision or "unclear"
        logger.debug(f"Routing from router: {router_decision}")
        return router_decision
    
    def _route_after_response(self, state: MultiCountryLegalState) -> str:
        """Route after response - check if we should continue assistance workflow"""
        # Check if we're in the middle of an assistance workflow
        assistance_step = state.assistance_step
        if assistance_step and assistance_step not in [None, "cancelled", "completed"]:
            # CRITICAL FIX: Only continue if we have new user input to process
            # This prevents infinite loops when no new user input is available
            has_new_user_input = self._has_new_user_input(state)
            
            if has_new_user_input:
                logger.info(f"🔄 Continuing assistance workflow from response: {assistance_step}")
                return "continue_assistance"
            else:
                logger.info("⏸️  No new user input - waiting for user response")
                return "end"
        
        # Normal end of conversation
        logger.debug("✅ Ending conversation - no assistance workflow active")
        return "end"
    
    def _has_new_user_input(self, state: MultiCountryLegalState) -> bool:
        """Check if there's new user input to process in assistance workflow"""
        if not state.messages:
            return False
        
        # Get the last message
        last_message = state.messages[-1] if state.messages else None
        
        # Check if the last message is from user and not already processed
        if last_message and last_message.get("role") == "user":
            # Check message metadata to see if it's been processed in current assistance step
            message_meta = last_message.get("meta", {})
            processed_in_step = message_meta.get("processed_in_assistance_step")
            current_step = state.assistance_step
            
            # If this message hasn't been processed in the current assistance step, it's new input
            if processed_in_step != current_step:
                logger.info(f"📥 New user input detected for assistance step: {current_step}")
                return True
        
        logger.info("📭 No new user input detected")
        return False

    def _create_country_retrieval_node(self, country_code: str):
        """Create a dynamic country retrieval node (closure factory)"""
        async def country_retrieval_node(state: MultiCountryLegalState, config: RunnableConfig) -> Dict[str, Any]:
            logger.info(f"Country retrieval for: {country_code}")
            return await self.retrieval_nodes.country_retrieval_node(state, config, country_code)
        return country_retrieval_node

    async def _create_process_assistance_node(self, state: MultiCountryLegalState, config: RunnableConfig) -> Dict[str, Any]:
        """Process assistance after approval"""
        logger.info("Processing assistance request")
        
        # Mark assistance as completed with supplemental message
        return {
            "email_status": "sent",
            "approval_status": "approved", 
            "assistance_step": "completed",
            "messages": [],
            # "supplemental_message": "Votre demande d'assistance a été traitée avec succès et envoyée à notre équipe juridique."
        }

    def debug_state(self, state: MultiCountryLegalState, step: str) -> None:
        """Debug state information"""
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"=== STATE DEBUG at {step} ===")
            logger.debug(f"Router decision: {getattr(state, 'router_decision', 'None')}")
            logger.debug(f"Assistance step: {getattr(state, 'assistance_step', 'None')}")
            logger.debug(f"User email: {getattr(state, 'user_email', 'None')}")
            logger.debug(f"Assistance description: {getattr(state, 'assistance_description', 'None')}")
            logger.debug(f"Supplemental message: {getattr(state, 'supplemental_message', 'None')}")
            logger.debug(f"Available countries: {list(self.country_retrievers.keys())}")
            logger.debug("=== END STATE DEBUG ===")