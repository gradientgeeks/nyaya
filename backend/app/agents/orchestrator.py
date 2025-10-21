"""
LangGraph Multi-Agent Orchestrator

Coordinates the four specialized agents based on detected intent.
This is the heart of Nyaya's multi-agent system.
"""

import logging
from typing import Dict, Any, Annotated
import operator
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_vertexai import ChatVertexAI

from ..core.config import Settings
from ..models.schemas import AgentState, Intent
from ..services.intent_detection import IntentDetector
from ..services.preprocessing import DocumentPreprocessor
from ..services.classification_service import ClassificationService
from ..services.embedding_service import EmbeddingService
from ..services.pinecone_service import PineconeService
from ..services.context_manager import ContextManager

from .classification_agent import classification_agent_node
from .similarity_agent import similarity_agent_node
from .prediction_agent import prediction_agent_node
from .rag_agent import rag_agent_node

logger = logging.getLogger(__name__)


class MultiAgentOrchestrator:
    """
    Orchestrates the four specialized agents using LangGraph.
    
    Agents:
    1. **Classification Agent**: Upload → Classify → Embed → Store
    2. **Similarity Agent**: Find similar cases
    3. **Prediction Agent**: Predict outcomes
    4. **RAG Agent**: Role-aware question answering
    
    The orchestrator uses intent detection to route to the appropriate agent.
    """
    
    def __init__(self, settings: Settings):
        self.settings = settings
        
        # Initialize services
        logger.info("🔧 Initializing services...")
        self.intent_detector = IntentDetector()
        self.preprocessor = DocumentPreprocessor()
        self.classifier = ClassificationService(settings)
        self.embedder = EmbeddingService(settings)
        self.pinecone = PineconeService(settings)
        self.context_manager = ContextManager()
        
        # Initialize LLM for RAG
        self.llm = ChatVertexAI(
            model_name=settings.llm_model_name,
            temperature=0.1,  # Low temperature for factual answers
            max_output_tokens=1024
        )
        
        # Build graph
        self.graph = self._build_graph()
        self.checkpointer = MemorySaver()  # In-memory checkpointing
        
        logger.info("✅ MultiAgentOrchestrator initialized")
    
    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph workflow with conditional routing.
        
        Graph structure:
        
        START → detect_intent → [route based on intent]
                                    ↓
                ┌───────────────────┼───────────────────┐
                ↓                   ↓                   ↓
        classification_agent   similarity_agent   prediction_agent   rag_agent
                ↓                   ↓                   ↓              ↓
                └───────────────────┴───────────────────┴──────────→ END
        """
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("detect_intent", self._detect_intent_node)
        workflow.add_node("classification_agent", self._classification_wrapper)
        workflow.add_node("similarity_agent", self._similarity_wrapper)
        workflow.add_node("prediction_agent", self._prediction_wrapper)
        workflow.add_node("rag_agent", self._rag_wrapper)
        
        # Set entry point
        workflow.set_entry_point("detect_intent")
        
        # Add conditional edges from intent detection
        workflow.add_conditional_edges(
            "detect_intent",
            self._route_to_agent,
            {
                "classification_agent": "classification_agent",
                "similarity_agent": "similarity_agent",
                "prediction_agent": "prediction_agent",
                "rag_agent": "rag_agent"
            }
        )
        
        # All agents go to END
        workflow.add_edge("classification_agent", END)
        workflow.add_edge("similarity_agent", END)
        workflow.add_edge("prediction_agent", END)
        workflow.add_edge("rag_agent", END)
        
        return workflow.compile(checkpointer=self.checkpointer)
    
    async def _detect_intent_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Intent detection node.
        
        Determines which agent to route to based on:
        - User query content
        - File attachment presence
        - Session context
        """
        logger.info("🔍 Detecting intent...")
        
        query = state.get("user_query", "")
        has_file = bool(state.get("file_path"))
        session_id = state.get("session_id")
        
        # Get session context
        session_context = None
        if session_id:
            session = self.context_manager.get_session(session_id)
            if session:
                session_context = {
                    "active_case_id": session.active_case_id,
                    "has_history": len(session.conversation_history) > 0
                }
        
        # Detect intent
        intent, target_roles = self.intent_detector.detect_intent(
            query=query,
            has_file=has_file,
            session_context=session_context
        )
        
        logger.info(f"✅ Detected intent: {intent.value}")
        if target_roles:
            logger.info(f"   Target roles: {[r.value for r in target_roles]}")
        
        return {
            "intent": intent,
            "target_roles": target_roles
        }
    
    def _route_to_agent(self, state: AgentState) -> str:
        """
        Route to appropriate agent based on detected intent.
        """
        intent = state.get("intent")
        
        routing_map = {
            Intent.UPLOAD_AND_CLASSIFY: "classification_agent",
            Intent.SIMILARITY_SEARCH: "similarity_agent",
            Intent.PREDICT_OUTCOME: "prediction_agent",
            Intent.ROLE_SPECIFIC_QA: "rag_agent",
            Intent.SEARCH_CASES: "similarity_agent",  # Use similarity for search
            Intent.GENERAL_QA: "rag_agent"
        }
        
        agent = routing_map.get(intent, "rag_agent")  # Default to RAG
        logger.info(f"🔀 Routing to: {agent}")
        
        return agent
    
    async def _classification_wrapper(self, state: AgentState) -> Dict[str, Any]:
        """Wrapper for classification agent with service injection."""
        return await classification_agent_node(
            state=state,
            preprocessor=self.preprocessor,
            classifier=self.classifier,
            embedder=self.embedder,
            pinecone=self.pinecone
        )
    
    async def _similarity_wrapper(self, state: AgentState) -> Dict[str, Any]:
        """Wrapper for similarity agent with service injection."""
        return await similarity_agent_node(
            state=state,
            embedder=self.embedder,
            pinecone=self.pinecone,
            role_weights=self.settings.similarity_role_weights
        )
    
    async def _prediction_wrapper(self, state: AgentState) -> Dict[str, Any]:
        """Wrapper for prediction agent with service injection."""
        return await prediction_agent_node(
            state=state,
            embedder=self.embedder,
            pinecone=self.pinecone
        )
    
    async def _rag_wrapper(self, state: AgentState) -> Dict[str, Any]:
        """Wrapper for RAG agent with service injection."""
        return await rag_agent_node(
            state=state,
            embedder=self.embedder,
            pinecone=self.pinecone,
            llm=self.llm,
            top_k=self.settings.rag_top_k
        )
    
    async def process_query(
        self,
        user_query: str,
        session_id: str,
        file_path: str = None,
        case_id: str = None
    ) -> Dict[str, Any]:
        """
        Process a user query through the multi-agent system.
        
        Args:
            user_query: User's question or command
            session_id: Session identifier
            file_path: Optional file path (for uploads)
            case_id: Optional case ID (for specific case queries)
        
        Returns:
            Dict with final answer and metadata
        """
        logger.info(f"📝 Processing query: '{user_query[:50]}...'")
        
        # Build initial state
        initial_state = AgentState(
            messages=[{"role": "user", "content": user_query}],
            user_query=user_query,
            session_id=session_id,
            file_path=file_path,
            case_id=case_id,
            intent=None,
            target_roles=[],
            search_results=[],
            final_answer=""
        )
        
        # Run through graph
        config = {"configurable": {"thread_id": session_id}}
        
        try:
            final_state = await self.graph.ainvoke(initial_state, config)
            
            # Add to conversation history
            self.context_manager.add_message(
                session_id=session_id,
                role="user",
                content=user_query
            )
            
            if final_state.get("final_answer"):
                self.context_manager.add_message(
                    session_id=session_id,
                    role="assistant",
                    content=final_state["final_answer"]
                )
            
            # Update active case if classification was performed
            if final_state.get("classification_result"):
                result = final_state["classification_result"]
                self.context_manager.update_session(
                    session_id=session_id,
                    active_case_id=result.get("case_id")
                )
            
            logger.info("✅ Query processing complete")
            
            return {
                "answer": final_state.get("final_answer", ""),
                "intent": final_state.get("intent", {}).value if final_state.get("intent") else None,
                "classification_result": final_state.get("classification_result"),
                "search_results": final_state.get("search_results", []),
                "prediction": final_state.get("prediction"),
                "rag_response": final_state.get("rag_response")
            }
            
        except Exception as e:
            logger.error(f"❌ Error processing query: {e}", exc_info=True)
            
            return {
                "answer": f"❌ Error processing your request: {str(e)}",
                "error": str(e)
            }
