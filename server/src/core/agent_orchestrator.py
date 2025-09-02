"""
Agent Orchestrator for Legal Document Analysis

This module implements the intelligent routing system that directs queries to 
appropriate retrievers, tools, and processors based on query intent and content type.
"""

import logging
import re
from typing import List, Dict, Any, Optional, Union, Tuple
from enum import Enum
from dataclasses import dataclass

from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatVertexAI
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

from .legal_rag import LegalRAGSystem, RhetoricalRole
from .conversation_manager import ConversationManager, MessageType
from .document_processor import LegalDocumentProcessor

logger = logging.getLogger(__name__)

class QueryType(Enum):
    """Types of user queries"""
    DOCUMENT_ANALYSIS = "document_analysis"
    ROLE_SPECIFIC_QUERY = "role_specific_query"
    CASE_SUMMARY = "case_summary"
    PRECEDENT_SEARCH = "precedent_search"
    LEGAL_RESEARCH = "legal_research"
    PROCEDURAL_QUERY = "procedural_query"
    DOCUMENT_UPLOAD = "document_upload"
    CONVERSATION_QUERY = "conversation_query"
    PREDICTION_REQUEST = "prediction_request"

class Intent(Enum):
    """User intent categories"""
    SEARCH = "search"
    SUMMARIZE = "summarize"
    ANALYZE = "analyze"
    COMPARE = "compare"
    EXPLAIN = "explain"
    PREDICT = "predict"
    UPLOAD = "upload"
    CLARIFY = "clarify"

@dataclass
class QueryClassification:
    """Classification result for user query"""
    query_type: QueryType
    intent: Intent
    relevant_roles: List[str]
    confidence: float
    requires_context: bool
    suggested_tools: List[str]
    metadata: Dict[str, Any]

class QueryRouter:
    """
    Intelligent query routing based on content analysis and intent detection
    """
    
    def __init__(self):
        """Initialize query router with classification patterns"""
        self.role_keywords = {
            RhetoricalRole.FACTS.value: [
                "facts", "background", "what happened", "events", "circumstances",
                "story", "timeline", "incident", "details", "sequence of events"
            ],
            RhetoricalRole.ISSUE.value: [
                "issue", "question", "problem", "dispute", "legal question",
                "main issue", "central question", "point of law", "controversy"
            ],
            RhetoricalRole.ARGUMENTS_PETITIONER.value: [
                "petitioner", "plaintiff", "appellant", "arguments for",
                "petitioner argues", "plaintiff claims", "case for",
                "petitioner's position", "grounds of appeal"
            ],
            RhetoricalRole.ARGUMENTS_RESPONDENT.value: [
                "respondent", "defendant", "appellee", "arguments against",
                "respondent argues", "defendant claims", "defense",
                "respondent's position", "counter arguments"
            ],
            RhetoricalRole.REASONING.value: [
                "reasoning", "rationale", "why", "because", "analysis",
                "court reasoning", "legal analysis", "justification",
                "judicial reasoning", "court's analysis", "legal principle"
            ],
            RhetoricalRole.DECISION.value: [
                "decision", "judgment", "ruling", "verdict", "outcome",
                "final decision", "court decided", "held that", "conclusion",
                "order", "judgment", "disposition"
            ]
        }
        
        self.intent_patterns = {
            Intent.SEARCH: [
                "find", "search", "look for", "locate", "discover",
                "retrieve", "get", "show me", "give me"
            ],
            Intent.SUMMARIZE: [
                "summarize", "summary", "brief", "overview", "gist",
                "key points", "main points", "essence", "synopsis"
            ],
            Intent.ANALYZE: [
                "analyze", "analysis", "examine", "study", "review",
                "evaluate", "assess", "break down", "dissect"
            ],
            Intent.COMPARE: [
                "compare", "comparison", "contrast", "difference",
                "similar", "different", "versus", "vs", "against"
            ],
            Intent.EXPLAIN: [
                "explain", "explanation", "clarify", "elaborate",
                "detail", "describe", "tell me about", "what is"
            ],
            Intent.PREDICT: [
                "predict", "prediction", "forecast", "likely outcome",
                "probable result", "chances", "likelihood", "expect"
            ],
            Intent.UPLOAD: [
                "upload", "add document", "process document", "analyze document",
                "submit file", "load document"
            ]
        }
        
        self.query_type_patterns = {
            QueryType.DOCUMENT_ANALYSIS: [
                "analyze document", "document analysis", "process document",
                "examine document", "review document"
            ],
            QueryType.CASE_SUMMARY: [
                "case summary", "summarize case", "case overview",
                "brief of case", "case brief"
            ],
            QueryType.PRECEDENT_SEARCH: [
                "precedent", "similar case", "case law", "prior decision",
                "earlier ruling", "established law"
            ],
            QueryType.LEGAL_RESEARCH: [
                "legal research", "law research", "statute", "regulation",
                "legal provision", "constitutional provision"
            ],
            QueryType.PROCEDURAL_QUERY: [
                "procedure", "process", "how to", "steps", "filing",
                "court procedure", "legal process"
            ]
        }
    
    def classify_query(self, query: str, context: Dict[str, Any] = None) -> QueryClassification:
        """
        Classify user query to determine routing strategy
        
        Args:
            query: User query text
            context: Additional context from conversation
            
        Returns:
            QueryClassification with routing information
        """
        query_lower = query.lower()
        context = context or {}
        
        # Detect query type
        query_type = self._detect_query_type(query_lower)
        
        # Detect intent
        intent = self._detect_intent(query_lower)
        
        # Detect relevant roles
        relevant_roles = self._detect_relevant_roles(query_lower)
        
        # Determine if context is required
        requires_context = self._requires_conversation_context(query_lower, context)
        
        # Suggest appropriate tools
        suggested_tools = self._suggest_tools(query_type, intent, relevant_roles)
        
        # Calculate confidence based on keyword matches
        confidence = self._calculate_confidence(query_lower, query_type, intent, relevant_roles)
        
        return QueryClassification(
            query_type=query_type,
            intent=intent,
            relevant_roles=relevant_roles,
            confidence=confidence,
            requires_context=requires_context,
            suggested_tools=suggested_tools,
            metadata={
                "query_length": len(query),
                "has_legal_terms": self._has_legal_terms(query_lower),
                "is_question": query.strip().endswith('?'),
                "context_available": bool(context)
            }
        )
    
    def _detect_query_type(self, query: str) -> QueryType:
        """Detect the type of query"""
        for qtype, patterns in self.query_type_patterns.items():
            if any(pattern in query for pattern in patterns):
                return qtype
        
        # Default classification based on content
        if any(word in query for word in ["facts", "background", "what happened"]):
            return QueryType.ROLE_SPECIFIC_QUERY
        elif any(word in query for word in ["summary", "summarize", "overview"]):
            return QueryType.CASE_SUMMARY
        elif any(word in query for word in ["upload", "document", "file"]):
            return QueryType.DOCUMENT_UPLOAD
        elif any(word in query for word in ["predict", "outcome", "likely"]):
            return QueryType.PREDICTION_REQUEST
        else:
            return QueryType.LEGAL_RESEARCH
    
    def _detect_intent(self, query: str) -> Intent:
        """Detect user intent"""
        intent_scores = {}
        
        for intent, patterns in self.intent_patterns.items():
            score = sum(1 for pattern in patterns if pattern in query)
            if score > 0:
                intent_scores[intent] = score
        
        if intent_scores:
            return max(intent_scores.items(), key=lambda x: x[1])[0]
        
        # Default intent based on query characteristics
        if query.endswith('?'):
            return Intent.EXPLAIN
        elif any(word in query for word in ["get", "show", "find"]):
            return Intent.SEARCH
        else:
            return Intent.CLARIFY
    
    def _detect_relevant_roles(self, query: str) -> List[str]:
        """Detect which rhetorical roles are relevant to the query"""
        relevant_roles = []
        
        for role, keywords in self.role_keywords.items():
            if any(keyword in query for keyword in keywords):
                relevant_roles.append(role)
        
        # If no specific roles detected, include common ones for legal queries
        if not relevant_roles:
            if any(word in query for word in ["case", "judgment", "court", "legal"]):
                relevant_roles = [
                    RhetoricalRole.FACTS.value,
                    RhetoricalRole.REASONING.value,
                    RhetoricalRole.DECISION.value
                ]
        
        return relevant_roles
    
    def _requires_conversation_context(self, query: str, context: Dict[str, Any]) -> bool:
        """Determine if query requires conversation context"""
        # Pronouns and references that indicate context dependency
        context_indicators = [
            "this", "that", "it", "they", "them", "the case",
            "the document", "mentioned", "above", "earlier",
            "previous", "same", "also", "additionally"
        ]
        
        has_context_indicators = any(indicator in query for indicator in context_indicators)
        has_conversation_history = bool(context.get("conversation_history"))
        
        return has_context_indicators or (has_conversation_history and len(query.split()) < 10)
    
    def _suggest_tools(self, query_type: QueryType, intent: Intent, 
                      relevant_roles: List[str]) -> List[str]:
        """Suggest appropriate tools for handling the query"""
        tools = []
        
        # Add role-specific retriever if roles are identified
        if relevant_roles:
            tools.append("role_specific_retriever")
        
        # Add tools based on query type
        if query_type == QueryType.DOCUMENT_ANALYSIS:
            tools.extend(["document_processor", "role_classifier"])
        elif query_type == QueryType.CASE_SUMMARY:
            tools.extend(["case_summarizer", "structured_response_generator"])
        elif query_type == QueryType.PRECEDENT_SEARCH:
            tools.extend(["precedent_search", "similarity_retriever"])
        elif query_type == QueryType.PREDICTION_REQUEST:
            tools.extend(["prediction_module", "precedent_analyzer"])
        
        # Add tools based on intent
        if intent == Intent.SEARCH:
            tools.append("semantic_search")
        elif intent == Intent.COMPARE:
            tools.append("comparison_analyzer")
        elif intent == Intent.SUMMARIZE:
            tools.append("summarization_engine")
        
        # Always include general RAG for comprehensive responses
        tools.append("general_rag")
        
        return list(set(tools))  # Remove duplicates
    
    def _calculate_confidence(self, query: str, query_type: QueryType, 
                            intent: Intent, relevant_roles: List[str]) -> float:
        """Calculate confidence score for classification"""
        score = 0.0
        total_checks = 4
        
        # Query type confidence
        type_patterns = self.query_type_patterns.get(query_type, [])
        if any(pattern in query for pattern in type_patterns):
            score += 1.0
        
        # Intent confidence
        intent_patterns = self.intent_patterns.get(intent, [])
        if any(pattern in query for pattern in intent_patterns):
            score += 1.0
        
        # Role detection confidence
        if relevant_roles:
            score += 1.0
        
        # Legal term presence
        if self._has_legal_terms(query):
            score += 1.0
        
        return score / total_checks
    
    def _has_legal_terms(self, query: str) -> bool:
        """Check if query contains legal terminology"""
        legal_terms = [
            "court", "judge", "judgment", "case", "law", "legal", "statute",
            "constitution", "article", "section", "petition", "appeal",
            "precedent", "ruling", "decision", "order", "bail", "arrest",
            "conviction", "sentence", "trial", "hearing", "evidence",
            "witness", "testimony", "plaintiff", "defendant", "prosecution",
            "defense", "civil", "criminal", "constitutional", "supreme court",
            "high court", "district court", "writ", "habeas corpus", "mandamus"
        ]
        
        return any(term in query for term in legal_terms)

class AgentOrchestrator:
    """
    Main orchestrator that coordinates all components of the legal analysis system
    """
    
    def __init__(self, 
                 db_path: str = "legal_system.db",
                 vector_db_path: str = "legal_vectors",
                 device: str = "cpu"):
        """
        Initialize the agent orchestrator
        
        Args:
            db_path: Path to conversation database
            vector_db_path: Path to vector database
            device: Device for ML models
        """
        self.device = device
        
        # Initialize components
        self.query_router = QueryRouter()
        self.conversation_manager = ConversationManager(db_path)
        self.document_processor = LegalDocumentProcessor()
        self.rag_system = LegalRAGSystem(device=device)
        
        # Initialize LLM for response generation
        self.llm = ChatVertexAI(
            temperature=0.1,
            model_name="gemini-2.5-flash",
            max_tokens=2048
        )
        
        logger.info("Agent Orchestrator initialized successfully")
    
    def process_query(self, query: str, session_id: str = None, 
                     context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Main query processing function that orchestrates the entire workflow
        
        Args:
            query: User query
            session_id: Optional conversation session ID
            context: Additional context
            
        Returns:
            Comprehensive response with metadata
        """
        try:
            # Get or create session
            if not session_id:
                session_id = self.conversation_manager.start_conversation()
            
            # Add user message to conversation
            self.conversation_manager.add_user_message(session_id, query)
            
            # Get conversation context
            conv_context = self.conversation_manager.get_conversation_for_rag(session_id)
            full_context = {**(context or {}), **conv_context}
            
            # Classify the query
            classification = self.query_router.classify_query(query, full_context)
            
            # Route to appropriate handler
            response = self._route_and_process(query, classification, full_context, session_id)
            
            # Add assistant response to conversation
            self.conversation_manager.add_assistant_message(
                session_id, 
                response["answer"],
                {"classification": classification.__dict__, "tools_used": response.get("tools_used", [])}
            )
            
            # Add session metadata
            response["session_id"] = session_id
            response["classification"] = classification.__dict__
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            error_response = {
                "answer": "I apologize, but I encountered an error while processing your query. Please try rephrasing your question or contact support if the issue persists.",
                "error": str(e),
                "session_id": session_id
            }
            
            if session_id:
                self.conversation_manager.add_assistant_message(
                    session_id, 
                    error_response["answer"],
                    {"error": True, "error_message": str(e)}
                )
            
            return error_response
    
    def _route_and_process(self, query: str, classification: QueryClassification, 
                          context: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """
        Route query to appropriate processing pipeline
        
        Args:
            query: User query
            classification: Query classification result
            context: Full context including conversation
            session_id: Session identifier
            
        Returns:
            Processed response
        """
        tools_used = []
        
        # Handle different query types
        if classification.query_type == QueryType.DOCUMENT_UPLOAD:
            return self._handle_document_upload(query, context, session_id)
        
        elif classification.query_type == QueryType.CASE_SUMMARY:
            tools_used.append("case_summarizer")
            return self._handle_case_summary(query, classification, context)
        
        elif classification.query_type == QueryType.ROLE_SPECIFIC_QUERY:
            tools_used.append("role_specific_retriever")
            return self._handle_role_specific_query(query, classification, context)
        
        elif classification.query_type == QueryType.PRECEDENT_SEARCH:
            tools_used.append("precedent_search")
            return self._handle_precedent_search(query, classification, context)
        
        elif classification.query_type == QueryType.PREDICTION_REQUEST:
            tools_used.append("prediction_module")
            return self._handle_prediction_request(query, classification, context)
        
        elif context.get("immediate_document_query") or context.get("followup_query"):
            tools_used.append("document_specific_rag")
            return self._handle_document_specific_query(query, classification, context)
        
        else:
            # Default to general legal research
            tools_used.append("general_rag")
            return self._handle_general_query(query, classification, context)
    
    def _handle_role_specific_query(self, query: str, classification: QueryClassification, 
                                   context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle queries targeting specific rhetorical roles"""
        # Use role-aware RAG with specific roles
        response = self.rag_system.query_legal_rag(
            query,
            auto_detect_roles=False,
            specific_roles=classification.relevant_roles,
            k=8
        )
        
        # Enhance response with conversation context if needed
        if classification.requires_context and context.get("conversation_history"):
            enhanced_prompt = f"""
            Given the following conversation context:
            {context.get('conversation_history', '')}
            
            And the following retrieved information:
            {response['answer']}
            
            Please provide a comprehensive answer to: {query}
            
            Make sure to reference the conversation context when relevant and provide specific, accurate legal information.
            """
            
            enhanced_response = self.llm.invoke([HumanMessage(content=enhanced_prompt)])
            response["answer"] = enhanced_response.content
        
        response["tools_used"] = ["role_specific_retriever", "legal_rag"]
        return response
    
    def _handle_case_summary(self, query: str, classification: QueryClassification, 
                           context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle case summary requests"""
        # Extract case reference from query or context
        case_ref = self._extract_case_reference(query, context)
        
        if case_ref:
            # Generate comprehensive case summary
            summary = self.rag_system.get_case_summary(case_ref)
            
            response = {
                "answer": self._format_case_summary(summary),
                "case_reference": case_ref,
                "role_breakdown": summary.get("role_statistics", {}),
                "tools_used": ["case_summarizer", "role_classifier"]
            }
        else:
            # General query about case summaries
            response = self._handle_general_query(query, classification, context)
            response["tools_used"] = ["general_rag", "case_summarizer"]
        
        return response
    
    def _handle_precedent_search(self, query: str, classification: QueryClassification, 
                               context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle precedent and similar case searches"""
        # Extract legal issues or facts for precedent search
        search_terms = self._extract_precedent_search_terms(query)
        
        # Search for similar cases focusing on Facts and Issues
        precedent_response = self.rag_system.query_legal_rag(
            query,
            specific_roles=[RhetoricalRole.FACTS.value, RhetoricalRole.ISSUE.value, 
                          RhetoricalRole.REASONING.value, RhetoricalRole.DECISION.value],
            k=10
        )
        
        # Format as precedent analysis
        precedent_prompt = f"""
        Based on the following legal precedents and case information:
        
        {precedent_response['answer']}
        
        Provide a comprehensive precedent analysis for the query: {query}
        
        Structure your response as:
        1. Relevant Precedents
        2. Key Legal Principles
        3. Application to Current Situation
        4. Distinguishing Factors (if any)
        
        Focus on providing practical legal guidance based on established precedent.
        """
        
        enhanced_response = self.llm.invoke([HumanMessage(content=precedent_prompt)])
        
        return {
            "answer": enhanced_response.content,
            "precedent_analysis": True,
            "search_terms": search_terms,
            "retrieved_cases": precedent_response.get("total_documents", 0),
            "tools_used": ["precedent_search", "similarity_retriever", "legal_analysis"]
        }
    
    def _handle_prediction_request(self, query: str, classification: QueryClassification, 
                                 context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle requests for judgment prediction"""
        # This would integrate with the prediction module when implemented
        disclaimer = """
        **LEGAL DISCLAIMER**: The following analysis is for informational purposes only 
        and should not be considered as legal advice. Actual court decisions depend on 
        numerous factors and legal precedents may be interpreted differently.
        """
        
        # For now, provide analysis based on similar cases
        similar_cases_response = self.rag_system.query_legal_rag(
            query,
            specific_roles=[RhetoricalRole.FACTS.value, RhetoricalRole.DECISION.value],
            k=10
        )
        
        prediction_prompt = f"""
        {disclaimer}
        
        Based on the following similar cases and their outcomes:
        
        {similar_cases_response['answer']}
        
        Provide a probabilistic analysis for: {query}
        
        Structure your response as:
        1. Case Analysis
        2. Similar Precedents
        3. Factors Favoring Different Outcomes
        4. Estimated Likelihood Assessment
        5. Key Variables That Could Affect Outcome
        
        Always emphasize the uncertainty inherent in legal predictions.
        """
        
        prediction_response = self.llm.invoke([HumanMessage(content=prediction_prompt)])
        
        return {
            "answer": prediction_response.content,
            "prediction_analysis": True,
            "disclaimer": disclaimer,
            "confidence_level": "Moderate (based on precedent analysis)",
            "tools_used": ["prediction_module", "precedent_analyzer", "legal_reasoning"]
        }
    
    def _handle_general_query(self, query: str, classification: QueryClassification, 
                            context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle general legal queries"""
        response = self.rag_system.query_legal_rag(
            query,
            auto_detect_roles=True,
            k=10
        )
        
        # Enhance with conversation context if needed
        if classification.requires_context and context.get("conversation_history"):
            context_prompt = f"""
            Previous conversation:
            {context.get('conversation_history', '')}
            
            Current query: {query}
            
            Retrieved information:
            {response['answer']}
            
            Please provide a comprehensive answer that takes into account the conversation 
            context and provides accurate legal information.
            """
            
            enhanced_response = self.llm.invoke([HumanMessage(content=context_prompt)])
            response["answer"] = enhanced_response.content
            response["context_enhanced"] = True
        
        response["tools_used"] = ["general_rag", "legal_analysis"]
        return response
    
    def _handle_document_upload(self, query: str, context: Dict[str, Any], 
                              session_id: str) -> Dict[str, Any]:
        """Handle document upload requests"""
        return {
            "answer": "I'm ready to help you analyze a legal document. Please upload your document (PDF or text file) and I'll process it to extract the rhetorical roles (Facts, Issues, Arguments, Reasoning, Decision) and make it searchable for detailed analysis.",
            "upload_ready": True,
            "supported_formats": ["PDF", "TXT"],
            "features": [
                "Rhetorical role classification",
                "Content extraction and cleaning", 
                "Searchable document analysis",
                "Structured summaries"
            ],
            "tools_used": ["document_processor"]
        }
    
    def _extract_case_reference(self, query: str, context: Dict[str, Any]) -> Optional[str]:
        """Extract case reference from query or context"""
        # Look for case names, citations, or references
        case_patterns = [
            r"([A-Z][a-zA-Z\s&\.]+)\s+v\.?\s+([A-Z][a-zA-Z\s&\.]+)",
            r"\(\d{4}\)\s+\d+\s+SCC\s+\d+",
            r"AIR\s+\d{4}\s+SC\s+\d+"
        ]
        
        for pattern in case_patterns:
            match = re.search(pattern, query)
            if match:
                return match.group(0)
        
        return None
    
    def _extract_precedent_search_terms(self, query: str) -> List[str]:
        """Extract key terms for precedent search"""
        # Simple keyword extraction - could be enhanced with NLP
        legal_terms = []
        
        # Remove common words and extract meaningful terms
        words = query.lower().split()
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        
        for word in words:
            if word not in stop_words and len(word) > 3:
                legal_terms.append(word)
        
        return legal_terms[:10]  # Limit to top 10 terms
    
    def _handle_document_specific_query(self, query: str, classification: QueryClassification, 
                                      context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle queries specific to an uploaded document"""
        try:
            document_id = context.get("document_id")
            filename = context.get("filename", "uploaded document")
            role_filter = context.get("role_filter")
            
            # If we have role filter from context, use it; otherwise use detected roles
            target_roles = role_filter if role_filter else classification.relevant_roles
            
            # Query the document with specific focus if we have document context
            if document_id:
                # Enhanced query with document context
                enhanced_query = f"""
                Query about document '{filename}': {query}
                
                Please provide specific information from this document.
                """
                
                response = self.rag_system.query_legal_rag(
                    enhanced_query,
                    auto_detect_roles=not bool(target_roles),
                    specific_roles=target_roles,
                    k=8
                )
            else:
                # Fallback to general document query
                response = self.rag_system.query_legal_rag(
                    query,
                    auto_detect_roles=not bool(target_roles),
                    specific_roles=target_roles,
                    k=8
                )
            
            # Add document context to the response
            if context.get("immediate_document_query"):
                response["answer"] = f"Based on the document '{filename}' you just uploaded:\n\n{response['answer']}"
            elif context.get("followup_query"):
                response["answer"] = f"Continuing analysis of '{filename}':\n\n{response['answer']}"
            
            # Add metadata about document-specific processing
            response.update({
                "document_specific": True,
                "document_id": document_id,
                "filename": filename,
                "roles_queried": target_roles or classification.relevant_roles,
                "tools_used": ["document_specific_rag", "role_classifier", "legal_rag"]
            })
            
            return response
            
        except Exception as e:
            logger.error(f"Error in document-specific query: {e}")
            return {
                "answer": f"I encountered an error while analyzing the document. Please try rephrasing your question or upload the document again. Error: {str(e)}",
                "error": str(e),
                "document_specific": True,
                "tools_used": ["error_handler"]
            }
    
    def _format_case_summary(self, summary: Dict[str, Any]) -> str:
        """Format case summary for presentation"""
        formatted = "## Case Summary\n\n"
        
        role_summaries = summary.get("role_summaries", {})
        for role, content in role_summaries.items():
            if content and content.strip():
                formatted += f"**{role}:**\n{content}\n\n"
        
        # Add statistics
        stats = summary.get("role_statistics", {})
        if stats:
            formatted += "**Document Statistics:**\n"
            for role, count in stats.items():
                formatted += f"- {role}: {count} sentences\n"
        
        return formatted
    
    def upload_document(self, file_content: bytes, filename: str, 
                       session_id: str = None) -> Dict[str, Any]:
        """
        Process uploaded document
        
        Args:
            file_content: Document content as bytes
            filename: Name of the uploaded file
            session_id: Optional session ID
            
        Returns:
            Processing result
        """
        try:
            from io import BytesIO
            
            # Process the document
            file_obj = BytesIO(file_content)
            processed_doc = self.document_processor.process_document(file_obj, filename)
            
            if processed_doc.metadata.processing_status == "completed":
                # Classify rhetorical roles and add to RAG system
                tagged_docs = self.rag_system.process_legal_document(
                    processed_doc.content,
                    doc_metadata={
                        "filename": filename,
                        "case_name": processed_doc.metadata.case_name,
                        "court": processed_doc.metadata.court,
                        "upload_session": session_id
                    }
                )
                
                # Add to vector store
                self.rag_system.add_documents_to_store(tagged_docs)
                
                # Generate summary
                summary = self.rag_system.get_case_summary(processed_doc.content)
                
                # Generate document ID for reference
                import uuid
                document_id = str(uuid.uuid4())
                
                result = {
                    "success": True,
                    "document_processed": True,
                    "document_id": document_id,
                    "filename": filename,
                    "metadata": processed_doc.metadata.dict(),
                    "summary": self._format_case_summary(summary),
                    "rhetorical_roles_found": list(summary.get("role_statistics", {}).keys()),
                    "total_sentences": summary.get("total_sentences", 0),
                    "message": f"Successfully processed '{filename}'. The document has been analyzed and is now searchable."
                }
                
                # Add to conversation if session exists
                if session_id:
                    self.conversation_manager.add_assistant_message(
                        session_id,
                        result["message"],
                        {"document_upload": True, "filename": filename}
                    )
                
                return result
                
            else:
                return {
                    "success": False,
                    "error": processed_doc.metadata.error_message,
                    "message": f"Failed to process '{filename}': {processed_doc.metadata.error_message}"
                }
                
        except Exception as e:
            logger.error(f"Error uploading document: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to process '{filename}' due to an unexpected error."
            }

# Example usage
if __name__ == "__main__":
    # Initialize orchestrator
    orchestrator = AgentOrchestrator()
    
    # Test queries
    test_queries = [
        "What are the facts of the case?",
        "Summarize the Supreme Court judgment in Kesavananda Bharati case",
        "Find similar cases about fundamental rights violations",
        "What is the likely outcome if I file a bail application?",
        "Explain Article 21 of the Constitution"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        response = orchestrator.process_query(query)
        print(f"Answer: {response['answer'][:200]}...")
        print(f"Tools used: {response.get('tools_used', [])}")
        print("-" * 80)