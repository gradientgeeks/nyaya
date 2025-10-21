"""
Pydantic models for API requests and responses.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class RhetoricalRole(str, Enum):
    """The 7 rhetorical roles in legal documents."""
    FACTS = "Facts"
    ISSUE = "Issue"
    ARGUMENTS_PETITIONER = "Arguments of Petitioner"
    ARGUMENTS_RESPONDENT = "Arguments of Respondent"
    REASONING = "Reasoning"
    DECISION = "Decision"
    NONE = "None"


class Intent(str, Enum):
    """User intent types."""
    UPLOAD_AND_CLASSIFY = "UPLOAD_AND_CLASSIFY"
    SEARCH_EXISTING_CASES = "SEARCH_EXISTING_CASES"
    SIMILARITY_SEARCH = "SIMILARITY_SEARCH"
    PREDICT_OUTCOME = "PREDICT_OUTCOME"
    ROLE_SPECIFIC_QA = "ROLE_SPECIFIC_QA"
    GENERAL_QA = "GENERAL_QA"


# ============================================================================
# Request Models
# ============================================================================

class UploadDocumentRequest(BaseModel):
    """Request model for document upload."""
    # File will be handled separately via FastAPI's UploadFile
    case_title: Optional[str] = Field(None, description="Optional case title")
    case_metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


class QueryRequest(BaseModel):
    """Request model for querying the system."""
    query: str = Field(..., min_length=1, max_length=1000, description="User's question")
    session_id: Optional[str] = Field(None, description="Session ID for conversation context")
    case_id: Optional[str] = Field(None, description="Specific case to query")
    role_filter: Optional[List[RhetoricalRole]] = Field(None, description="Filter by specific roles")


class SearchRequest(BaseModel):
    """Request model for searching cases."""
    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of results")
    case_id: Optional[str] = Field(None, description="Find cases similar to this case")


class PredictOutcomeRequest(BaseModel):
    """Request model for outcome prediction."""
    case_id: str = Field(..., description="Case ID to predict outcome for")
    session_id: Optional[str] = Field(None, description="Session ID")


# ============================================================================
# Response Models
# ============================================================================

class ClassifiedSentence(BaseModel):
    """A single sentence with its classified role."""
    text: str
    role: RhetoricalRole
    confidence: float
    sentence_index: int


class ClassificationResult(BaseModel):
    """Result of document classification."""
    case_id: str
    total_sentences: int
    role_distribution: Dict[str, int]
    sentences: List[ClassifiedSentence]
    processing_time: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SimilarCase(BaseModel):
    """Information about a similar case."""
    case_id: str
    case_title: Optional[str] = None
    overall_similarity: float
    role_scores: Dict[str, float]
    matching_roles: List[str]
    snippet: Optional[str] = None


class SearchResult(BaseModel):
    """Result of case search."""
    query: str
    similar_cases: List[SimilarCase]
    total_found: int
    processing_time: float


class PredictionResult(BaseModel):
    """Result of outcome prediction."""
    case_id: str
    predicted_outcome: str  # "Favorable", "Unfavorable", "Neutral"
    confidence: float
    based_on_cases: int
    key_factors: List[str]
    similar_precedents: List[SimilarCase]
    processing_time: float


class RAGResponse(BaseModel):
    """Response from RAG query."""
    query: str
    answer: str
    sources: List[Dict[str, Any]]
    role_filter_applied: Optional[List[RhetoricalRole]] = None
    confidence: float
    processing_time: float


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    answer: str
    intent: Optional[str] = None
    classification_result: Optional[Dict[str, Any]] = None
    search_results: Optional[List[Dict[str, Any]]] = None
    prediction: Optional[Dict[str, Any]] = None
    rag_response: Optional[Dict[str, Any]] = None
    session_id: str


class SessionResponse(BaseModel):
    """Response model for session creation."""
    session_id: str
    message: str = "Session created successfully"


class ChatMessage(BaseModel):
    """Chat message model for conversation history."""
    role: str  # 'user' or 'assistant'
    content: str
    message_type: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ============================================================================
# Internal Models (for LangGraph state)
# ============================================================================

class AgentState(BaseModel):
    """State for LangGraph agent - using TypedDict compatible structure."""
    # Conversation
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    user_query: str = ""
    session_id: Optional[str] = None
    
    # Intent & Routing
    intent: Optional[Intent] = None
    target_roles: List[RhetoricalRole] = Field(default_factory=list)
    
    # File Upload
    file_path: Optional[str] = None
    case_id: Optional[str] = None
    
    # Classification Results
    classification_result: Optional[Dict[str, Any]] = None
    
    # Search & Retrieval
    search_results: List[Any] = Field(default_factory=list)
    rag_response: Optional[Any] = None
    
    # Prediction
    prediction: Optional[Any] = None
    
    # Final Output
    final_answer: str = ""
    
    class Config:
        arbitrary_types_allowed = True


class SessionContext(BaseModel):
    """Session context for conversation management."""
    session_id: str
    user_id: Optional[str] = None
    active_case_id: Optional[str] = None
    conversation_history: List[ChatMessage] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        arbitrary_types_allowed = True
