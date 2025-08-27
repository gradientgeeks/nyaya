"""
FastAPI Backend for Legal Document Analysis System

This module provides REST API endpoints for all system functionality including
document upload, role classification, RAG queries, conversation management,
and judgment prediction.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import uvicorn

from src.core.agent_orchestrator import AgentOrchestrator
from src.core.legal_rag import LegalRAGSystem
from src.core.conversation_manager import ConversationManager, MessageType
from src.core.document_processor import LegalDocumentProcessor
from src.core.prediction_module import JudgmentPredictor, CaseType, JudgmentOutcome
from src.models.role_classifier import RoleClassifier

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for API requests/responses
class QueryRequest(BaseModel):
    """Request model for legal queries"""
    query: str = Field(..., description="Legal query text")
    session_id: Optional[str] = Field(None, description="Conversation session ID")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    role_filter: Optional[List[str]] = Field(None, description="Filter by specific rhetorical roles")

class QueryResponse(BaseModel):
    """Response model for legal queries"""
    answer: str = Field(..., description="Generated answer")
    session_id: str = Field(..., description="Conversation session ID")
    confidence: Optional[float] = Field(None, description="Response confidence score")
    sources: Optional[List[Dict[str, Any]]] = Field(None, description="Source documents")
    classification: Optional[Dict[str, Any]] = Field(None, description="Query classification")
    tools_used: Optional[List[str]] = Field(None, description="Tools used for processing")

class DocumentUploadResponse(BaseModel):
    """Response model for document upload"""
    success: bool = Field(..., description="Upload success status")
    document_id: Optional[str] = Field(None, description="Document identifier")
    filename: str = Field(..., description="Uploaded filename")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Extracted metadata")
    summary: Optional[str] = Field(None, description="Document summary")
    message: str = Field(..., description="Status message")

class PredictionRequest(BaseModel):
    """Request model for judgment prediction"""
    case_facts: str = Field(..., description="Facts of the case")
    case_issues: Optional[str] = Field(None, description="Legal issues")
    case_type: Optional[str] = Field(None, description="Type of case")
    session_id: Optional[str] = Field(None, description="Conversation session ID")

class PredictionResponse(BaseModel):
    """Response model for judgment prediction"""
    predicted_outcome: str = Field(..., description="Predicted judgment outcome")
    confidence: float = Field(..., description="Prediction confidence")
    probability_distribution: Dict[str, float] = Field(..., description="Outcome probabilities")
    similar_cases: List[Dict[str, Any]] = Field(..., description="Similar precedent cases")
    key_factors: List[str] = Field(..., description="Key influencing factors")
    reasoning: str = Field(..., description="Prediction reasoning")
    disclaimer: str = Field(..., description="Legal disclaimer")

class ConversationInfo(BaseModel):
    """Conversation session information"""
    session_id: str = Field(..., description="Session identifier")
    title: str = Field(..., description="Conversation title")
    status: str = Field(..., description="Session status")
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")
    message_count: int = Field(..., description="Number of messages")

class RoleClassificationRequest(BaseModel):
    """Request for rhetorical role classification"""
    text: str = Field(..., description="Legal document text")
    context_mode: Optional[str] = Field("prev", description="Context mode for classification")

class RoleClassificationResponse(BaseModel):
    """Response for rhetorical role classification"""
    sentences: List[Dict[str, Any]] = Field(..., description="Classified sentences with roles")
    statistics: Dict[str, int] = Field(..., description="Role distribution statistics")

# Initialize the FastAPI app
app = FastAPI(
    title="Nyaya - Legal Document Analysis API",
    description="Intelligent Agent for Legal Document Analysis using RAG, Role Classifier, and Multi-turn Conversation Support",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Global components (initialized on startup)
orchestrator: Optional[AgentOrchestrator] = None
rag_system: Optional[LegalRAGSystem] = None
conversation_manager: Optional[ConversationManager] = None
document_processor: Optional[LegalDocumentProcessor] = None
role_classifier: Optional[RoleClassifier] = None
prediction_module: Optional[JudgmentPredictor] = None

@app.on_event("startup")
async def startup_event():
    """Initialize system components on startup"""
    global orchestrator, rag_system, conversation_manager, document_processor, role_classifier, prediction_module
    
    try:
        logger.info("Initializing Legal Document Analysis System...")
        
        # Initialize components
        logger.info("Loading Agent Orchestrator...")
        orchestrator = AgentOrchestrator()
        
        logger.info("Loading RAG System...")
        rag_system = orchestrator.rag_system
        
        logger.info("Loading Conversation Manager...")
        conversation_manager = orchestrator.conversation_manager
        
        logger.info("Loading Document Processor...")
        document_processor = orchestrator.document_processor
        
        logger.info("Loading Role Classifier...")
        role_classifier = orchestrator.rag_system.role_classifier
        
        logger.info("Loading Prediction Module...")
        prediction_module = JudgmentPredictor(rag_system)
        
        logger.info("System initialization completed successfully!")
        
    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")
        raise e

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Legal Document Analysis System...")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "system": "Legal Document Analysis API",
        "version": "1.0.0"
    }

# Authentication dependency (placeholder - implement proper auth as needed)
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current user from authentication token"""
    # Implement proper authentication here
    return {"user_id": "demo_user", "permissions": ["read", "write"]}

# Main query endpoint
@app.post("/api/query", response_model=QueryResponse)
async def process_query(
    request: QueryRequest,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Process a legal query using the agent orchestrator
    """
    try:
        if not orchestrator:
            raise HTTPException(status_code=503, detail="System not initialized")
        
        # Process the query
        result = orchestrator.process_query(
            query=request.query,
            session_id=request.session_id,
            context=request.context
        )
        
        return QueryResponse(
            answer=result["answer"],
            session_id=result["session_id"],
            confidence=result.get("confidence"),
            sources=result.get("sources"),
            classification=result.get("classification"),
            tools_used=result.get("tools_used", [])
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Document upload endpoint
@app.post("/api/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    session_id: Optional[str] = None,
    user: Dict[str, Any] = Depends(get_current_user),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Upload and process a legal document
    """
    try:
        if not orchestrator:
            raise HTTPException(status_code=503, detail="System not initialized")
        
        # Validate file type
        allowed_types = [".pdf", ".txt", ".text"]
        file_extension = Path(file.filename).suffix.lower()
        
        if file_extension not in allowed_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type. Allowed types: {allowed_types}"
            )
        
        # Read file content
        content = await file.read()
        
        # Process document
        result = orchestrator.upload_document(
            file_content=content,
            filename=file.filename,
            session_id=session_id
        )
        
        return DocumentUploadResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Role classification endpoint
@app.post("/api/classify/roles", response_model=RoleClassificationResponse)
async def classify_rhetorical_roles(
    request: RoleClassificationRequest,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Classify rhetorical roles in legal document text
    """
    try:
        if not role_classifier:
            raise HTTPException(status_code=503, detail="Role classifier not initialized")
        
        # Classify rhetorical roles
        results = role_classifier.classify_document(
            document_text=request.text,
            context_mode=request.context_mode
        )
        
        # Calculate statistics
        role_stats = {}
        for result in results:
            role = result["role"]
            role_stats[role] = role_stats.get(role, 0) + 1
        
        return RoleClassificationResponse(
            sentences=results,
            statistics=role_stats
        )
        
    except Exception as e:
        logger.error(f"Error in role classification: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Judgment prediction endpoint
@app.post("/api/predict/judgment", response_model=PredictionResponse)
async def predict_judgment(
    request: PredictionRequest,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Predict judgment outcome for a pending case
    """
    try:
        if not prediction_module:
            raise HTTPException(status_code=503, detail="Prediction module not initialized")
        
        # Parse case type if provided
        case_type = None
        if request.case_type:
            try:
                case_type = CaseType(request.case_type.lower())
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid case type: {request.case_type}")
        
        # Make prediction
        prediction = prediction_module.predict_judgment(
            case_facts=request.case_facts,
            case_issues=request.case_issues,
            case_type=case_type
        )
        
        # Format similar cases for response
        similar_cases_data = []
        for case in prediction.similar_cases:
            similar_cases_data.append({
                "case_name": case.case_name,
                "court": case.court,
                "year": case.year,
                "outcome": case.outcome.value,
                "similarity_score": case.similarity_score,
                "citation": case.citation
            })
        
        return PredictionResponse(
            predicted_outcome=prediction.predicted_outcome.value,
            confidence=prediction.confidence,
            probability_distribution=prediction.probability_distribution,
            similar_cases=similar_cases_data,
            key_factors=prediction.key_factors,
            reasoning=prediction.reasoning,
            disclaimer=prediction.disclaimer
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in judgment prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Conversation management endpoints
@app.post("/api/conversations/start")
async def start_conversation(
    title: Optional[str] = "Legal Consultation",
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Start a new conversation session
    """
    try:
        if not conversation_manager:
            raise HTTPException(status_code=503, detail="Conversation manager not initialized")
        
        session_id = conversation_manager.start_conversation(
            user_id=user["user_id"],
            title=title
        )
        
        return {"session_id": session_id, "title": title, "status": "active"}
        
    except Exception as e:
        logger.error(f"Error starting conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/conversations/{session_id}", response_model=ConversationInfo)
async def get_conversation_info(
    session_id: str,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get conversation session information
    """
    try:
        if not conversation_manager:
            raise HTTPException(status_code=503, detail="Conversation manager not initialized")
        
        session_info = conversation_manager.get_session_info(session_id)
        
        if not session_info:
            raise HTTPException(status_code=404, detail="Conversation session not found")
        
        return ConversationInfo(**session_info)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting conversation info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/conversations/{session_id}/history")
async def get_conversation_history(
    session_id: str,
    limit: int = 50,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get conversation history
    """
    try:
        if not conversation_manager:
            raise HTTPException(status_code=503, detail="Conversation manager not initialized")
        
        messages = conversation_manager.memory.get_conversation_history(session_id, limit)
        
        # Format messages for response
        formatted_messages = []
        for msg in messages:
            formatted_messages.append({
                "id": msg.id,
                "content": msg.content,
                "type": msg.message_type.value,
                "timestamp": msg.timestamp.isoformat(),
                "metadata": msg.metadata
            })
        
        return {
            "session_id": session_id,
            "messages": formatted_messages,
            "total_count": len(formatted_messages)
        }
        
    except Exception as e:
        logger.error(f"Error getting conversation history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/conversations/{session_id}")
async def end_conversation(
    session_id: str,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    End a conversation session
    """
    try:
        if not conversation_manager:
            raise HTTPException(status_code=503, detail="Conversation manager not initialized")
        
        conversation_manager.end_conversation(session_id)
        
        return {"message": f"Conversation {session_id} ended successfully"}
        
    except Exception as e:
        logger.error(f"Error ending conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# System information endpoints
@app.get("/api/system/info")
async def get_system_info(user: Dict[str, Any] = Depends(get_current_user)):
    """
    Get system information and statistics
    """
    return {
        "system_name": "Nyaya - Legal Document Analysis System",
        "version": "1.0.0",
        "features": [
            "Rhetorical Role Classification",
            "Role-Aware RAG System",
            "Multi-turn Conversations",
            "Document Processing",
            "Judgment Prediction",
            "Precedent Analysis"
        ],
        "supported_formats": ["PDF", "TXT"],
        "rhetorical_roles": [
            "Facts",
            "Issue", 
            "Arguments of Petitioner",
            "Arguments of Respondent",
            "Reasoning",
            "Decision",
            "None"
        ],
        "case_types": [
            "civil",
            "criminal", 
            "constitutional",
            "commercial",
            "family",
            "tax",
            "labor",
            "property"
        ]
    }

@app.get("/api/system/stats")
async def get_system_stats(user: Dict[str, Any] = Depends(get_current_user)):
    """
    Get system usage statistics
    """
    # This would be implemented with proper tracking
    return {
        "documents_processed": 0,
        "queries_handled": 0,
        "active_conversations": 0,
        "predictions_made": 0,
        "uptime": "System just started"
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "message": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "message": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

# Main entry point
if __name__ == "__main__":
    # Configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "true").lower() == "true"
    
    # Run the application
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )
