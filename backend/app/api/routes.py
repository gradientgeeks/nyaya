"""
FastAPI Routes

API endpoints for Nyaya's role-aware legal RAG system.
"""

import logging
import uuid
import os
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import JSONResponse

from ..models.schemas import (
    QueryRequest, SearchRequest, PredictOutcomeRequest,
    QueryResponse, SessionResponse
)
from ..agents.orchestrator import MultiAgentOrchestrator
from ..core.config import Settings

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/api/v1", tags=["nyaya"])

# Global orchestrator instance (will be injected)
_orchestrator: Optional[MultiAgentOrchestrator] = None


def get_orchestrator() -> MultiAgentOrchestrator:
    """Dependency injection for orchestrator."""
    if _orchestrator is None:
        raise HTTPException(
            status_code=500,
            detail="Orchestrator not initialized"
        )
    return _orchestrator


def init_orchestrator(settings: Settings):
    """Initialize the orchestrator (called from main.py)."""
    global _orchestrator
    _orchestrator = MultiAgentOrchestrator(settings)
    logger.info("✅ Orchestrator initialized in routes")


@router.post("/sessions", response_model=SessionResponse)
async def create_session(user_id: Optional[str] = None):
    """
    Create a new conversation session.
    
    Returns:
        SessionResponse with session_id
    """
    orchestrator = get_orchestrator()
    
    session_id = orchestrator.context_manager.create_session(user_id=user_id)
    
    return SessionResponse(
        session_id=session_id,
        message="Session created successfully"
    )


@router.post("/upload", response_model=QueryResponse)
async def upload_document(
    file: UploadFile = File(...),
    session_id: str = Form(...),
    case_id: Optional[str] = Form(None),
    orchestrator: MultiAgentOrchestrator = Depends(get_orchestrator)
):
    """
    Upload and classify a legal document.
    
    Flow:
    1. Save uploaded file temporarily
    2. Process through classification agent
    3. Return classification results
    
    Args:
        file: PDF or TXT file
        session_id: Session identifier
        case_id: Optional case identifier (generated if not provided)
    
    Returns:
        QueryResponse with classification results
    """
    try:
        # Validate file type
        if not file.filename.endswith(('.pdf', '.txt')):
            raise HTTPException(
                status_code=400,
                detail="Only PDF and TXT files are supported"
            )
        
        # Generate case_id if not provided
        if not case_id:
            case_id = f"case_{uuid.uuid4().hex[:8]}"
        
        # Save file temporarily
        temp_dir = "/tmp/nyaya_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        
        file_path = os.path.join(temp_dir, f"{case_id}_{file.filename}")
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        logger.info(f"📄 Uploaded file saved: {file_path}")
        
        # Process through orchestrator
        result = await orchestrator.process_query(
            user_query=f"Classify and analyze the uploaded document: {file.filename}",
            session_id=session_id,
            file_path=file_path,
            case_id=case_id
        )
        
        # Clean up temp file (optional: keep for debugging)
        # os.remove(file_path)
        
        return QueryResponse(
            answer=result.get("answer", ""),
            intent=result.get("intent"),
            classification_result=result.get("classification_result"),
            session_id=session_id
        )
        
    except Exception as e:
        logger.error(f"❌ Upload error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query", response_model=QueryResponse)
async def query_document(
    request: QueryRequest,
    orchestrator: MultiAgentOrchestrator = Depends(get_orchestrator)
):
    """
    Ask a question about legal documents (role-aware RAG).
    
    This is the main endpoint for:
    - Role-specific questions ("What were the facts?")
    - General questions about cases
    - Follow-up questions
    
    Args:
        request: QueryRequest with query text and session_id
    
    Returns:
        QueryResponse with answer and context
    """
    try:
        result = await orchestrator.process_query(
            user_query=request.query,
            session_id=request.session_id,
            case_id=request.case_id
        )
        
        return QueryResponse(
            answer=result.get("answer", ""),
            intent=result.get("intent"),
            rag_response=result.get("rag_response"),
            session_id=request.session_id
        )
        
    except Exception as e:
        logger.error(f"❌ Query error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search", response_model=QueryResponse)
async def search_similar_cases(
    request: SearchRequest,
    orchestrator: MultiAgentOrchestrator = Depends(get_orchestrator)
):
    """
    Find similar legal cases using role-weighted similarity.
    
    Args:
        request: SearchRequest with search query
    
    Returns:
        QueryResponse with similar cases
    """
    try:
        result = await orchestrator.process_query(
            user_query=request.query,
            session_id=request.session_id,
            case_id=request.case_id  # Optional: exclude this case from results
        )
        
        return QueryResponse(
            answer=result.get("answer", ""),
            intent=result.get("intent"),
            search_results=result.get("search_results", []),
            session_id=request.session_id
        )
        
    except Exception as e:
        logger.error(f"❌ Search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict", response_model=QueryResponse)
async def predict_outcome(
    request: PredictOutcomeRequest,
    orchestrator: MultiAgentOrchestrator = Depends(get_orchestrator)
):
    """
    Predict case outcome based on similar precedents.
    
    Args:
        request: PredictOutcomeRequest with case details
    
    Returns:
        QueryResponse with prediction
    """
    try:
        # Build query from case details
        query = f"Predict outcome for: {request.case_description}"
        if request.relevant_laws:
            query += f"\nRelevant laws: {', '.join(request.relevant_laws)}"
        
        result = await orchestrator.process_query(
            user_query=query,
            session_id=request.session_id,
            case_id=request.case_id
        )
        
        return QueryResponse(
            answer=result.get("answer", ""),
            intent=result.get("intent"),
            prediction=result.get("prediction"),
            session_id=request.session_id
        )
        
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "service": "Nyaya Legal RAG API",
            "version": "1.0.0"
        }
    )


@router.get("/stats")
async def get_stats(orchestrator: MultiAgentOrchestrator = Depends(get_orchestrator)):
    """
    Get system statistics.
    
    Returns:
        Dict with Pinecone index stats and session count
    """
    try:
        pinecone_stats = orchestrator.pinecone.get_index_stats()
        session_count = orchestrator.context_manager.get_session_count()
        
        return {
            "pinecone": pinecone_stats,
            "sessions": {
                "active_count": session_count
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Stats error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
