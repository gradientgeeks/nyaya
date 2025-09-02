"""
Query endpoints for legal document analysis
"""

import logging
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from ..core.agent_orchestrator import AgentOrchestrator
from .auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["queries"])

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

# Global orchestrator (will be set by main.py)
orchestrator: Optional[AgentOrchestrator] = None

def set_orchestrator(orch: AgentOrchestrator):
    """Set the global orchestrator instance"""
    global orchestrator
    orchestrator = orch

@router.post("/query", response_model=QueryResponse)
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