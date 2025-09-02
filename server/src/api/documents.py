"""
Document management endpoints
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path
from fastapi import APIRouter, HTTPException, UploadFile, File, Depends, BackgroundTasks
from pydantic import BaseModel, Field

from ..core.agent_orchestrator import AgentOrchestrator
from .auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/documents", tags=["documents"])

class DocumentUploadResponse(BaseModel):
    """Response model for document upload"""
    success: bool = Field(..., description="Upload success status")
    document_id: Optional[str] = Field(None, description="Document identifier")
    filename: str = Field(..., description="Uploaded filename")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Extracted metadata")
    summary: Optional[str] = Field(None, description="Document summary")
    message: str = Field(..., description="Status message")

# Global orchestrator (will be set by main.py)
orchestrator: Optional[AgentOrchestrator] = None

def set_orchestrator(orch: AgentOrchestrator):
    """Set the global orchestrator instance"""
    global orchestrator
    orchestrator = orch

@router.post("/upload", response_model=DocumentUploadResponse)
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