"""
Document Query endpoints for simultaneous file upload and questioning
"""

import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from fastapi import APIRouter, HTTPException, UploadFile, File, Depends, Form
from pydantic import BaseModel, Field

from ..core.agent_orchestrator import AgentOrchestrator
from ..core.conversation_manager import ConversationManager
from .auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/document-query", tags=["document-query"])

class DocumentQueryRequest(BaseModel):
    """Request model for document upload with immediate query"""
    query: str = Field(..., description="Question about the uploaded document")
    session_id: Optional[str] = Field(None, description="Conversation session ID")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    role_filter: Optional[List[str]] = Field(None, description="Filter by specific rhetorical roles")

class DocumentQueryResponse(BaseModel):
    """Response model for document upload with query"""
    success: bool = Field(..., description="Upload and query success status")
    document_id: str = Field(..., description="Document identifier")
    filename: str = Field(..., description="Uploaded filename")
    session_id: str = Field(..., description="Conversation session ID")
    answer: str = Field(..., description="Answer to the query")
    document_metadata: Optional[Dict[str, Any]] = Field(None, description="Extracted document metadata")
    sources: Optional[List[Dict[str, Any]]] = Field(None, description="Source segments used for answer")
    classification: Optional[Dict[str, Any]] = Field(None, description="Query classification")
    tools_used: Optional[List[str]] = Field(None, description="Tools used for processing")
    confidence: Optional[float] = Field(None, description="Response confidence score")

# Global components (will be set by main.py)
orchestrator: Optional[AgentOrchestrator] = None
conversation_manager: Optional[ConversationManager] = None

def set_components(orch: AgentOrchestrator, conv_manager: ConversationManager):
    """Set the global component instances"""
    global orchestrator, conversation_manager
    orchestrator = orch
    conversation_manager = conv_manager

@router.post("/upload-and-ask", response_model=DocumentQueryResponse)
async def upload_document_and_ask(
    file: UploadFile = File(...),
    query: str = Form(...),
    session_id: Optional[str] = Form(None),
    role_filter: Optional[str] = Form(None),  # JSON string of role list
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Upload a document and immediately ask a question about it
    """
    try:
        if not orchestrator or not conversation_manager:
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
        
        # Parse role filter if provided
        parsed_role_filter = None
        if role_filter:
            try:
                import json
                parsed_role_filter = json.loads(role_filter)
            except json.JSONDecodeError:
                logger.warning(f"Invalid role_filter JSON: {role_filter}")
        
        # Start or use existing conversation session
        if not session_id:
            session_id = conversation_manager.start_conversation(
                user_id=user.get("user_id"),
                title=f"Analysis of {file.filename}"
            )
        
        # Step 1: Upload and process document
        logger.info(f"Processing document: {file.filename}")
        upload_result = orchestrator.upload_document(
            file_content=content,
            filename=file.filename,
            session_id=session_id
        )
        
        if not upload_result["success"]:
            raise HTTPException(status_code=400, detail=upload_result["message"])
        
        # Step 2: Add document context to conversation
        conversation_manager.add_user_message(
            session_id=session_id,
            content=f"[DOCUMENT_UPLOADED] {file.filename}",
            metadata={
                "document_id": upload_result["document_id"],
                "filename": file.filename,
                "document_metadata": upload_result.get("metadata")
            }
        )
        
        # Step 3: Process the query with document context
        logger.info(f"Processing query: {query}")
        query_context = {
            "document_id": upload_result["document_id"],
            "filename": file.filename,
            "document_metadata": upload_result.get("metadata"),
            "role_filter": parsed_role_filter,
            "immediate_document_query": True
        }
        
        query_result = orchestrator.process_query(
            query=query,
            session_id=session_id,
            context=query_context
        )
        
        # Step 4: Add query and response to conversation
        conversation_manager.add_user_message(
            session_id=session_id,
            content=query,
            metadata={"query_type": "document_query"}
        )
        
        conversation_manager.add_assistant_message(
            session_id=session_id,
            content=query_result["answer"],
            metadata={
                "sources": query_result.get("sources"),
                "tools_used": query_result.get("tools_used"),
                "confidence": query_result.get("confidence")
            }
        )
        
        return DocumentQueryResponse(
            success=True,
            document_id=upload_result["document_id"],
            filename=file.filename,
            session_id=session_id,
            answer=query_result["answer"],
            document_metadata=upload_result.get("metadata"),
            sources=query_result.get("sources"),
            classification=query_result.get("classification"),
            tools_used=query_result.get("tools_used", []),
            confidence=query_result.get("confidence")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in document upload and query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ask-followup")
async def ask_followup_question(
    query: str = Form(...),
    session_id: str = Form(...),
    role_filter: Optional[str] = Form(None),  # JSON string of role list
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Ask a follow-up question in an existing document conversation
    """
    try:
        if not orchestrator or not conversation_manager:
            raise HTTPException(status_code=503, detail="System not initialized")
        
        # Parse role filter if provided
        parsed_role_filter = None
        if role_filter:
            try:
                import json
                parsed_role_filter = json.loads(role_filter)
            except json.JSONDecodeError:
                logger.warning(f"Invalid role_filter JSON: {role_filter}")
        
        # Get conversation context (includes document information)
        conversation_context = conversation_manager.get_conversation_for_rag(session_id)
        
        if not conversation_context:
            raise HTTPException(status_code=404, detail="Conversation session not found")
        
        # Extract document context from conversation history
        document_context = {}
        for message in conversation_context.get("messages", []):
            if message.get("metadata", {}).get("document_id"):
                document_context = {
                    "document_id": message["metadata"]["document_id"],
                    "filename": message["metadata"]["filename"],
                    "document_metadata": message["metadata"].get("document_metadata"),
                    "role_filter": parsed_role_filter,
                    "followup_query": True
                }
                break
        
        # Process the follow-up query
        query_result = orchestrator.process_query(
            query=query,
            session_id=session_id,
            context=document_context
        )
        
        # Add to conversation history
        conversation_manager.add_user_message(
            session_id=session_id,
            content=query,
            metadata={"query_type": "followup_query"}
        )
        
        conversation_manager.add_assistant_message(
            session_id=session_id,
            content=query_result["answer"],
            metadata={
                "sources": query_result.get("sources"),
                "tools_used": query_result.get("tools_used"),
                "confidence": query_result.get("confidence")
            }
        )
        
        return {
            "success": True,
            "session_id": session_id,
            "answer": query_result["answer"],
            "sources": query_result.get("sources"),
            "classification": query_result.get("classification"),
            "tools_used": query_result.get("tools_used", []),
            "confidence": query_result.get("confidence")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in follow-up query: {e}")
        raise HTTPException(status_code=500, detail=str(e))