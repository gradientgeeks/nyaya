"""
Conversation management endpoints
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from ..core.conversation_manager import ConversationManager
from .auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/conversations", tags=["conversations"])

class ConversationInfo(BaseModel):
    """Conversation session information"""
    session_id: str = Field(..., description="Session identifier")
    title: str = Field(..., description="Conversation title")
    status: str = Field(..., description="Session status")
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")
    message_count: int = Field(..., description="Number of messages")

# Global conversation manager (will be set by main.py)
conversation_manager: Optional[ConversationManager] = None

def set_conversation_manager(manager: ConversationManager):
    """Set the global conversation manager instance"""
    global conversation_manager
    conversation_manager = manager

@router.post("/start")
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

@router.get("/{session_id}", response_model=ConversationInfo)
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

@router.get("/{session_id}/history")
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

@router.delete("/{session_id}")
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