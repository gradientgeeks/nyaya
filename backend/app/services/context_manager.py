"""
Context Manager

Manages conversation sessions, history, and state for multi-turn interactions.
Integrates with LangGraph's checkpointing system.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid
from ..models.schemas import SessionContext, ChatMessage

logger = logging.getLogger(__name__)


class ContextManager:
    """
    Session and conversation state management.
    
    Key responsibilities:
    - Track active cases per session
    - Maintain conversation history
    - Store user preferences
    - Provide context for follow-up questions
    - Session lifecycle management
    """
    
    def __init__(self):
        self.sessions: Dict[str, SessionContext] = {}
        logger.info("✅ ContextManager initialized")
    
    def create_session(
        self,
        user_id: Optional[str] = None,
        initial_case_id: Optional[str] = None
    ) -> str:
        """
        Create a new session.
        
        Args:
            user_id: Optional user identifier
            initial_case_id: Optional initial case to focus on
        
        Returns:
            session_id: New session identifier
        """
        session_id = str(uuid.uuid4())
        
        self.sessions[session_id] = SessionContext(
            session_id=session_id,
            user_id=user_id,
            active_case_id=initial_case_id,
            conversation_history=[],
            metadata={}
        )
        
        logger.info(f"📝 Created session: {session_id}")
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[SessionContext]:
        """Get session context by ID."""
        return self.sessions.get(session_id)
    
    def update_session(
        self,
        session_id: str,
        active_case_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update session state.
        
        Args:
            session_id: Session identifier
            active_case_id: New active case ID (if changing)
            metadata: Additional metadata to merge
        
        Returns:
            bool: Success status
        """
        session = self.sessions.get(session_id)
        if not session:
            logger.warning(f"⚠️  Session not found: {session_id}")
            return False
        
        if active_case_id is not None:
            session.active_case_id = active_case_id
            logger.info(f"🔄 Updated active case for session {session_id}: {active_case_id}")
        
        if metadata:
            session.metadata.update(metadata)
        
        return True
    
    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        message_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add a message to conversation history.
        
        Args:
            session_id: Session identifier
            role: Message role ('user' or 'assistant')
            content: Message content
            message_type: Optional message type (text, analysis, prediction, etc.)
            metadata: Optional additional metadata
        
        Returns:
            bool: Success status
        """
        session = self.sessions.get(session_id)
        if not session:
            logger.warning(f"⚠️  Session not found: {session_id}")
            return False
        
        message = ChatMessage(
            role=role,
            content=content,
            message_type=message_type,
            metadata=metadata or {}
        )
        
        session.conversation_history.append(message)
        
        logger.info(
            f"💬 Added message to session {session_id}: "
            f"{role} ({len(content)} chars)"
        )
        
        return True
    
    def get_conversation_history(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> List[ChatMessage]:
        """
        Get conversation history for a session.
        
        Args:
            session_id: Session identifier
            limit: Optional limit on number of messages (most recent)
        
        Returns:
            List of ChatMessage objects
        """
        session = self.sessions.get(session_id)
        if not session:
            logger.warning(f"⚠️  Session not found: {session_id}")
            return []
        
        history = session.conversation_history
        
        if limit:
            history = history[-limit:]
        
        return history
    
    def get_context_for_query(
        self,
        session_id: str,
        include_history: bool = True,
        history_window: int = 5
    ) -> Dict[str, Any]:
        """
        Get contextual information for processing a query.
        
        This is used by agents to understand:
        - What case is currently being discussed
        - Recent conversation turns (for pronoun resolution)
        - User preferences/metadata
        
        Args:
            session_id: Session identifier
            include_history: Whether to include conversation history
            history_window: Number of recent messages to include
        
        Returns:
            Dict with context information
        """
        session = self.sessions.get(session_id)
        if not session:
            logger.warning(f"⚠️  Session not found: {session_id}")
            return {}
        
        context = {
            "session_id": session_id,
            "active_case_id": session.active_case_id,
            "user_id": session.user_id,
            "metadata": session.metadata
        }
        
        if include_history and session.conversation_history:
            recent_history = session.conversation_history[-history_window:]
            context["recent_history"] = [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "type": msg.message_type,
                    "timestamp": msg.timestamp.isoformat()
                }
                for msg in recent_history
            ]
        
        return context
    
    def has_active_case(self, session_id: str) -> bool:
        """Check if session has an active case."""
        session = self.sessions.get(session_id)
        return bool(session and session.active_case_id)
    
    def clear_session(self, session_id: str) -> bool:
        """
        Clear session data (logout/reset).
        
        Args:
            session_id: Session identifier
        
        Returns:
            bool: Success status
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"🗑️  Cleared session: {session_id}")
            return True
        
        logger.warning(f"⚠️  Session not found: {session_id}")
        return False
    
    def cleanup_old_sessions(self, max_age_hours: int = 24) -> int:
        """
        Clean up old inactive sessions.
        
        Args:
            max_age_hours: Maximum session age in hours
        
        Returns:
            int: Number of sessions cleaned up
        """
        current_time = datetime.now()
        sessions_to_remove = []
        
        for session_id, session in self.sessions.items():
            # Check if session has any recent activity
            if session.conversation_history:
                last_message = session.conversation_history[-1]
                age_hours = (current_time - last_message.timestamp).total_seconds() / 3600
                
                if age_hours > max_age_hours:
                    sessions_to_remove.append(session_id)
        
        # Remove old sessions
        for session_id in sessions_to_remove:
            del self.sessions[session_id]
        
        if sessions_to_remove:
            logger.info(f"🧹 Cleaned up {len(sessions_to_remove)} old sessions")
        
        return len(sessions_to_remove)
    
    def get_session_count(self) -> int:
        """Get total number of active sessions."""
        return len(self.sessions)
