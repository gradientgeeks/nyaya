"""
Conversation Manager for Multi-turn Legal Q&A

This module manages conversation context, memory, and session state 
for multi-turn interactions with the legal RAG system.
"""

import uuid
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

from pydantic import BaseModel
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

class MessageType(Enum):
    """Types of messages in conversation"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class ConversationStatus(Enum):
    """Status of conversation session"""
    ACTIVE = "active"
    PAUSED = "paused"
    ENDED = "ended"

@dataclass
class Message:
    """Individual message in conversation"""
    id: str
    content: str
    message_type: MessageType
    timestamp: datetime
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary"""
        return {
            "id": self.id,
            "content": self.content,
            "message_type": self.message_type.value,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata or {}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create message from dictionary"""
        return cls(
            id=data["id"],
            content=data["content"],
            message_type=MessageType(data["message_type"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {})
        )

class ConversationSession(BaseModel):
    """Conversation session model"""
    session_id: str
    user_id: Optional[str] = None
    title: str = "Legal Consultation"
    status: ConversationStatus = ConversationStatus.ACTIVE
    created_at: datetime
    updated_at: datetime
    messages: List[Message] = []
    context: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}
    
    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            MessageType: lambda v: v.value,
            ConversationStatus: lambda v: v.value
        }

class ConversationMemory:
    """
    Manages conversation memory including short-term and long-term storage
    """
    
    def __init__(self, db_path: str = "conversations.db", max_short_term: int = 20):
        """
        Initialize conversation memory
        
        Args:
            db_path: Path to SQLite database for persistent storage
            max_short_term: Maximum messages to keep in short-term memory
        """
        self.db_path = db_path
        self.max_short_term = max_short_term
        self.active_sessions: Dict[str, ConversationSession] = {}
        
        # Initialize database
        self._init_database()
        
        logger.info("Conversation Memory initialized")
    
    def _init_database(self):
        """Initialize SQLite database for conversation storage"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    title TEXT,
                    status TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    context TEXT,
                    metadata TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    session_id TEXT,
                    content TEXT,
                    message_type TEXT,
                    timestamp TEXT,
                    metadata TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_session_timestamp 
                ON messages (session_id, timestamp)
            """)
    
    def create_session(self, user_id: Optional[str] = None, 
                      title: str = "Legal Consultation") -> ConversationSession:
        """
        Create a new conversation session
        
        Args:
            user_id: Optional user identifier
            title: Title for the conversation
            
        Returns:
            New ConversationSession
        """
        session_id = str(uuid.uuid4())
        now = datetime.utcnow()
        
        session = ConversationSession(
            session_id=session_id,
            user_id=user_id,
            title=title,
            created_at=now,
            updated_at=now
        )
        
        self.active_sessions[session_id] = session
        self._save_session_to_db(session)
        
        logger.info(f"Created new conversation session: {session_id}")
        return session
    
    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """
        Get conversation session by ID
        
        Args:
            session_id: Session identifier
            
        Returns:
            ConversationSession if found, None otherwise
        """
        # Check active sessions first
        if session_id in self.active_sessions:
            return self.active_sessions[session_id]
        
        # Load from database
        session = self._load_session_from_db(session_id)
        if session:
            self.active_sessions[session_id] = session
        
        return session
    
    def add_message(self, session_id: str, content: str, 
                   message_type: MessageType, metadata: Dict[str, Any] = None) -> Message:
        """
        Add message to conversation session
        
        Args:
            session_id: Session identifier
            content: Message content
            message_type: Type of message
            metadata: Additional metadata
            
        Returns:
            Created Message object
        """
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        message = Message(
            id=str(uuid.uuid4()),
            content=content,
            message_type=message_type,
            timestamp=datetime.utcnow(),
            metadata=metadata or {}
        )
        
        session.messages.append(message)
        session.updated_at = datetime.utcnow()
        
        # Manage short-term memory
        if len(session.messages) > self.max_short_term:
            # Keep recent messages in memory, archive older ones
            archived_messages = session.messages[:-self.max_short_term]
            session.messages = session.messages[-self.max_short_term:]
            
            # Save archived messages to database
            for msg in archived_messages:
                self._save_message_to_db(session_id, msg)
        
        # Update session in database
        self._save_session_to_db(session)
        self._save_message_to_db(session_id, message)
        
        return message
    
    def get_conversation_history(self, session_id: str, 
                               limit: int = 50) -> List[Message]:
        """
        Get conversation history for a session
        
        Args:
            session_id: Session identifier
            limit: Maximum number of messages to return
            
        Returns:
            List of messages in chronological order
        """
        session = self.get_session(session_id)
        if not session:
            return []
        
        # Get recent messages from memory
        recent_messages = session.messages.copy()
        
        # Get older messages from database if needed
        if len(recent_messages) < limit:
            older_messages = self._load_messages_from_db(
                session_id, 
                limit - len(recent_messages),
                exclude_ids=[msg.id for msg in recent_messages]
            )
            # Combine and sort by timestamp
            all_messages = older_messages + recent_messages
            all_messages.sort(key=lambda x: x.timestamp)
            return all_messages[-limit:]
        
        return recent_messages[-limit:]
    
    def get_context_summary(self, session_id: str) -> Dict[str, Any]:
        """
        Get contextual summary of conversation for RAG system
        
        Args:
            session_id: Session identifier
            
        Returns:
            Context summary including recent messages and extracted topics
        """
        messages = self.get_conversation_history(session_id, limit=10)
        
        if not messages:
            return {"messages": [], "topics": [], "entities": []}
        
        # Extract recent context
        recent_user_messages = [
            msg.content for msg in messages[-5:] 
            if msg.message_type == MessageType.USER
        ]
        
        recent_assistant_messages = [
            msg.content for msg in messages[-5:] 
            if msg.message_type == MessageType.ASSISTANT
        ]
        
        # Extract mentioned legal topics (simple keyword extraction)
        legal_keywords = [
            "constitutional", "fundamental rights", "article", "section",
            "petition", "appellant", "respondent", "court", "judgment",
            "precedent", "case law", "statute", "regulation", "bail",
            "habeas corpus", "mandamus", "certiorari", "appeal"
        ]
        
        mentioned_topics = []
        all_content = " ".join([msg.content.lower() for msg in messages])
        for keyword in legal_keywords:
            if keyword in all_content:
                mentioned_topics.append(keyword)
        
        return {
            "recent_user_queries": recent_user_messages,
            "recent_responses": recent_assistant_messages,
            "mentioned_topics": mentioned_topics,
            "message_count": len(messages),
            "session_duration": (messages[-1].timestamp - messages[0].timestamp).total_seconds() / 60 if len(messages) > 1 else 0
        }
    
    def update_session_context(self, session_id: str, context_update: Dict[str, Any]):
        """
        Update session context with new information
        
        Args:
            session_id: Session identifier
            context_update: Dictionary of context updates
        """
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        session.context.update(context_update)
        session.updated_at = datetime.utcnow()
        self._save_session_to_db(session)
    
    def end_session(self, session_id: str):
        """
        End a conversation session
        
        Args:
            session_id: Session identifier
        """
        session = self.get_session(session_id)
        if session:
            session.status = ConversationStatus.ENDED
            session.updated_at = datetime.utcnow()
            self._save_session_to_db(session)
            
            # Remove from active sessions
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
    
    def _save_session_to_db(self, session: ConversationSession):
        """Save session to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO sessions 
                (session_id, user_id, title, status, created_at, updated_at, context, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session.session_id,
                session.user_id,
                session.title,
                session.status.value,
                session.created_at.isoformat(),
                session.updated_at.isoformat(),
                json.dumps(session.context),
                json.dumps(session.metadata)
            ))
    
    def _save_message_to_db(self, session_id: str, message: Message):
        """Save message to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO messages 
                (id, session_id, content, message_type, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                message.id,
                session_id,
                message.content,
                message.message_type.value,
                message.timestamp.isoformat(),
                json.dumps(message.metadata or {})
            ))
    
    def _load_session_from_db(self, session_id: str) -> Optional[ConversationSession]:
        """Load session from database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT user_id, title, status, created_at, updated_at, context, metadata
                FROM sessions WHERE session_id = ?
            """, (session_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            # Load recent messages
            messages = self._load_messages_from_db(session_id, self.max_short_term)
            
            return ConversationSession(
                session_id=session_id,
                user_id=row[0],
                title=row[1],
                status=ConversationStatus(row[2]),
                created_at=datetime.fromisoformat(row[3]),
                updated_at=datetime.fromisoformat(row[4]),
                messages=messages,
                context=json.loads(row[5]) if row[5] else {},
                metadata=json.loads(row[6]) if row[6] else {}
            )
    
    def _load_messages_from_db(self, session_id: str, limit: int = 50, 
                              exclude_ids: List[str] = None) -> List[Message]:
        """Load messages from database"""
        exclude_ids = exclude_ids or []
        
        with sqlite3.connect(self.db_path) as conn:
            placeholders = ','.join(['?' for _ in exclude_ids]) if exclude_ids else "''"
            query = f"""
                SELECT id, content, message_type, timestamp, metadata
                FROM messages 
                WHERE session_id = ? AND id NOT IN ({placeholders})
                ORDER BY timestamp DESC
                LIMIT ?
            """
            
            params = [session_id] + exclude_ids + [limit]
            cursor = conn.execute(query, params)
            
            messages = []
            for row in cursor.fetchall():
                message = Message(
                    id=row[0],
                    content=row[1],
                    message_type=MessageType(row[2]),
                    timestamp=datetime.fromisoformat(row[3]),
                    metadata=json.loads(row[4]) if row[4] else {}
                )
                messages.append(message)
            
            # Reverse to get chronological order
            return list(reversed(messages))

class ConversationManager:
    """
    High-level conversation manager that coordinates memory and context
    """
    
    def __init__(self, db_path: str = "conversations.db"):
        """
        Initialize conversation manager
        
        Args:
            db_path: Path to conversation database
        """
        self.memory = ConversationMemory(db_path)
        logger.info("Conversation Manager initialized")
    
    def start_conversation(self, user_id: Optional[str] = None, 
                          title: str = "Legal Consultation") -> str:
        """
        Start a new conversation
        
        Args:
            user_id: Optional user identifier
            title: Conversation title
            
        Returns:
            Session ID
        """
        session = self.memory.create_session(user_id, title)
        
        # Add welcome message
        self.memory.add_message(
            session.session_id,
            "Hello! I'm your legal assistant. I can help you understand legal documents, case law, and provide legal information. How can I assist you today?",
            MessageType.ASSISTANT,
            {"type": "welcome"}
        )
        
        return session.session_id
    
    def add_user_message(self, session_id: str, content: str, 
                        metadata: Dict[str, Any] = None) -> Message:
        """
        Add user message to conversation
        
        Args:
            session_id: Session identifier
            content: Message content
            metadata: Additional metadata
            
        Returns:
            Created message
        """
        return self.memory.add_message(
            session_id, content, MessageType.USER, metadata
        )
    
    def add_assistant_message(self, session_id: str, content: str, 
                            metadata: Dict[str, Any] = None) -> Message:
        """
        Add assistant message to conversation
        
        Args:
            session_id: Session identifier
            content: Message content
            metadata: Additional metadata
            
        Returns:
            Created message
        """
        return self.memory.add_message(
            session_id, content, MessageType.ASSISTANT, metadata
        )
    
    def get_conversation_for_rag(self, session_id: str) -> Dict[str, Any]:
        """
        Get conversation context formatted for RAG system
        
        Args:
            session_id: Session identifier
            
        Returns:
            Formatted conversation context
        """
        context = self.memory.get_context_summary(session_id)
        messages = self.memory.get_conversation_history(session_id, limit=10)
        
        # Format for RAG system
        conversation_text = []
        for msg in messages[-6:]:  # Last 6 messages for context
            prefix = "User" if msg.message_type == MessageType.USER else "Assistant"
            conversation_text.append(f"{prefix}: {msg.content}")
        
        return {
            "conversation_history": "\n".join(conversation_text),
            "context_summary": context,
            "current_topics": context.get("mentioned_topics", []),
            "session_metadata": {
                "session_id": session_id,
                "message_count": context.get("message_count", 0),
                "duration_minutes": context.get("session_duration", 0)
            }
        }
    
    def end_conversation(self, session_id: str):
        """
        End conversation session
        
        Args:
            session_id: Session identifier
        """
        self.memory.end_session(session_id)
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session information
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session information dictionary
        """
        session = self.memory.get_session(session_id)
        if not session:
            return None
        
        return {
            "session_id": session.session_id,
            "title": session.title,
            "status": session.status.value,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
            "message_count": len(session.messages),
            "user_id": session.user_id
        }

# Example usage
if __name__ == "__main__":
    # Initialize conversation manager
    conv_manager = ConversationManager()
    
    # Start a conversation
    session_id = conv_manager.start_conversation(user_id="user123")
    print(f"Started conversation: {session_id}")
    
    # Add some messages
    conv_manager.add_user_message(
        session_id, 
        "I want to understand Article 21 of the Indian Constitution"
    )
    
    conv_manager.add_assistant_message(
        session_id, 
        "Article 21 of the Indian Constitution guarantees the right to life and personal liberty. It states that no person shall be deprived of his life or personal liberty except according to procedure established by law."
    )
    
    conv_manager.add_user_message(
        session_id, 
        "Can you explain the concept of due process in relation to Article 21?"
    )
    
    # Get conversation context for RAG
    rag_context = conv_manager.get_conversation_for_rag(session_id)
    print("RAG Context:", rag_context)
    
    # Get session info
    session_info = conv_manager.get_session_info(session_id)
    print("Session Info:", session_info)