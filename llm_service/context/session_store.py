"""
session_store.py - Context Component for MCP Architecture

This module represents the Context component in a Model-Context-Protocol architecture.
It handles all conversation state and persistence, including:
1. Database models (SessionDB, MessageDB)
2. Session CRUD operations
3. Message history management
4. Formatted conversation history retrieval

TODO: Implement database migration system
TODO: Add context pruning for long conversations
TODO: Implement backup/restore functionality for the database
TODO: Add context search/filter capabilities

The ContextComponent is designed to be independent of any specific model or protocol.
"""

import os
import traceback
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Union

from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, Text
from sqlalchemy import text, Index
from sqlalchemy.orm import sessionmaker, declarative_base, relationship, Session as OrmSession
from sqlalchemy.pool import StaticPool
from sqlalchemy.engine.url import make_url
from typing import Any

# --- Database models ---

Base = declarative_base()

class SessionDB(Base):
    """Database model for a conversation session."""
    __tablename__ = "sessions"
    id = Column(String, primary_key=True, index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    last_active = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    meta = Column(Text, nullable=True)  # Renamed from 'metadata' to avoid conflicts
    messages = relationship("MessageDB", back_populates="session", cascade="all, delete-orphan")

class MessageDB(Base):
    """Database model for a message within a conversation session."""
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, ForeignKey("sessions.id"))
    role = Column(String)
    content = Column(Text)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    session = relationship("SessionDB", back_populates="messages")

    def to_dict(self) -> Dict[str, Any]:
        """Convert a message to a dictionary."""
        return {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "created_at": self.created_at
        }


# --- ContextComponent class ---

class ContextComponent:
    """Context component for MCP architecture handling conversation state."""
    
    def __init__(self, database_url: Optional[str] = None):
        """Initialize the context component with a database connection."""
        if database_url is None:
            database_url = os.environ.get("MLC_SQLITE_URL", "sqlite:///mlc_sessions.db")

        # Determine SQLite in-memory vs file-backed
        url = make_url(database_url)
        is_sqlite = url.get_backend_name() == "sqlite"
        is_memory = is_sqlite and (url.database in (None, "", ":memory:"))

        # Configure engine kwargs: future flag, thread safety, and static pool for memory
        engine_kwargs: dict[str, Any] = {"future": True}
        if is_sqlite:
            engine_kwargs["connect_args"] = {"check_same_thread": False}
            if is_memory:
                engine_kwargs["poolclass"] = StaticPool
        self.engine = create_engine(database_url, **engine_kwargs)

        # 1) Create tables before any PRAGMAs
        Base.metadata.create_all(self.engine)

        # 2) Enable WAL only for file-backed SQLite
        if is_sqlite and not is_memory:
            try:
                with self.engine.connect() as conn:
                    conn.exec_driver_sql("PRAGMA journal_mode=WAL")
            except Exception:
                pass

        # Configure sessionmaker: no autoflush/autocommit, no expire on commit, future
        self.SessionLocal = sessionmaker(
            bind=self.engine,
            autoflush=False,
            autocommit=False,
            expire_on_commit=False,
            future=True,
        )

        # Create index for (session_id, id) for faster lookups
        Index("ix_messages_session_id_id", MessageDB.session_id, MessageDB.id)
    
    def get_db_session(self) -> OrmSession:
        """Get a database session. Remember to close it when done."""
        return self.SessionLocal()
    
    # --- Session management methods ---
    
    def create_session(self) -> Dict[str, Any]:
        """Create a new conversation session."""
        session_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        
        db = self.get_db_session()
        try:
            session = SessionDB(id=session_id, created_at=now, last_active=now)
            db.add(session)
            db.commit()
            db.refresh(session)
            
            return {
                "session_id": session.id,
                "created_at": session.created_at
            }
        finally:
            db.close()
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get a session by ID."""
        db = self.get_db_session()
        try:
            session = db.query(SessionDB).filter(SessionDB.id == session_id).first()
            if not session:
                return None
            
            return {
                "session_id": session.id,
                "created_at": session.created_at,
                "last_active": session.last_active,
                "meta": session.meta
            }
        finally:
            db.close()
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all available sessions."""
        db = self.get_db_session()
        try:
            sessions = db.query(SessionDB).all()
            return [
                {
                    "session_id": s.id,
                    "created_at": s.created_at,
                    "last_active": s.last_active
                }
                for s in sessions
            ]
        finally:
            db.close()
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session by ID."""
        db = self.get_db_session()
        try:
            session = db.query(SessionDB).filter(SessionDB.id == session_id).first()
            if not session:
                return False
            
            db.delete(session)
            db.commit()
            return True
        finally:
            db.close()
    
    def delete_all_sessions(self) -> int:
        """Delete all sessions. Returns the number of deleted sessions."""
        db = self.get_db_session()
        try:
            count = db.query(SessionDB).delete()
            db.commit()
            return count
        finally:
            db.close()
    
    def update_session_activity(self, session_id: str) -> bool:
        """Update the last_active timestamp of a session."""
        db = self.get_db_session()
        try:
            session = db.query(SessionDB).filter(SessionDB.id == session_id).first()
            if not session:
                return False
            
            setattr(session, 'last_active', datetime.now(timezone.utc))
            db.commit()
            return True
        finally:
            db.close()
    
    def update_session_meta(self, session_id: str, meta: str) -> bool:
        """Update the metadata of a session."""
        db = self.get_db_session()
        try:
            session = db.query(SessionDB).filter(SessionDB.id == session_id).first()
            if not session:
                return False
            
            setattr(session, 'meta', meta)
            db.commit()
            return True
        finally:
            db.close()
    
    # --- Message management methods ---
    
    def add_message(self, session_id: str, role: str, content: str) -> Optional[Dict[str, Any]]:
        """Add a message to a session."""
        db = self.get_db_session()
        try:
            # Check if session exists
            session = db.query(SessionDB).filter(SessionDB.id == session_id).first()
            if not session:
                return None
            
            # Create and add message
            message = MessageDB(
                session_id=session_id,
                role=role,
                content=content,
                created_at=datetime.now(timezone.utc)
            )
            db.add(message)
            
            # Update session's last_active timestamp
            setattr(session, 'last_active', message.created_at)
            
            db.commit()
            db.refresh(message)
            
            return message.to_dict()
        finally:
            db.close()
    
    def get_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all messages for a session."""
        db = self.get_db_session()
        try:
            # Check if session exists
            session = db.query(SessionDB).filter(SessionDB.id == session_id).first()
            if not session:
                return []
            
            # Get all messages in order
            messages = (
                db.query(MessageDB)
                .filter(MessageDB.session_id == session_id)
                .order_by(MessageDB.id)
                .all()
            )
            
            return [message.to_dict() for message in messages]
        finally:
            db.close()
    
    def get_conversation_history(self, session_id: str, format: str = "chat") -> List[Dict[str, Any]]:
        """
        Get conversation history formatted for various uses.
        
        Args:
            session_id: The ID of the session
            format: The format to return:
                   - 'chat': Format for chat models (with role/content)
                   - 'full': Full message details including ID and timestamps
        
        Returns:
            A list of messages in the requested format
        """
        db = self.get_db_session()
        try:
            # Check if session exists
            session = db.query(SessionDB).filter(SessionDB.id == session_id).first()
            if not session:
                return []
            
            # Get all messages in order
            messages = (
                db.query(MessageDB)
                .filter(MessageDB.session_id == session_id)
                .order_by(MessageDB.id)
                .all()
            )
            
            if format == "chat":
                # Simple chat format for LLM input
                return [
                    {"role": m.role, "content": m.content}
                    for m in messages
                ]
            else:
                # Full format with all details
                return [message.to_dict() for message in messages]
        finally:
            db.close()
    
    def add_user_message_and_get_history(self, session_id: str, content: str) -> List[Dict[str, Any]]:
        """
        Add a user message and return the updated conversation history in chat format.
        Convenience method for common pattern in chat applications.
        """
        self.add_message(session_id, "user", content)
        return self.get_conversation_history(session_id, format="chat")
    
    def add_assistant_message(self, session_id: str, content: str) -> Optional[Dict[str, Any]]:
        """Add an assistant message to the conversation."""
        return self.add_message(session_id, "assistant", content)
    
    def add_tool_message(self, session_id: str, tool_call_id: str, name: str, content: str) -> Optional[Dict[str, Any]]:
        """Add a tool message to the conversation with tool-specific attributes."""
        db = self.get_db_session()
        try:
            # Check if session exists
            session = db.query(SessionDB).filter(SessionDB.id == session_id).first()
            if not session:
                return None
            
            # Tool messages use a specific format
            tool_content = {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": name,
                "content": content
            }
            
            # Store as JSON string
            import json
            serialized_content = json.dumps(tool_content)
            
            message = MessageDB(
                session_id=session_id,
                role="tool",
                content=serialized_content,
                created_at=datetime.now(timezone.utc)
            )
            db.add(message)
            
            # Update session's last_active timestamp
            setattr(session, 'last_active', message.created_at)
            
            db.commit()
            db.refresh(message)
            
        finally:
            db.close()

    def add_assistant_toolcall(self, history: list[dict], call_id: str, name: str, args: dict):
        # DEBUG: dump stack whenever a tool‐call is recorded
        print(f"DEBUG add_assistant_toolcall -> id={call_id} name={name} args={args}")
        traceback.print_stack(limit=5)
    
    def add_tool_result(self, history: list[dict], call_id: str, name: str, result: Any):
        # DEBUG: dump stack whenever a tool result is recorded
        print(f"DEBUG add_tool_result      -> id={call_id} name={name}")
        traceback.print_stack(limit=5)