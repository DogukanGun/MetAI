"""
MongoDB Models for Chat Sessions

Defines the MongoDB document structure and database operations for chat sessions.
"""

import time
from typing import List, Optional, Dict, Any
from datetime import datetime
from bson import ObjectId


class ChatMessageDocument:
    """MongoDB document structure for chat messages."""
    
    def __init__(self, role: str, content: str, timestamp: float):
        self.role = role
        self.content = content
        self.timestamp = timestamp
    
    def to_dict(self) -> dict:
        """Convert to dictionary for MongoDB storage."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ChatMessageDocument':
        """Create instance from MongoDB document."""
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=data["timestamp"]
        )


class ChatSessionDocument:
    """MongoDB document structure for chat sessions."""
    
    def __init__(
        self,
        chat_id: str,
        context: str,
        messages: Optional[List[ChatMessageDocument]] = None,
        created_at: Optional[float] = None,
        updated_at: Optional[float] = None,
        _id: Optional[ObjectId] = None
    ):
        self._id = _id
        self.chat_id = chat_id
        self.context = context
        self.messages = messages or []
        self.created_at = created_at or time.time()
        self.updated_at = updated_at or time.time()
    
    def to_dict(self) -> dict:
        """Convert to dictionary for MongoDB storage."""
        doc = {
            "chat_id": self.chat_id,
            "context": self.context,
            "messages": [msg.to_dict() for msg in self.messages],
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
        
        if self._id:
            doc["_id"] = self._id
            
        return doc
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ChatSessionDocument':
        """Create instance from MongoDB document."""
        messages = [
            ChatMessageDocument.from_dict(msg_data) 
            for msg_data in data.get("messages", [])
        ]
        
        return cls(
            _id=data.get("_id"),
            chat_id=data["chat_id"],
            context=data["context"],
            messages=messages,
            created_at=data["created_at"],
            updated_at=data["updated_at"]
        )
    
    def add_message(self, role: str, content: str) -> None:
        """Add a new message to the session."""
        message = ChatMessageDocument(role, content, time.time())
        self.messages.append(message)
        self.updated_at = time.time()
    
    def get_message_count(self) -> int:
        """Get total number of messages in session."""
        return len(self.messages)
    
    def get_last_message(self) -> Optional[ChatMessageDocument]:
        """Get the last message in the session."""
        return self.messages[-1] if self.messages else None
    
    def to_api_response(self) -> dict:
        """Convert to API response format."""
        return {
            "chat_id": self.chat_id,
            "context": self.context,
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp
                }
                for msg in self.messages
            ],
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }


class ChatSessionRepository:
    """Repository class for chat session database operations."""
    
    def __init__(self, collection):
        self.collection = collection
    
    async def create_session(self, chat_session: ChatSessionDocument) -> bool:
        """Create a new chat session in MongoDB."""
        try:
            result = await self.collection.insert_one(chat_session.to_dict())
            return result.inserted_id is not None
        except Exception as e:
            print(f"Error creating chat session: {e}")
            return False
    
    async def get_session_by_chat_id(self, chat_id: str) -> Optional[ChatSessionDocument]:
        """Get chat session by chat_id."""
        try:
            doc = await self.collection.find_one({"chat_id": chat_id})
            if doc:
                return ChatSessionDocument.from_dict(doc)
            return None
        except Exception as e:
            print(f"Error getting chat session: {e}")
            return None
    
    async def update_session(self, chat_session: ChatSessionDocument) -> bool:
        """Update an existing chat session."""
        try:
            result = await self.collection.update_one(
                {"chat_id": chat_session.chat_id},
                {"$set": chat_session.to_dict()}
            )
            return result.modified_count > 0
        except Exception as e:
            print(f"Error updating chat session: {e}")
            return False
    
    async def add_message_to_session(
        self, 
        chat_id: str, 
        role: str, 
        content: str
    ) -> bool:
        """Add a message to an existing session."""
        try:
            message_doc = ChatMessageDocument(role, content, time.time()).to_dict()
            
            result = await self.collection.update_one(
                {"chat_id": chat_id},
                {
                    "$push": {"messages": message_doc},
                    "$set": {"updated_at": time.time()}
                }
            )
            return result.modified_count > 0
        except Exception as e:
            print(f"Error adding message to session: {e}")
            return False
    
    async def delete_session(self, chat_id: str) -> bool:
        """Delete a chat session."""
        try:
            result = await self.collection.delete_one({"chat_id": chat_id})
            return result.deleted_count > 0
        except Exception as e:
            print(f"Error deleting chat session: {e}")
            return False
    
    async def list_sessions(
        self, 
        limit: int = 100, 
        skip: int = 0,
        sort_by: str = "updated_at",
        sort_order: int = -1
    ) -> List[ChatSessionDocument]:
        """List chat sessions with pagination."""
        try:
            cursor = self.collection.find().sort(sort_by, sort_order).skip(skip).limit(limit)
            sessions = []
            async for doc in cursor:
                sessions.append(ChatSessionDocument.from_dict(doc))
            return sessions
        except Exception as e:
            print(f"Error listing chat sessions: {e}")
            return []
    
    async def get_session_count(self) -> int:
        """Get total number of chat sessions."""
        try:
            return await self.collection.count_documents({})
        except Exception as e:
            print(f"Error counting chat sessions: {e}")
            return 0
    
    async def get_sessions_by_date_range(
        self, 
        start_time: float, 
        end_time: float
    ) -> List[ChatSessionDocument]:
        """Get sessions created within a date range."""
        try:
            cursor = self.collection.find({
                "created_at": {"$gte": start_time, "$lte": end_time}
            }).sort("created_at", -1)
            
            sessions = []
            async for doc in cursor:
                sessions.append(ChatSessionDocument.from_dict(doc))
            return sessions
        except Exception as e:
            print(f"Error getting sessions by date range: {e}")
            return []
    
    async def cleanup_old_sessions(self, older_than_days: int = 30) -> int:
        """Clean up sessions older than specified days."""
        try:
            cutoff_time = time.time() - (older_than_days * 24 * 60 * 60)
            result = await self.collection.delete_many({
                "created_at": {"$lt": cutoff_time}
            })
            return result.deleted_count
        except Exception as e:
            print(f"Error cleaning up old sessions: {e}")
            return 0