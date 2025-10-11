"""
Chat Service

Manages chat sessions, message history, and integrates with AI agent.
"""

import time
import uuid
import logging
from typing import Dict, List, Optional
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from modules.ai_agent import MeetingAnalysisAgent
from api.models.response_models import ChatSession, ChatMessage

logger = logging.getLogger(__name__)


class ChatService:
    """
    Service for managing chat sessions and AI agent interactions.
    """
    
    def __init__(self, query_engine=None):
        """
        Initialize the chat service.
        
        Args:
            query_engine: QueryEngine instance for knowledge base access
        """
        self.query_engine = query_engine
        self.sessions: Dict[str, ChatSession] = {}
        self.agent = MeetingAnalysisAgent(query_engine=query_engine)
        logger.info("Chat service initialized")
    
    def create_chat_session(self, context: str) -> str:
        """
        Create a new chat session with the given context.
        
        Args:
            context: Initial context for the chat session
            
        Returns:
            chat_id: Unique identifier for the chat session
        """
        chat_id = str(uuid.uuid4())
        timestamp = time.time()
        
        session = ChatSession(
            chat_id=chat_id,
            context=context,
            messages=[],
            created_at=timestamp,
            updated_at=timestamp
        )
        
        self.sessions[chat_id] = session
        logger.info(f"Created new chat session: {chat_id}")
        
        return chat_id
    
    def get_chat_session(self, chat_id: str) -> Optional[ChatSession]:
        """
        Get a chat session by ID.
        
        Args:
            chat_id: Chat session ID
            
        Returns:
            ChatSession if found, None otherwise
        """
        return self.sessions.get(chat_id)
    
    def send_message(self, chat_id: str, user_message: str) -> tuple[ChatMessage, ChatMessage]:
        """
        Send a message to the AI agent and get a response.
        
        Args:
            chat_id: Chat session ID
            user_message: User's message content
            
        Returns:
            Tuple of (user_message, assistant_message)
            
        Raises:
            ValueError: If chat session not found
        """
        session = self.sessions.get(chat_id)
        if not session:
            raise ValueError(f"Chat session not found: {chat_id}")
        
        timestamp = time.time()
        
        # Create user message
        user_msg = ChatMessage(
            role="user",
            content=user_message,
            timestamp=timestamp
        )
        
        # Generate AI response
        try:
            ai_response = self._generate_ai_response(session, user_message)
        except Exception as e:
            logger.error(f"Failed to generate AI response: {e}")
            ai_response = f"I apologize, but I encountered an error while processing your message: {str(e)}"
        
        # Create assistant message
        assistant_msg = ChatMessage(
            role="assistant",
            content=ai_response,
            timestamp=time.time()
        )
        
        # Add messages to session history
        session.messages.append(user_msg)
        session.messages.append(assistant_msg)
        session.updated_at = time.time()
        
        logger.info(f"Processed message in chat session: {chat_id}")
        
        return user_msg, assistant_msg
    
    def _generate_ai_response(self, session: ChatSession, user_message: str) -> str:
        """
        Generate AI response using the agent with context and history.
        
        Args:
            session: Chat session
            user_message: User's message
            
        Returns:
            AI response text
        """
        # Build conversation history for context
        conversation_history = []
        for msg in session.messages:
            conversation_history.append(f"{msg.role}: {msg.content}")
        
        # Create a prompt that includes context, history, and current message
        prompt_parts = [
            "# Context",
            session.context,
            "",
            "# Conversation History"
        ]
        
        if conversation_history:
            prompt_parts.extend(conversation_history)
        else:
            prompt_parts.append("(No previous conversation)")
        
        prompt_parts.extend([
            "",
            "# Current User Message",
            user_message,
            "",
            "# Instructions",
            "You are an AI assistant helping with meeting analysis and emotion recognition.",
            "Based on the provided context and conversation history, respond to the user's message.",
            "If the context mentions emotion analysis results or meeting data, reference that information.",
            "Be helpful, professional, and provide actionable insights when possible.",
            "If you can retrieve relevant information from the knowledge base, do so.",
            "",
            "Please provide a helpful response to the user's message."
        ])
        
        full_prompt = "\n".join(prompt_parts)
        
        # Use the AI agent to generate response
        if self.agent.client:
            try:
                # Use context as system prompt if available, otherwise use default
                if session.context.strip():
                    system_content = f"""You are an AI assistant helping with the following specific context:

{session.context}

Based on this context, respond to user questions and provide relevant insights. 
Be helpful, professional, and reference the context when appropriate."""
                else:
                    system_content = """You are an expert AI assistant specializing in meeting analysis and emotion recognition. 
You help users understand emotion recognition results, provide insights about team dynamics, and answer questions 
about meetings. You have access to a knowledge base and can provide evidence-based recommendations."""
                
                response = self.agent.client.chat.completions.create(
                    model=self.agent.model,
                    messages=[
                        {
                            "role": "system",
                            "content": system_content
                        },
                        {
                            "role": "user",
                            "content": full_prompt
                        }
                    ],
                    temperature=self.agent.temperature,
                    max_tokens=self.agent.max_tokens
                )
                
                return response.choices[0].message.content
                
            except Exception as e:
                logger.error(f"OpenAI API call failed: {e}")
                return self._fallback_response(user_message, session.context)
        else:
            return self._fallback_response(user_message, session.context)
    
    def _fallback_response(self, user_message: str, context: str) -> str:
        """
        Generate a fallback response when AI agent is not available.
        
        Args:
            user_message: User's message
            context: Session context
            
        Returns:
            Fallback response text
        """
        logger.info("Using fallback response generation")
        
        # Simple rule-based responses
        message_lower = user_message.lower()
        
        if any(word in message_lower for word in ['hello', 'hi', 'hey']):
            return f"Hello! I'm here to help you with your analysis. Based on your context: {context[:100]}... How can I assist you today?"
        
        elif any(word in message_lower for word in ['emotion', 'feeling', 'mood']):
            return "I can help you understand emotion recognition results and their implications for team dynamics. What specific emotions or patterns would you like to discuss?"
        
        elif any(word in message_lower for word in ['recommendation', 'advice', 'suggest']):
            return "Based on the analysis context, I can provide recommendations for improving team dynamics and meeting effectiveness. What specific area would you like recommendations for?"
        
        elif any(word in message_lower for word in ['help', 'support']):
            return "I'm here to help you understand your meeting analysis results and provide insights. You can ask me about emotions detected, team dynamics, recommendations, or any specific patterns you've noticed."
        
        else:
            return f"Thank you for your message. I understand you're asking about: '{user_message}'. Based on your analysis context, I can help provide insights. However, my AI capabilities are currently limited. Could you rephrase your question or ask about specific emotion patterns you'd like to understand?"
    
    def get_message_history(self, chat_id: str) -> List[ChatMessage]:
        """
        Get message history for a chat session.
        
        Args:
            chat_id: Chat session ID
            
        Returns:
            List of chat messages
        """
        session = self.sessions.get(chat_id)
        if not session:
            return []
        
        return session.messages
    
    def delete_chat_session(self, chat_id: str) -> bool:
        """
        Delete a chat session.
        
        Args:
            chat_id: Chat session ID
            
        Returns:
            True if deleted, False if not found
        """
        if chat_id in self.sessions:
            del self.sessions[chat_id]
            logger.info(f"Deleted chat session: {chat_id}")
            return True
        
        return False
    
    def list_chat_sessions(self) -> List[ChatSession]:
        """
        List all active chat sessions.
        
        Returns:
            List of chat sessions
        """
        return list(self.sessions.values())