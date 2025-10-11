"""
Chat Controller

Handles HTTP endpoints for AI chat functionality.
"""

import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from api.models.response_models import (
    StartChatRequest,
    StartChatResponse,
    SendMessageRequest,
    SendMessageResponse,
    ErrorResponse
)
from api.services.chat_service_mongo import ChatServiceMongo

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/chat", tags=["AI Chat"])

# Global chat service instance with MongoDB
chat_service = ChatServiceMongo()


@router.post(
    "/start",
    response_model=StartChatResponse,
    summary="Start a new chat session",
    description="Initialize a new chat session with context and get a chat ID"
)
async def start_chat(request: StartChatRequest) -> StartChatResponse:
    """
    Start a new chat session with the provided context.
    
    The context will be used throughout the conversation to provide
    relevant responses from the AI agent.
    
    Args:
        request: StartChatRequest with context
        
    Returns:
        StartChatResponse with chat_id
        
    Raises:
        HTTPException: If chat creation fails
    """
    try:
        logger.info("Starting new chat session")
        
        if not request.context.strip():
            raise HTTPException(
                status_code=400,
                detail={
                    "success": False,
                    "error": "ValidationError",
                    "message": "Context cannot be empty",
                    "details": {"field": "context"}
                }
            )
        
        chat_id = await chat_service.create_chat_session(request.context)
        
        logger.info(f"Successfully created chat session: {chat_id}")
        
        return StartChatResponse(
            success=True,
            chat_id=chat_id,
            message="Chat session created successfully"
        )
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    
    except Exception as e:
        logger.error(f"Error creating chat session: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": "InternalServerError",
                "message": f"Failed to create chat session: {str(e)}",
                "details": {"context_length": len(request.context)}
            }
        )


@router.post(
    "/message",
    response_model=SendMessageResponse,
    summary="Send a message to the AI agent",
    description="Send a message to an existing chat session and get AI response"
)
async def send_message(request: SendMessageRequest) -> SendMessageResponse:
    """
    Send a message to the AI agent in an existing chat session.
    
    The AI agent will use the chat context and message history to provide
    a relevant response. The conversation history is maintained throughout
    the session.
    
    Args:
        request: SendMessageRequest with chat_id and message
        
    Returns:
        SendMessageResponse with user and assistant messages
        
    Raises:
        HTTPException: If chat session not found or processing fails
    """
    try:
        logger.info(f"Processing message for chat session: {request.chat_id}")
        
        if not request.message.strip():
            raise HTTPException(
                status_code=400,
                detail={
                    "success": False,
                    "error": "ValidationError",
                    "message": "Message cannot be empty",
                    "details": {"field": "message"}
                }
            )
        
        # Check if chat session exists
        session = await chat_service.get_chat_session(request.chat_id)
        if not session:
            raise HTTPException(
                status_code=404,
                detail={
                    "success": False,
                    "error": "ChatNotFound",
                    "message": f"Chat session not found: {request.chat_id}",
                    "details": {"chat_id": request.chat_id}
                }
            )
        
        # Send message and get response
        user_message, assistant_message = await chat_service.send_message(
            request.chat_id,
            request.message
        )
        
        logger.info(f"Successfully processed message in chat session: {request.chat_id}")
        
        return SendMessageResponse(
            success=True,
            chat_id=request.chat_id,
            user_message=user_message,
            assistant_message=assistant_message,
            message="Message processed successfully"
        )
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    
    except ValueError as e:
        # Handle chat not found from service
        raise HTTPException(
            status_code=404,
            detail={
                "success": False,
                "error": "ChatNotFound",
                "message": str(e),
                "details": {"chat_id": request.chat_id}
            }
        )
    
    except Exception as e:
        logger.error(f"Error processing message: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": "InternalServerError",
                "message": f"Failed to process message: {str(e)}",
                "details": {
                    "chat_id": request.chat_id,
                    "message_length": len(request.message)
                }
            }
        )


@router.get(
    "/session/{chat_id}",
    summary="Get chat session details",
    description="Retrieve chat session information and message history"
)
async def get_chat_session(chat_id: str):
    """
    Get chat session details including message history.
    
    Args:
        chat_id: Chat session ID
        
    Returns:
        Chat session details with message history
        
    Raises:
        HTTPException: If chat session not found
    """
    try:
        session = await chat_service.get_chat_session(chat_id)
        if not session:
            raise HTTPException(
                status_code=404,
                detail={
                    "success": False,
                    "error": "ChatNotFound",
                    "message": f"Chat session not found: {chat_id}",
                    "details": {"chat_id": chat_id}
                }
            )
        
        return {
            "success": True,
            "session": session.dict(),
            "message": "Chat session retrieved successfully"
        }
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    
    except Exception as e:
        logger.error(f"Error retrieving chat session: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": "InternalServerError",
                "message": f"Failed to retrieve chat session: {str(e)}",
                "details": {"chat_id": chat_id}
            }
        )


@router.delete(
    "/session/{chat_id}",
    summary="Delete chat session",
    description="Delete a chat session and all its message history"
)
async def delete_chat_session(chat_id: str):
    """
    Delete a chat session and all its message history.
    
    Args:
        chat_id: Chat session ID
        
    Returns:
        Deletion confirmation
        
    Raises:
        HTTPException: If chat session not found
    """
    try:
        deleted = await chat_service.delete_chat_session(chat_id)
        if not deleted:
            raise HTTPException(
                status_code=404,
                detail={
                    "success": False,
                    "error": "ChatNotFound",
                    "message": f"Chat session not found: {chat_id}",
                    "details": {"chat_id": chat_id}
                }
            )
        
        return {
            "success": True,
            "message": f"Chat session deleted successfully: {chat_id}"
        }
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    
    except Exception as e:
        logger.error(f"Error deleting chat session: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": "InternalServerError",
                "message": f"Failed to delete chat session: {str(e)}",
                "details": {"chat_id": chat_id}
            }
        )


@router.get(
    "/sessions",
    summary="List all chat sessions",
    description="Get a list of all active chat sessions"
)
async def list_chat_sessions():
    """
    List all active chat sessions.
    
    Returns:
        List of all chat sessions with basic information
    """
    try:
        sessions = await chat_service.list_chat_sessions()
        
        # Return basic session info (without full message history)
        session_summaries = []
        for session in sessions:
            session_summaries.append({
                "chat_id": session.chat_id,
                "context": session.context[:100] + "..." if len(session.context) > 100 else session.context,
                "message_count": len(session.messages),
                "created_at": session.created_at,
                "updated_at": session.updated_at
            })
        
        return {
            "success": True,
            "sessions": session_summaries,
            "total_sessions": len(session_summaries),
            "message": f"Retrieved {len(session_summaries)} active chat sessions"
        }
    
    except Exception as e:
        logger.error(f"Error listing chat sessions: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": "InternalServerError",
                "message": f"Failed to list chat sessions: {str(e)}",
                "details": {}
            }
        )


@router.get(
    "/health",
    summary="Chat service health check",
    description="Check if the chat service is running and healthy"
)
async def health_check():
    """
    Health check endpoint for the chat service.
    
    Returns:
        Health status of the chat service
    """
    try:
        active_sessions = await chat_service.get_session_count()
        ai_agent_available = chat_service.agent.client is not None
        
        return {
            "status": "healthy",
            "service": "AI Chat Service",
            "version": "1.0.0",
            "active_sessions": active_sessions,
            "ai_agent_available": ai_agent_available,
            "provider": chat_service.agent.provider,
            "model": chat_service.agent.model
        }
    
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return {
            "status": "unhealthy",
            "service": "AI Chat Service",
            "error": str(e)
        }