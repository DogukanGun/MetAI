"""
Emotion Recognition API

FastAPI application for multimodal emotion recognition.
"""

import logging
import sys
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from contextlib import asynccontextmanager

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.controllers.emotion_controller import router as emotion_router
from api.controllers.knowledge_controller import router as knowledge_router
from api.controllers.complete_analysis_controller import router as complete_router
from api.controllers.chat_controller import router as chat_router


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events for the application."""
    # Startup
    logger.info("Starting Emotion Recognition API")
    yield
    # Shutdown
    logger.info("Shutting down Emotion Recognition API")


# Initialize FastAPI app
app = FastAPI(
    title="Emotion Recognition API",
    description="""
    ## Multimodal Emotion Recognition & Knowledge Base API
    
    This API provides advanced emotion recognition from video files using multiple
    state-of-the-art models running in parallel, plus a multimodal knowledge base
    for document management and intelligent retrieval.
    
    ### Emotion Recognition Features:
    - **5 Different Models**: Hybrid, RFRBoost, Maelfabien, Emotion-LLaMA, Simple
    - **Parallel Processing**: All models run simultaneously for fast results
    - **Multimodal Analysis**: Audio, Visual, and Text modalities
    - **Temporal Analysis**: Emotion tracking over time
    - **Mental Health Scoring**: AI-based mental health assessment
    - **LLM-Friendly Output**: Structured JSON responses for easy integration
    
    ### Knowledge Base Features:
    - **Document Ingestion**: PDF, TXT, DOCX, Images
    - **Multimodal RAG**: Text and image retrieval using CLIP
    - **Vector Storage**: FAISS-based similarity search
    - **Separate Pipelines**: Independent document processing and querying
    
    ### Emotion Recognition Models:
    1. **Hybrid (Best)**: Ensemble combining RFRBoost + Deep Learning + Attention
    2. **RFRBoost Only**: Random Feature Representation Boosting
    3. **Maelfabien Multimodal**: Specialized models per modality
    4. **Emotion-LLaMA**: Transformer-based with reasoning
    5. **Simple Concatenation**: Baseline approach
    
    ### Supported Emotions:
    - Neutral
    - Happy
    - Sad
    - Angry
    - Fear
    - Disgust
    - Surprise
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors."""
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error": "ValidationError",
            "message": "Request validation failed",
            "details": exc.errors()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "InternalServerError",
            "message": "An unexpected error occurred",
            "details": {"error": str(exc)}
        }
    )


# Include routers
app.include_router(emotion_router)
app.include_router(knowledge_router)
app.include_router(complete_router)
app.include_router(chat_router)


# Root endpoint
@app.get(
    "/",
    tags=["Root"],
    summary="API Root",
    description="Get API information"
)
async def root():
    """
    Root endpoint providing API information.
    
    Returns:
        API information and available endpoints
    """
    return {
        "service": "Emotion Recognition & Knowledge Base API",
        "version": "1.0.0",
        "description": "Multimodal emotion recognition + RAG knowledge base + AI agent analysis",
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc",
            "complete_analysis": "/api/v1/analyze/complete",
            "emotion_analyze": "/api/v1/emotion/analyze",
            "emotion_models": "/api/v1/emotion/models",
            "emotion_health": "/api/v1/emotion/health",
            "knowledge_upload": "/api/v1/knowledge/documents",
            "knowledge_query": "/api/v1/knowledge/query",
            "knowledge_stats": "/api/v1/knowledge/stats",
            "system_status": "/api/v1/analyze/status",
            "chat_start": "/api/v1/chat/start",
            "chat_message": "/api/v1/chat/message",
            "chat_sessions": "/api/v1/chat/sessions"
        },
        "features": [
            "5 different emotion recognition models running in parallel",
            "AI Agent with GPT-4 integration",
            "Knowledge base with RAG (Retrieval-Augmented Generation)",
            "Temporal emotion analysis with mental health scoring",
            "Multimodal fusion (Audio + Visual + Text)",
            "Citation-based AI analysis",
            "LLM-friendly JSON responses",
            "Interactive AI chat with context management"
        ],
        "recommended_workflow": [
            "1. Upload documents to knowledge base: POST /api/v1/knowledge/documents",
            "2. Run complete analysis: POST /api/v1/analyze/complete",
            "3. Start chat session with results: POST /api/v1/chat/start",
            "4. Interact with AI agent: POST /api/v1/chat/message"
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
