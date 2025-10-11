"""
Complete Analysis Controller

Provides a comprehensive endpoint that combines:
- Video emotion recognition (all 5 models in parallel)
- Knowledge base retrieval
- AI agent analysis

Single endpoint for complete meeting analysis.
"""

import time
import tempfile
import os
import logging
import sys
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from typing import Optional

# Add app directory to path
app_dir = Path(__file__).parent.parent.parent
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

from api.services.emotion_analysis_service import EmotionAnalysisService
from api.data.config_loader import load_api_config
from modules.ai_agent import MeetingAnalysisAgent
from knowledge_base.retrieval.query_engine import QueryEngine
from knowledge_base.ingestion.vector_store_manager import VectorStoreManager


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/analyze", tags=["Complete Analysis"])

# Singleton components
_vector_store: Optional[VectorStoreManager] = None
_query_engine: Optional[QueryEngine] = None


def get_query_engine() -> Optional[QueryEngine]:
    """Get or create query engine."""
    global _vector_store, _query_engine
    try:
        if _query_engine is None:
            if _vector_store is None:
                storage_path = str(app_dir / "knowledge_base" / "storage")
                _vector_store = VectorStoreManager(
                    storage_path=storage_path,
                    embedding_model="sentence-transformers"
                )
            _query_engine = QueryEngine(_vector_store)
        return _query_engine
    except Exception as e:
        logger.warning(f"Could not initialize query engine: {e}")
        return None


@router.post(
    "/complete",
    summary="Complete video analysis with AI agent",
    description="Analyze video with all emotion recognition models + knowledge base + AI agent"
)
async def analyze_complete(
    video: UploadFile = File(..., description="Video file to analyze"),
    enable_ai_agent: bool = Form(True, description="Enable AI agent analysis"),
    knowledge_query: Optional[str] = Form(None, description="Custom knowledge base query"),
    llm_provider: str = Form("cloud", description="LLM provider: 'cloud' or 'local'")
):
    """
    Complete analysis pipeline:
    1. Extract audio, video, and text features
    2. Run all 5 emotion recognition models in parallel
    3. Query knowledge base for relevant context
    4. Generate AI agent analysis with citations
    5. Return comprehensive results
    
    Args:
        video: Video file
        enable_ai_agent: Whether to run AI agent (requires OpenAI API key)
        knowledge_query: Optional custom query for knowledge base
        
    Returns:
        Complete analysis with emotion recognition and AI insights
    """
    start_time = time.time()
    temp_path = None
    
    try:
        logger.info(f"Starting complete analysis for: {video.filename}")
        
        # Save uploaded file
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=f".{video.filename.split('.')[-1]}"
        ) as tmp_file:
            content = await video.read()
            tmp_file.write(content)
            temp_path = tmp_file.name
        
        # Load config
        config = load_api_config()
        
        # Step 1: Emotion recognition with all models
        logger.info("Running emotion recognition with all models...")
        emotion_service = EmotionAnalysisService(config)
        model_results = emotion_service.process_video_with_all_models(temp_path)
        
        # Use the first model's results for AI agent (typically Hybrid which is best)
        primary_result = model_results[0] if model_results else None
        
        if not primary_result:
            raise Exception("Emotion recognition failed for all models")
        
        # Step 2: Prepare data for AI agent
        emotion_data = {
            'overall_prediction': {
                'predicted_emotion': primary_result.overall_prediction.predicted_emotion,
                'confidence': primary_result.overall_prediction.confidence,
                'all_confidences': primary_result.overall_prediction.all_confidences.dict()
            },
            'temporal_predictions': [
                {
                    'timestamp': tp.timestamp,
                    'emotion': tp.emotion,
                    'confidence': tp.confidence,
                    'confidences': tp.all_confidences.dict()
                }
                for tp in primary_result.temporal_predictions
            ] if primary_result.temporal_predictions else [],
            'mental_health_analysis': primary_result.mental_health_analysis.dict() 
                if primary_result.mental_health_analysis else None
        }
        
        video_meta = {
            'filename': video.filename,
            'duration': primary_result.video_metadata.duration,
            'upload_date': primary_result.video_metadata.upload_date.isoformat()
        }
        
        transcription = primary_result.transcription.text if primary_result.transcription else ""
        
        # Step 3: AI Agent Analysis (if enabled)
        ai_analysis = None
        if enable_ai_agent:
            try:
                logger.info("Running AI agent analysis...")
                
                # Get query engine
                query_engine = get_query_engine()
                
                # Initialize AI agent with selected provider
                agent = MeetingAnalysisAgent(
                    query_engine=query_engine,
                    provider=llm_provider
                )
                
                # Generate context query
                if not knowledge_query:
                    emotion = emotion_data['overall_prediction']['predicted_emotion']
                    knowledge_query = f"meeting analysis video call {emotion} emotion"
                    if transcription:
                        knowledge_query += f" {transcription[:100]}"
                
                # Analyze
                ai_analysis = agent.analyze_meeting(
                    emotion_results=emotion_data,
                    video_metadata=video_meta,
                    transcription=transcription,
                    context_query=knowledge_query
                )
                
                logger.info("AI agent analysis completed")
            
            except Exception as e:
                logger.error(f"AI agent analysis failed: {e}", exc_info=True)
                ai_analysis = {
                    "error": str(e),
                    "message": "AI agent analysis failed. Results include emotion recognition only.",
                    "agent_available": False
                }
        
        # Build response
        total_time = time.time() - start_time
        
        response = {
            "success": True,
            "message": "Complete analysis finished successfully",
            "emotion_recognition": {
                "model_results": [
                    {
                        "model_name": r.model_name,
                        "fusion_strategy": r.fusion_strategy,
                        "processing_time": r.processing_time,
                        "predicted_emotion": r.overall_prediction.predicted_emotion,
                        "confidence": r.overall_prediction.confidence,
                        "all_confidences": r.overall_prediction.all_confidences.dict()
                    }
                    for r in model_results
                ],
                "primary_model": {
                    "overall_prediction": emotion_data['overall_prediction'],
                    "temporal_predictions": emotion_data['temporal_predictions'],
                    "mental_health_analysis": emotion_data['mental_health_analysis']
                },
                "video_metadata": primary_result.video_metadata.dict(),
                "transcription": {
                    "text": transcription,
                    "word_count": len(transcription.split()) if transcription else 0
                },
                "features": primary_result.features.dict()
            },
            "ai_analysis": ai_analysis,
            "total_processing_time": total_time,
            "timestamp": time.time()
        }
        
        logger.info(f"Complete analysis finished in {total_time:.2f}s")
        
        return response
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"Complete analysis failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": "AnalysisError",
                "message": f"Complete analysis failed: {str(e)}",
                "details": {"filename": video.filename}
            }
        )
    
    finally:
        # Clean up
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception as e:
                logger.warning(f"Failed to clean up temp file: {e}")


@router.get(
    "/status",
    summary="Check analysis system status",
    description="Check if all components are available"
)
async def system_status():
    """
    Check system status and component availability.
    
    Returns:
        Status of all components
    """
    import os
    
    status = {
        "emotion_recognition": True,
        "knowledge_base": False,
        "ai_agent": False,
        "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
        "components": {}
    }
    
    # Check knowledge base
    try:
        query_engine = get_query_engine()
        if query_engine:
            stats = query_engine.vector_store.get_stats()
            status["knowledge_base"] = True
            status["components"]["knowledge_base"] = {
                "available": True,
                "text_chunks": stats.get('text_chunks', 0),
                "images": stats.get('images', 0)
            }
    except Exception as e:
        status["components"]["knowledge_base"] = {
            "available": False,
            "error": str(e)
        }
    
    # Check AI agent
    try:
        from modules.ai_agent import MeetingAnalysisAgent
        agent = MeetingAnalysisAgent()
        status["ai_agent"] = bool(os.getenv("OPENAI_API_KEY"))
        status["components"]["ai_agent"] = {
            "available": bool(os.getenv("OPENAI_API_KEY")),
            "fallback_available": True
        }
    except Exception as e:
        status["components"]["ai_agent"] = {
            "available": False,
            "error": str(e)
        }
    
    return {
        "success": True,
        "status": status,
        "message": "System operational" if all([
            status["emotion_recognition"],
            status["knowledge_base"] or True,  # KB is optional
            status["ai_agent"] or True  # AI agent is optional
        ]) else "Some components unavailable"
    }
