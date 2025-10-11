"""
Emotion Recognition Controller

Handles HTTP endpoints for emotion recognition.
"""

import time
import tempfile
import os
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from api.models.response_models import MultiModelResponse, ErrorResponse
from api.services.emotion_analysis_service import EmotionAnalysisService
from api.utils.video_validator import VideoValidator
from api.data.config_loader import load_api_config


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/emotion", tags=["Emotion Recognition"])


@router.post(
    "/analyze",
    response_model=MultiModelResponse,
    summary="Analyze emotions in a video",
    description="Upload a video file and get emotion analysis from all available models in parallel"
)
async def analyze_emotions(
    video: UploadFile = File(..., description="Video file to analyze (MP4, AVI, MOV, WebM)")
) -> MultiModelResponse:
    """
    Analyze emotions in an uploaded video using all available models.
    
    The analysis runs all models in parallel and returns standardized results.
    
    Args:
        video: Uploaded video file
        
    Returns:
        MultiModelResponse with results from all models
        
    Raises:
        HTTPException: If video validation fails or processing error occurs
    """
    start_time = time.time()
    temp_path = None
    
    try:
        # Validate video file
        logger.info(f"Received video upload: {video.filename}")
        VideoValidator.validate_upload(video)
        
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=f".{video.filename.split('.')[-1]}"
        ) as tmp_file:
            content = await video.read()
            tmp_file.write(content)
            temp_path = tmp_file.name
            logger.info(f"Saved video to temporary file: {temp_path}")
        
        # Validate video content
        VideoValidator.validate_video_file(temp_path)
        
        # Load configuration
        config = load_api_config()
        
        # Process video with all models
        service = EmotionAnalysisService(config)
        results = service.process_video_with_all_models(temp_path)
        
        total_time = time.time() - start_time
        
        logger.info(f"Successfully analyzed video with all models in {total_time:.2f}s")
        
        return MultiModelResponse(
            success=True,
            message=f"Successfully analyzed video with {len(results)} models",
            results=results,
            total_processing_time=total_time
        )
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    
    except Exception as e:
        logger.error(f"Error processing video: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": "ProcessingError",
                "message": f"Failed to process video: {str(e)}",
                "details": {"filename": video.filename}
            }
        )
    
    finally:
        # Clean up temporary file
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
                logger.info(f"Cleaned up temporary file: {temp_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temp file: {e}")


@router.get(
    "/models",
    summary="List available models",
    description="Get information about all available emotion recognition models"
)
async def list_models():
    """
    List all available emotion recognition models.
    
    Returns:
        Dictionary with model information
    """
    return {
        "success": True,
        "models": [
            {
                "name": "Hybrid (Best)",
                "type": "hybrid",
                "description": "Ensemble combining RFRBoost + Deep Learning + Attention",
                "features": [
                    "Modality importance weights",
                    "Model agreement analysis",
                    "High accuracy"
                ]
            },
            {
                "name": "RFRBoost Only",
                "type": "rfrboost",
                "description": "Random Feature Representation Boosting with SWIM features",
                "features": [
                    "Gradient boosting",
                    "Robust to overfitting",
                    "Fast processing"
                ]
            },
            {
                "name": "Maelfabien Multimodal",
                "type": "maelfabien",
                "description": "Text CNN-LSTM + Audio Time-CNN + Video XCeption",
                "features": [
                    "Specialized models per modality",
                    "Weighted ensemble",
                    "Good interpretability"
                ]
            },
            {
                "name": "Emotion-LLaMA",
                "type": "emotion_llama",
                "description": "Transformer-based with emotion reasoning",
                "features": [
                    "Natural language reasoning",
                    "Emotion intensity estimation",
                    "Context-aware predictions"
                ]
            },
            {
                "name": "Simple Concatenation",
                "type": "simple",
                "description": "Baseline feature concatenation approach",
                "features": [
                    "Simple and fast",
                    "Baseline comparison",
                    "Low resource usage"
                ]
            }
        ]
    }


@router.get(
    "/health",
    summary="Health check",
    description="Check if the API is running and healthy"
)
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Health status
    """
    return {
        "status": "healthy",
        "service": "Emotion Recognition API",
        "version": "1.0.0"
    }
