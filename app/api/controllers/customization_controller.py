"""
Customization Controller

Provides endpoints for configuring analysis parameters:
- Fusion strategy selection
- Modality toggles
- Model parameters
- LLM provider selection
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Optional, List
from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, Field

# Add app directory to path
app_dir = Path(__file__).parent.parent.parent
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/customization", tags=["Customization"])


class ModalityConfig(BaseModel):
    """Modality configuration."""
    audio: bool = Field(True, description="Enable audio analysis")
    visual: bool = Field(True, description="Enable visual analysis")
    text: bool = Field(True, description="Enable text analysis")


class ModelParameters(BaseModel):
    """Model parameters configuration."""
    n_layers: int = Field(3, ge=1, le=10, description="Number of layers")
    hidden_dim: int = Field(256, ge=64, le=512, description="Hidden dimension size")
    boost_lr: float = Field(0.5, ge=0.1, le=1.0, description="Boost learning rate")


class AnalysisConfig(BaseModel):
    """Complete analysis configuration."""
    fusion_strategy: str = Field(
        "Hybrid (Best)",
        description="Fusion strategy: 'Hybrid (Best)', 'RFRBoost Only', 'Maelfabien Multimodal', 'Emotion-LLaMA', 'Simple Concatenation'"
    )
    modalities: ModalityConfig = Field(..., description="Modality configuration")
    model_parameters: ModelParameters = Field(..., description="Model parameters")
    llm_provider: str = Field(
        "cloud",
        description="LLM provider: 'cloud' (OpenAI) or 'local' (Ollama/LM Studio)"
    )


class AvailableOptions(BaseModel):
    """Available customization options."""
    fusion_strategies: List[str] = Field(..., description="Available fusion strategies")
    modalities: List[str] = Field(..., description="Available modalities")
    llm_providers: List[str] = Field(..., description="Available LLM providers")
    model_parameter_ranges: Dict[str, Dict[str, Any]] = Field(..., description="Model parameter ranges")


@router.get(
    "/options",
    summary="Get available customization options",
    description="Retrieve all available configuration options"
)
async def get_available_options():
    """
    Get all available customization options.
    
    Returns:
        Available options for fusion strategies, modalities, LLM providers, and model parameters
    """
    try:
        options = AvailableOptions(
            fusion_strategies=[
                "Hybrid (Best)",
                "RFRBoost Only",
                "Maelfabien Multimodal",
                "Emotion-LLaMA",
                "Simple Concatenation"
            ],
            modalities=["audio", "visual", "text"],
            llm_providers=["cloud", "local"],
            model_parameter_ranges={
                "n_layers": {"min": 1, "max": 10, "default": 3},
                "hidden_dim": {"min": 64, "max": 512, "step": 64, "default": 256},
                "boost_lr": {"min": 0.1, "max": 1.0, "step": 0.1, "default": 0.5}
            }
        )
        
        return {
            "success": True,
            "options": options.dict()
        }
    
    except Exception as e:
        logger.error(f"Error getting available options: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get available options: {str(e)}"
        )


@router.post(
    "/validate",
    summary="Validate analysis configuration",
    description="Validate a custom analysis configuration"
)
async def validate_config(
    config: AnalysisConfig = Body(..., description="Analysis configuration to validate")
):
    """
    Validate an analysis configuration.
    
    Args:
        config: Analysis configuration
        
    Returns:
        Validation result
    """
    try:
        errors = []
        warnings = []
        
        # Validate fusion strategy
        valid_strategies = [
            "Hybrid (Best)",
            "RFRBoost Only",
            "Maelfabien Multimodal",
            "Emotion-LLaMA",
            "Simple Concatenation"
        ]
        if config.fusion_strategy not in valid_strategies:
            errors.append(f"Invalid fusion strategy: {config.fusion_strategy}")
        
        # Validate at least one modality is enabled
        if not any([config.modalities.audio, config.modalities.visual, config.modalities.text]):
            errors.append("At least one modality must be enabled")
        
        # Validate LLM provider
        if config.llm_provider not in ["cloud", "local"]:
            errors.append(f"Invalid LLM provider: {config.llm_provider}")
        
        # Warnings
        if not config.modalities.audio:
            warnings.append("Audio analysis disabled - may reduce accuracy")
        if not config.modalities.visual:
            warnings.append("Visual analysis disabled - may reduce accuracy")
        if not config.modalities.text:
            warnings.append("Text analysis disabled - may reduce accuracy")
        
        return {
            "success": len(errors) == 0,
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "config": config.dict()
        }
    
    except Exception as e:
        logger.error(f"Error validating config: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to validate config: {str(e)}"
        )


@router.post(
    "/default",
    summary="Get default analysis configuration",
    description="Get the default recommended configuration"
)
async def get_default_config():
    """
    Get the default analysis configuration.
    
    Returns:
        Default configuration
    """
    try:
        default_config = AnalysisConfig(
            fusion_strategy="Hybrid (Best)",
            modalities=ModalityConfig(audio=True, visual=True, text=True),
            model_parameters=ModelParameters(n_layers=3, hidden_dim=256, boost_lr=0.5),
            llm_provider="cloud"
        )
        
        return {
            "success": True,
            "config": default_config.dict(),
            "description": "Default recommended configuration for best accuracy"
        }
    
    except Exception as e:
        logger.error(f"Error getting default config: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get default config: {str(e)}"
        )

