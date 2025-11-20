"""
Visualization Controller

Provides endpoints for generating graphs, charts, and visualizations
from emotion analysis results.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, Field

# Add app directory to path
app_dir = Path(__file__).parent.parent.parent
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/visualization", tags=["Visualization"])


class ChartDataPoint(BaseModel):
    """Single data point for charts."""
    label: str = Field(..., description="Label for the data point")
    value: float = Field(..., description="Value for the data point")


class BarChartData(BaseModel):
    """Bar chart data structure."""
    title: str = Field(..., description="Chart title")
    x_axis: str = Field(..., description="X-axis label")
    y_axis: str = Field(..., description="Y-axis label")
    data: List[ChartDataPoint] = Field(..., description="Chart data points")


class LineChartData(BaseModel):
    """Line chart data structure."""
    title: str = Field(..., description="Chart title")
    x_axis: str = Field(..., description="X-axis label")
    y_axis: str = Field(..., description="Y-axis label")
    series: List[Dict[str, Any]] = Field(..., description="Chart series data")


class PlotlyChartData(BaseModel):
    """Plotly chart data structure (JSON-serializable)."""
    data: List[Dict[str, Any]] = Field(..., description="Plotly trace data")
    layout: Dict[str, Any] = Field(..., description="Plotly layout configuration")
    config: Optional[Dict[str, Any]] = Field(None, description="Plotly config options")


@router.post(
    "/confidence-distribution",
    summary="Generate confidence distribution bar chart",
    description="Create a bar chart showing emotion confidence distribution"
)
async def get_confidence_chart(
    all_confidences: Dict[str, float] = Body(..., description="Confidence scores for all emotions")
):
    """
    Generate bar chart data for emotion confidence distribution.
    
    Args:
        all_confidences: Dictionary mapping emotion names to confidence scores
        
    Returns:
        Bar chart data structure
    """
    try:
        data_points = [
            ChartDataPoint(label=emotion.title(), value=confidence)
            for emotion, confidence in all_confidences.items()
        ]
        
        chart_data = BarChartData(
            title="Emotion Confidence Distribution",
            x_axis="Emotion",
            y_axis="Confidence",
            data=data_points
        )
        
        return {
            "success": True,
            "chart_type": "bar",
            "chart_data": chart_data.dict()
        }
    
    except Exception as e:
        logger.error(f"Error generating confidence chart: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate confidence chart: {str(e)}"
        )


@router.post(
    "/facial-emotion-distribution",
    summary="Generate facial emotion distribution chart",
    description="Create a bar chart showing facial emotion distribution from FER analysis"
)
async def get_facial_emotion_chart(
    emotion_distribution: Dict[str, float] = Body(..., description="Emotion distribution percentages")
):
    """
    Generate bar chart data for facial emotion distribution.
    
    Args:
        emotion_distribution: Dictionary mapping emotions to percentages
        
    Returns:
        Bar chart data structure
    """
    try:
        data_points = [
            ChartDataPoint(label=emotion.title(), value=percentage)
            for emotion, percentage in emotion_distribution.items()
        ]
        
        chart_data = BarChartData(
            title="Facial Emotion Distribution",
            x_axis="Emotion",
            y_axis="Percentage (%)",
            data=data_points
        )
        
        return {
            "success": True,
            "chart_type": "bar",
            "chart_data": chart_data.dict()
        }
    
    except Exception as e:
        logger.error(f"Error generating facial emotion chart: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate facial emotion chart: {str(e)}"
        )


@router.post(
    "/temporal-emotion",
    summary="Generate temporal emotion analysis chart",
    description="Create a line chart showing emotion changes over time"
)
async def get_temporal_chart(
    temporal_predictions: List[Dict[str, Any]] = Body(..., description="Temporal emotion predictions")
):
    """
    Generate Plotly line chart data for temporal emotion analysis.
    
    Args:
        temporal_predictions: List of temporal predictions with timestamps and confidences
        
    Returns:
        Plotly chart data structure
    """
    try:
        if not temporal_predictions:
            raise ValueError("No temporal predictions provided")
        
        # Extract emotion labels from first prediction
        first_pred = temporal_predictions[0]
        emotion_labels = list(first_pred.get('confidences', {}).keys())
        
        # Prepare data for each emotion
        traces = []
        for emotion in emotion_labels:
            timestamps = [pred['timestamp'] for pred in temporal_predictions]
            confidences = [
                pred.get('confidences', {}).get(emotion, 0) * 100 
                for pred in temporal_predictions
            ]
            
            traces.append({
                "x": timestamps,
                "y": confidences,
                "type": "scatter",
                "mode": "lines+markers",
                "name": emotion.title(),
                "line": {"width": 2}
            })
        
        layout = {
            "title": "Temporal Emotion Analysis",
            "xaxis": {"title": "Time (seconds)"},
            "yaxis": {"title": "Confidence (%)"},
            "hovermode": "x unified",
            "height": 500,
            "legend": {"orientation": "h", "y": -0.2}
        }
        
        chart_data = PlotlyChartData(
            data=traces,
            layout=layout
        )
        
        return {
            "success": True,
            "chart_type": "plotly",
            "chart_data": chart_data.dict()
        }
    
    except Exception as e:
        logger.error(f"Error generating temporal chart: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate temporal chart: {str(e)}"
        )


@router.post(
    "/modality-weights",
    summary="Generate modality weights visualization",
    description="Create a visualization showing modality importance weights"
)
async def get_modality_weights_chart(
    modality_weights: Dict[str, float] = Body(..., description="Modality weights (audio, visual, text)")
):
    """
    Generate bar chart data for modality importance weights.
    
    Args:
        modality_weights: Dictionary with audio, visual, text weights
        
    Returns:
        Bar chart data structure
    """
    try:
        data_points = [
            ChartDataPoint(label=modality.title(), value=weight * 100)
            for modality, weight in modality_weights.items()
        ]
        
        chart_data = BarChartData(
            title="Modality Importance Weights",
            x_axis="Modality",
            y_axis="Weight (%)",
            data=data_points
        )
        
        return {
            "success": True,
            "chart_type": "bar",
            "chart_data": chart_data.dict(),
            "most_important": max(modality_weights, key=modality_weights.get)
        }
    
    except Exception as e:
        logger.error(f"Error generating modality weights chart: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate modality weights chart: {str(e)}"
        )


@router.post(
    "/all-charts",
    summary="Generate all available charts from analysis results",
    description="Create all charts and visualizations from complete analysis results"
)
async def get_all_charts(
    analysis_results: Dict[str, Any] = Body(..., description="Complete analysis results")
):
    """
    Generate all charts from analysis results.
    
    Args:
        analysis_results: Complete analysis results dictionary
        
    Returns:
        Dictionary containing all chart data
    """
    try:
        charts = {}
        
        # Confidence distribution chart
        if 'prediction' in analysis_results:
            pred = analysis_results['prediction']
            if 'all_confidences' in pred:
                charts['confidence_distribution'] = await get_confidence_chart(pred['all_confidences'])
        
        # Facial emotion distribution chart
        if 'mental_health_analysis' in analysis_results:
            mh = analysis_results['mental_health_analysis']
            if 'emotion_distribution' in mh:
                charts['facial_emotion_distribution'] = await get_facial_emotion_chart(mh['emotion_distribution'])
        
        # Temporal emotion chart
        if 'temporal_predictions' in analysis_results and analysis_results['temporal_predictions']:
            charts['temporal_emotion'] = await get_temporal_chart(analysis_results['temporal_predictions'])
        
        # Modality weights chart
        if 'prediction' in analysis_results:
            pred = analysis_results['prediction']
            if 'modality_weights' in pred:
                charts['modality_weights'] = await get_modality_weights_chart(pred['modality_weights'])
        
        return {
            "success": True,
            "charts": charts,
            "chart_count": len(charts)
        }
    
    except Exception as e:
        logger.error(f"Error generating all charts: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate charts: {str(e)}"
        )

