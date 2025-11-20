"""
Reports Controller

Provides endpoints for generating comprehensive reports:
- Text reports
- AI-commented reports
- Feature extraction summaries
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Any
from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, Field
from datetime import datetime
from io import StringIO

# Add app directory to path
app_dir = Path(__file__).parent.parent.parent
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/reports", tags=["Reports"])


class ReportRequest(BaseModel):
    """Request model for generating reports."""
    analysis_results: Dict[str, Any] = Field(..., description="Complete analysis results")
    include_ai_commentary: bool = Field(False, description="Include AI commentary in report")
    llm_provider: str = Field("cloud", description="LLM provider for AI commentary: 'cloud' or 'local'")


class ReportResponse(BaseModel):
    """Response model for reports."""
    success: bool = Field(..., description="Whether report generation was successful")
    report_type: str = Field(..., description="Type of report: 'text' or 'ai_commented'")
    report_content: str = Field(..., description="Report content as text")
    word_count: int = Field(..., description="Number of words in report")
    generated_at: str = Field(..., description="Report generation timestamp")


def generate_comprehensive_report(results: dict) -> str:
    """
    Generate a comprehensive text report from analysis results.
    
    Args:
        results: Results dictionary from analysis
        
    Returns:
        Report text as string
    """
    report = StringIO()
    
    # Header
    report.write("=" * 80 + "\n")
    report.write("MULTIMODAL EMOTION RECOGNITION ANALYSIS REPORT\n")
    report.write("=" * 80 + "\n\n")
    report.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    # Video Information
    report.write("=" * 80 + "\n")
    report.write("VIDEO INFORMATION\n")
    report.write("=" * 80 + "\n")
    metadata = results.get('metadata', {})
    report.write(f"Duration: {metadata.get('duration', 0):.2f} seconds\n")
    report.write(f"Resolution: {metadata.get('width', 0)}x{metadata.get('height', 0)}\n")
    report.write(f"FPS: {metadata.get('fps', 0):.2f}\n")
    report.write(f"Frames Extracted: {len(results.get('frames', []))}\n")
    report.write("\n")
    
    # Emotion Prediction
    if 'prediction' in results:
        report.write("=" * 80 + "\n")
        report.write("EMOTION PREDICTION\n")
        report.write("=" * 80 + "\n")
        pred = results['prediction']
        report.write(f"Predicted Emotion: {pred['predicted_emotion'].upper()}\n")
        report.write(f"Confidence: {pred['confidence']:.1%}\n")
        
        if 'fusion_method' in pred:
            report.write(f"Fusion Method: {pred['fusion_method']}\n")
        
        report.write("\nConfidence Distribution:\n")
        report.write("-" * 80 + "\n")
        for emotion, conf in pred.get('all_confidences', {}).items():
            report.write(f"  {emotion.capitalize():<20}: {conf:.1%}\n")
        
        # Individual model predictions
        if 'individual_models' in pred:
            report.write("\nIndividual Model Predictions:\n")
            report.write("-" * 80 + "\n")
            for model_name, model_pred in pred['individual_models'].items():
                report.write(f"  {model_name.replace('_', ' ').title():<30}: {model_pred}\n")
        
        # Modality weights
        if 'modality_weights' in pred:
            report.write("\nModality Importance Weights:\n")
            report.write("-" * 80 + "\n")
            mod_weights = pred['modality_weights']
            for modality, weight in mod_weights.items():
                report.write(f"  {modality.capitalize():<20}: {weight:.1%}\n")
        
        # Reasoning (if available)
        if 'reasoning' in pred:
            report.write("\nEmotion-LLaMA Reasoning:\n")
            report.write("-" * 80 + "\n")
            report.write(f"  {pred['reasoning']}\n")
        
        report.write("\n")
    
    # Mental Health Analysis (FER-based)
    if 'mental_health_analysis' in results:
        report.write("=" * 80 + "\n")
        report.write("MENTAL HEALTH ANALYSIS (Facial Expression Recognition)\n")
        report.write("=" * 80 + "\n")
        mh = results['mental_health_analysis']
        report.write(f"Mental Health Score: {mh['mental_health_score']:.1f}/100\n")
        report.write(f"Dominant Emotion: {mh['dominant_emotion'].capitalize()}\n")
        report.write(f"Average Confidence: {mh['avg_confidence']:.1%}\n")
        report.write(f"Frames Analyzed: {mh['num_frames']}\n")
        
        report.write("\nEmotion Distribution:\n")
        report.write("-" * 80 + "\n")
        for emotion, percentage in mh.get('emotion_distribution', {}).items():
            report.write(f"  {emotion.capitalize():<20}: {percentage:.1f}%\n")
        
        report.write(f"\nPositive Emotions: {mh.get('positive_percentage', 0):.1f}%\n")
        report.write(f"Negative Emotions: {mh.get('negative_percentage', 0):.1f}%\n")
        
        if 'status' in mh:
            report.write(f"\nStatus: {mh['status']}\n")
        if 'recommendation' in mh and mh['recommendation']:
            report.write(f"Recommendation: {mh['recommendation']}\n")
        
        report.write("\n")
    
    # Temporal Analysis
    if 'temporal_predictions' in results and results['temporal_predictions']:
        report.write("=" * 80 + "\n")
        report.write("TEMPORAL EMOTION ANALYSIS\n")
        report.write("=" * 80 + "\n")
        temporal = results['temporal_predictions']
        report.write(f"Total Time Points: {len(temporal)}\n")
        report.write("\nEmotion Timeline:\n")
        report.write("-" * 80 + "\n")
        for pred in temporal[:10]:  # Show first 10
            report.write(f"  {pred['timestamp']:>6.1f}s: {pred['emotion'].upper():<15} ({pred.get('confidence', 0):.1%})\n")
        if len(temporal) > 10:
            report.write(f"  ... and {len(temporal) - 10} more time points\n")
        report.write("\n")
    
    # Transcription
    if 'transcription' in results and results['transcription']:
        transcript = results['transcription']
        if transcript.get('text') and transcript.get('text') != 'No speech detected':
            report.write("=" * 80 + "\n")
            report.write("TRANSCRIPTION\n")
            report.write("=" * 80 + "\n")
            report.write(f"Language: {transcript.get('language', 'en').upper()}\n")
            report.write(f"Word Count: {len(transcript.get('text', '').split())}\n")
            report.write("\nTranscript:\n")
            report.write("-" * 80 + "\n")
            report.write(f"{transcript.get('text', '')}\n")
            report.write("\n")
    
    # Feature Extraction Summary
    report.write("=" * 80 + "\n")
    report.write("FEATURE EXTRACTION SUMMARY\n")
    report.write("=" * 80 + "\n")
    
    if 'features' in results and isinstance(results['features'], dict):
        features = results['features']
        if 'audio' in features:
            try:
                audio_feat = features['audio']
                if hasattr(audio_feat, '__len__'):
                    report.write(f"Audio Features: {len(audio_feat)}\n")
                else:
                    report.write("Audio Features: Extracted\n")
            except:
                report.write("Audio Features: Extracted\n")
        else:
            report.write("Audio Features: Not extracted\n")
        
        if 'visual' in features:
            try:
                visual_feat = features['visual']
                if hasattr(visual_feat, '__len__'):
                    report.write(f"Visual Features: {len(visual_feat)}\n")
                else:
                    report.write("Visual Features: Extracted\n")
            except:
                report.write("Visual Features: Extracted\n")
        else:
            report.write("Visual Features: Not extracted\n")
        
        if 'text' in features:
            try:
                text_feat = features['text']
                if hasattr(text_feat, '__len__'):
                    report.write(f"Text Features: {len(text_feat)}\n")
                else:
                    report.write("Text Features: Extracted\n")
            except:
                report.write("Text Features: Extracted\n")
        else:
            report.write("Text Features: Not extracted\n")
    
    report.write("\n")
    
    # Footer
    report.write("=" * 80 + "\n")
    report.write("END OF REPORT\n")
    report.write("=" * 80 + "\n")
    
    return report.getvalue()


@router.post(
    "/comprehensive",
    summary="Generate comprehensive text report",
    description="Generate a detailed text report from analysis results"
)
async def generate_report(
    request: ReportRequest = Body(..., description="Report generation request")
):
    """
    Generate a comprehensive text report from analysis results.
    
    Args:
        request: Report request with analysis results
        
    Returns:
        Comprehensive text report
    """
    try:
        report_content = generate_comprehensive_report(request.analysis_results)
        word_count = len(report_content.split())
        
        return ReportResponse(
            success=True,
            report_type="text",
            report_content=report_content,
            word_count=word_count,
            generated_at=datetime.now().isoformat()
        ).dict()
    
    except Exception as e:
        logger.error(f"Error generating report: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate report: {str(e)}"
        )


@router.post(
    "/ai-commented",
    summary="Generate AI-commented report",
    description="Generate a comprehensive report with AI commentary and insights"
)
async def generate_ai_commented_report(
    request: ReportRequest = Body(..., description="Report generation request with AI commentary")
):
    """
    Generate an AI-commented report using LLM to provide insights.
    
    Args:
        request: Report request with analysis results and AI commentary flag
        
    Returns:
        AI-commented report
    """
    try:
        # Generate base report
        base_report = generate_comprehensive_report(request.analysis_results)
        
        # Add AI commentary if requested
        if request.include_ai_commentary:
            try:
                from modules.ai_agent import MeetingAnalysisAgent
                
                # Initialize AI agent
                agent = MeetingAnalysisAgent(provider=request.llm_provider)
                
                if not agent.client:
                    # Fallback if AI not available
                    report_content = base_report + "\n\n" + "=" * 80 + "\n" + \
                                   "AI COMMENTARY UNAVAILABLE\n" + \
                                   "=" * 80 + "\n" + \
                                   "To enable AI commentary, please configure:\n" + \
                                   "- For OpenAI: Set OPENAI_API_KEY in .env\n" + \
                                   "- For Local LLaMA3: Set LOCAL_LLM_BASE_URL in .env and ensure Ollama/LM Studio is running\n"
                else:
                    # Build prompt for AI commentary
                    prompt = f"""You are an expert emotion analysis consultant reviewing a comprehensive emotion recognition report.

Please review the following report and provide detailed commentary, insights, and recommendations.

REPORT TO REVIEW:
{base_report}

INSTRUCTIONS:
1. Read through the entire report carefully
2. Provide your expert commentary on the findings
3. Highlight any concerning patterns or positive indicators
4. Offer actionable recommendations based on the data
5. Connect emotional patterns to potential implications
6. Be specific and reference data points from the report

Please structure your commentary as follows:

=== EXPERT COMMENTARY ===

## Overall Assessment
[Your overall assessment of the emotional analysis findings]

## Key Observations
[3-5 key observations about the emotional patterns, mental health indicators, and meeting dynamics]

## Pattern Analysis
[Analysis of any patterns you notice in the temporal data, emotion distribution, or other metrics]

## Implications & Recommendations
[Specific recommendations based on the findings, including any concerns or positive aspects to build upon]

## Action Items
[Concrete action items that should be considered based on this analysis]

=== END COMMENTARY ===

Please be thorough, professional, and provide actionable insights."""

                    # Generate AI commentary
                    response = agent.client.chat.completions.create(
                        model=agent.model,
                        messages=[
                            {
                                "role": "system",
                                "content": "You are an expert emotion analysis consultant with deep expertise in workplace psychology, team dynamics, and emotional intelligence. You provide insightful, actionable commentary on emotion recognition reports."
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        temperature=0.7,
                        max_tokens=2000
                    )
                    
                    ai_commentary = response.choices[0].message.content
                    report_content = base_report + "\n\n" + ai_commentary
            except Exception as e:
                logger.warning(f"AI commentary generation failed: {e}")
                report_content = base_report + "\n\n" + "=" * 80 + "\n" + \
                               "AI COMMENTARY UNAVAILABLE\n" + \
                               "=" * 80 + "\n" + \
                               f"Error: {str(e)}\n"
        else:
            report_content = base_report
        
        word_count = len(report_content.split())
        
        return ReportResponse(
            success=True,
            report_type="ai_commented" if request.include_ai_commentary else "text",
            report_content=report_content,
            word_count=word_count,
            generated_at=datetime.now().isoformat()
        ).dict()
    
    except Exception as e:
        logger.error(f"Error generating AI-commented report: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate AI-commented report: {str(e)}"
        )

