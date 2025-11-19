"""Results Tab"""

import streamlit as st
import pandas as pd
import json
import os
import logging
from datetime import datetime
from io import StringIO

# Try to import TTS functionality
try:
    from tts import get_tts_engine, text_to_speech
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    logging.warning("TTS module not available. Install EmotiVoice to enable text-to-speech.")

logger = logging.getLogger(__name__)


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
        report.write(f"Positive Emotions: {mh['positive_percentage']:.1f}%\n")
        report.write(f"Negative Emotions: {mh['negative_percentage']:.1f}%\n")
        
        report.write("\nEmotion Distribution:\n")
        report.write("-" * 80 + "\n")
        for emotion, percentage in mh.get('emotion_distribution', {}).items():
            report.write(f"  {emotion.capitalize():<20}: {percentage:.1f}%\n")
        report.write("\n")
    
    # Transcription
    if 'transcription' in results:
        report.write("=" * 80 + "\n")
        report.write("TRANSCRIPTION\n")
        report.write("=" * 80 + "\n")
        transcription = results['transcription']
        transcript_text = transcription.get('text', 'No speech detected')
        report.write(f"Language: {transcription.get('language', 'en').upper()}\n")
        report.write(f"Word Count: {len(transcript_text.split()) if transcript_text != 'No speech detected' else 0}\n")
        report.write("\nTranscript:\n")
        report.write("-" * 80 + "\n")
        report.write(f"{transcript_text}\n")
        report.write("\n")
    
    # AI Analysis
    if 'ai_analysis' in results and results['ai_analysis']:
        report.write("=" * 80 + "\n")
        report.write("AI MEETING ANALYSIS\n")
        report.write("=" * 80 + "\n")
        ai_analysis = results['ai_analysis']
        
        if ai_analysis.get('summary'):
            report.write("\nExecutive Summary:\n")
            report.write("-" * 80 + "\n")
            report.write(f"{ai_analysis['summary']}\n")
        
        if ai_analysis.get('key_insights'):
            report.write("\nKey Insights:\n")
            report.write("-" * 80 + "\n")
            for i, insight in enumerate(ai_analysis['key_insights'], 1):
                report.write(f"{i}. {insight}\n")
        
        if ai_analysis.get('emotional_dynamics'):
            report.write("\nEmotional Dynamics:\n")
            report.write("-" * 80 + "\n")
            ed = ai_analysis['emotional_dynamics']
            if isinstance(ed, dict) and 'analysis' in ed:
                report.write(f"{ed['analysis']}\n")
            elif isinstance(ed, dict):
                for key, value in ed.items():
                    report.write(f"{key.replace('_', ' ').title()}: {value}\n")
            else:
                report.write(f"{ed}\n")
        
        if ai_analysis.get('recommendations'):
            report.write("\nRecommendations:\n")
            report.write("-" * 80 + "\n")
            for i, rec in enumerate(ai_analysis['recommendations'], 1):
                report.write(f"{i}. {rec}\n")
        
        report.write("\n")
    
    # Temporal Analysis
    if 'temporal_predictions' in results and len(results['temporal_predictions']) > 0:
        report.write("=" * 80 + "\n")
        report.write("TEMPORAL EMOTION ANALYSIS\n")
        report.write("=" * 80 + "\n")
        temporal_data = results['temporal_predictions']
        report.write(f"Time Points Analyzed: {len(temporal_data)}\n\n")
        report.write("Emotion Timeline:\n")
        report.write("-" * 80 + "\n")
        report.write(f"{'Time (s)':<12} {'Emotion':<20} {'Confidence':<15}\n")
        report.write("-" * 80 + "\n")
        for pred in temporal_data[:20]:  # Limit to first 20 for readability
            time_str = f"{pred['timestamp']:.1f}"
            emotion = pred['emotion'].capitalize()
            confidence = f"{pred['confidences'][pred['emotion']]:.1%}"
            report.write(f"{time_str:<12} {emotion:<20} {confidence:<15}\n")
        if len(temporal_data) > 20:
            report.write(f"\n... and {len(temporal_data) - 20} more time points\n")
        report.write("\n")
    
    # Feature Information
    report.write("=" * 80 + "\n")
    report.write("EXTRACTED FEATURES\n")
    report.write("=" * 80 + "\n")
    features = results.get('features', {})
    if 'audio' in features:
        report.write(f"Audio Features: {len(features['audio'])} features extracted\n")
    else:
        report.write("Audio Features: Not extracted\n")
    
    if 'visual' in features:
        report.write(f"Visual Features: {len(features['visual'])} features extracted\n")
    else:
        report.write("Visual Features: Not extracted\n")
    
    if 'text' in features:
        report.write(f"Text Features: {len(features['text'])} features extracted\n")
    else:
        report.write("Text Features: Not extracted\n")
    
    report.write("\n")
    
    # Footer
    report.write("=" * 80 + "\n")
    report.write("END OF REPORT\n")
    report.write("=" * 80 + "\n")
    
    return report.getvalue()


def generate_ai_commented_report(results: dict) -> str:
    """
    Generate an AI-commented report using LLaMA3 to provide insights and commentary.
    
    Args:
        results: Results dictionary from analysis
        
    Returns:
        AI-commented report text as string
    """
    # First generate the base report
    base_report = generate_comprehensive_report(results)
    
    # Get LLM provider from session state
    llm_provider = st.session_state.get('llm_provider', 'Cloud (OpenAI)')
    provider = "local" if "Local" in llm_provider else "cloud"
    
    # Debug logging
    logger.info(f"AI Report Generation - Provider from session: {llm_provider}")
    logger.info(f"AI Report Generation - Using provider: {provider}")
    
    try:
        from modules.ai_agent import MeetingAnalysisAgent
        
        # Initialize AI agent
        logger.info(f"Initializing MeetingAnalysisAgent with provider={provider}")
        agent = MeetingAnalysisAgent(provider=provider)
        
        logger.info(f"Agent initialized - client: {agent.client}, model: {agent.model}, base_url: {agent.base_url}")
        
        if not agent.client:
            # Fallback if AI not available
            error_msg = f"Agent client is None. Provider: {provider}, API Key exists: {bool(agent.api_key)}"
            logger.error(error_msg)
            return base_report + "\n\n" + "=" * 80 + "\n" + \
                   "AI COMMENTARY UNAVAILABLE\n" + \
                   "=" * 80 + "\n" + \
                   "To enable AI commentary, please configure:\n" + \
                   "- For OpenAI: Set OPENAI_API_KEY in .env\n" + \
                   "- For Local LLaMA3: Set LOCAL_LLM_BASE_URL in .env and ensure Ollama/LM Studio is running\n" + \
                   f"\nDebug: {error_msg}\n"
        
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
        
        # Combine base report with AI commentary
        ai_report = base_report + "\n\n" + "=" * 80 + "\n" + \
                   "AI EXPERT COMMENTARY (Generated by " + agent.model + ")\n" + \
                   "=" * 80 + "\n\n" + \
                   ai_commentary + "\n\n" + \
                   "=" * 80 + "\n" + \
                   "END OF AI-COMMENTED REPORT\n" + \
                   "=" * 80 + "\n"
        
        logger.info(f"AI-commented report generated successfully using {agent.model}")
        return ai_report
        
    except ImportError:
        logger.error("AI agent module not available")
        return base_report + "\n\n" + "=" * 80 + "\n" + \
               "AI COMMENTARY UNAVAILABLE\n" + \
               "=" * 80 + "\n" + \
               "AI agent module not found. Please ensure modules.ai_agent is available.\n"
    except Exception as e:
        logger.error(f"Error generating AI commentary: {e}", exc_info=True)
        # Return base report with error message
        return base_report + "\n\n" + "=" * 80 + "\n" + \
               "AI COMMENTARY GENERATION FAILED\n" + \
               "=" * 80 + "\n" + \
               f"Error: {str(e)}\n" + \
               "The base report is still available above.\n"


def render_results_tab():
    """Render the results tab."""
    st.header("Analysis Results")
    
    if 'results' not in st.session_state:
        st.info("Upload and analyze a video to see results here")
    else:
        results = st.session_state['results']
        config = st.session_state['config']
        
        # Display results
        st.subheader("Video Information")
        metadata = results.get('metadata', {})
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Duration", f"{metadata.get('duration', 0):.1f}s")
        with col2:
            st.metric("Resolution", f"{metadata.get('width', 0)}x{metadata.get('height', 0)}")
        with col3:
            st.metric("FPS", f"{metadata.get('fps', 0):.1f}")
        with col4:
            st.metric("Frames", len(results.get('frames', [])))
        
        st.markdown("---")
        
        # Emotion Analysis by Modality
        st.header("Emotion Analysis by Modality")
        st.write("Results categorized by analysis type (facial, audio, text, and combined)")
        
        # Create tabs for different modalities
        modality_tabs = st.tabs([
            "Facial (FER)",
            "Audio/Voice", 
            "Text/Transcript",
            "Multimodal Combined"
        ])
        
        # Tab 1: Facial Emotions (FER) - From FER Analysis
        with modality_tabs[0]:
            st.subheader("Facial Expression Recognition")
            st.caption("Emotion detection based purely on facial expressions from video frames")
            
            if 'mental_health_analysis' in results and results['mental_health_analysis']:
                mh = results['mental_health_analysis']
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Dominant Facial Emotion", mh['dominant_emotion'].title())
                with col2:
                    st.metric("Average Confidence", f"{mh['avg_confidence']:.1%}")
                with col3:
                    st.metric("Frames Analyzed", mh['num_frames'])
                
                # Facial emotion distribution
                st.write("**Facial Emotion Distribution:**")
                facial_df = pd.DataFrame({
                    'Emotion': [e.title() for e in mh['emotion_distribution'].keys()],
                    'Percentage': list(mh['emotion_distribution'].values())
                })
                st.bar_chart(facial_df.set_index('Emotion'))
                
                # Positive vs Negative
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Positive Emotions", f"{mh['positive_percentage']:.1f}%", 
                             help="Happy, Surprise")
                with col2:
                    st.metric("Negative Emotions", f"{mh['negative_percentage']:.1f}%",
                             help="Sad, Angry, Fear, Disgust")
            else:
                st.info("Facial emotion analysis not available. Upload a video with visible faces.")
        
        # Tab 2: Audio/Voice Emotions
        with modality_tabs[1]:
            st.subheader("Audio & Voice Analysis")
            st.caption("Emotion detection from voice tone, prosody, pitch, and acoustic features")
            
            if 'audio_features' in results and results['audio_features']:
                st.write("**Audio Features Extracted:**")
                audio_feat = results['audio_features']
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Audio Features", len(audio_feat))
                with col2:
                    if 'prosodic' in audio_feat:
                        st.write("Prosodic (pitch, energy, rhythm)")
                    if 'spectral' in audio_feat:
                        st.write("Spectral (MFCCs, mel-spectrogram)")
                with col3:
                    if 'opensmile' in audio_feat:
                        st.write("OpenSMILE features")
                    st.write("Voice quality (jitter, shimmer)")
                
                st.info("Audio-specific emotion predictions are integrated in the 'Multimodal Combined' tab. Individual audio models analyze voice tone, speaking rate, and vocal patterns.")
            else:
                st.warning("No audio features extracted from this video")
        
        # Tab 3: Text Emotions
        with modality_tabs[2]:
            st.subheader("Text & Transcription Analysis")
            st.caption("Emotion detection from speech content, keywords, and semantic meaning")
            
            if 'transcription' in results and results['transcription']:
                transcript_text = results['transcription'].get('text', '')
                
                if transcript_text and transcript_text != 'No speech detected':
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Word Count", len(transcript_text.split()))
                    with col2:
                        st.metric("Language", results['transcription'].get('language', 'en').upper())
                    
                    # Show text features if available
                    if 'text_features' in results and results['text_features']:
                        st.write("**Text Features Extracted:**")
                        text_feat = results['text_features']
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Text Features", len(text_feat))
                        with col2:
                            if 'lexical' in text_feat:
                                st.write("Lexical features")
                            if 'sentiment' in text_feat:
                                st.write("Sentiment analysis")
                        with col3:
                            if 'embeddings' in text_feat:
                                st.write("Semantic embeddings")
                            st.write("Emotion lexicons")
                    
                    # Show transcript in expander
                    with st.expander("View Full Transcript"):
                        st.text_area("Transcript Content", transcript_text, height=200, key="transcript_text")
                    
                    st.info("Text-specific emotion predictions are integrated in the 'Multimodal Combined' tab. Text analysis examines word choice, sentiment, and semantic patterns.")
                else:
                    st.warning("No speech detected in the video")
            else:
                st.warning("Transcription not available")
        
        # Tab 4: Multimodal Combined
        with modality_tabs[3]:
            st.subheader("Multimodal Combined Analysis")
            st.caption("Fusion of facial expressions, audio, and text for comprehensive emotion detection")
            
            # Prediction results
            if 'prediction' in results:
                pred = results['prediction']
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown(f"### **{pred['predicted_emotion'].upper()}**")
                    st.metric("Confidence", f"{pred['confidence']:.1%}")
                    
                    # Show fusion strategy
                    if 'fusion_method' in pred:
                        st.caption(f"Method: {pred['fusion_method']}")
                    
                    # TTS for Emotion Result
                    if TTS_AVAILABLE:
                        st.markdown("---")
                        st.markdown("### 🔊 Text-to-Speech")
                        
                        # Create emotion description text
                        emotion_text = f"The detected emotion is {pred['predicted_emotion']} with {pred['confidence']:.0%} confidence."
                        
                        # Map emotion to TTS emotion
                        emotion_map = {
                            'happy': 'happy',
                            'sad': 'sad',
                            'angry': 'angry',
                            'surprise': 'surprised',
                            'fear': 'worried',
                            'disgust': 'sad',
                            'neutral': 'neutral'
                        }
                        tts_emotion = emotion_map.get(pred['predicted_emotion'].lower(), 'neutral')
                        
                        if st.button("🎵 Listen to Result", use_container_width=True, key="tts_emotion_result"):
                            with st.spinner("Generating speech..."):
                                try:
                                    tts_engine = get_tts_engine()
                                    if tts_engine.is_ready():
                                        audio_bytes = text_to_speech(
                                            text=emotion_text,
                                            emotion=tts_emotion,
                                            speaker="8051"
                                        )
                                        if audio_bytes:
                                            st.audio(audio_bytes, format="audio/wav")
                                            st.success("Audio generated!")
                                        else:
                                            st.error("Failed to generate audio")
                                    else:
                                        st.warning("TTS engine not ready. Please wait...")
                                except Exception as e:
                                    st.error(f"TTS error: {e}")
                                    logger.error(f"TTS error: {e}", exc_info=True)
                        
                        st.caption(f"Voice emotion: {tts_emotion.title()}")
                
                with col2:
                    st.markdown("**Confidence Distribution:**")
                    emotion_labels = list(pred['all_confidences'].keys())
                    confidences = list(pred['all_confidences'].values())
                    
                    # Bar chart
                    df = pd.DataFrame({
                        'Emotion': emotion_labels,
                        'Confidence': confidences
                    })
                    st.bar_chart(df.set_index('Emotion'))
            
                # Model-specific analysis
                if 'reasoning' in pred:
                    st.markdown("---")
                    st.subheader("Emotion-LLaMA Analysis")
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown("**Reasoning:**")
                        st.info(pred['reasoning'])
                    
                    with col2:
                        if 'intensity' in pred:
                            st.markdown("**Emotion Intensity:**")
                            st.progress(pred['intensity'], text=f"{pred['intensity']:.1%}")
                
                # Hybrid model specific information
                if 'modality_weights' in pred:
                    st.markdown("---")
                    st.subheader("Hybrid Model Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Modality Importance:**")
                        mod_weights = pred['modality_weights']
                        st.progress(mod_weights['audio'], text=f"Audio: {mod_weights['audio']:.1%}")
                        st.progress(mod_weights['visual'], text=f"Visual: {mod_weights['visual']:.1%}")
                        st.progress(mod_weights['text'], text=f"Text: {mod_weights['text']:.1%}")
                        
                        # Highlight most important modality
                        most_important = max(mod_weights, key=mod_weights.get)
                        st.info(f"**Most important**: {most_important.title()} ({mod_weights[most_important]:.1%})")
                    
                    with col2:
                        st.markdown("**Model Agreement:**")
                        if 'individual_models' in pred:
                            individual = pred['individual_models']
                            st.text(f"RFRBoost:      {individual['rfrboost']}")
                            st.text(f"Attention+Deep: {individual['attention_deep']}")
                            st.text(f"MLP Baseline:   {individual['mlp_baseline']}")
                            
                            # Check agreement
                            predictions_list = list(individual.values())
                            if len(set(predictions_list)) == 1:
                                st.success("All models agree!")
                            elif predictions_list.count(pred['predicted_emotion']) >= 2:
                                st.success("Majority agreement")
                            else:
                                st.warning("Models disagree")
                
                # Maelfabien model specific information
                if 'individual_models' in pred and any(k in pred['individual_models'] for k in ['text_cnn_lstm', 'audio_time_cnn', 'video_xception']):
                    st.markdown("---")
                    st.subheader("Maelfabien Multimodal - Individual Modality Predictions")
                    
                    col1, col2, col3 = st.columns(3)
                    individual = pred['individual_models']
                    
                    with col1:
                        st.markdown("**Text CNN-LSTM:**")
                        st.write(individual.get('text_cnn_lstm', 'N/A').title())
                        st.caption("Analyzes transcript")
                    
                    with col2:
                        st.markdown("**Audio Time-CNN:**")
                        st.write(individual.get('audio_time_cnn', 'N/A').title())
                        st.caption("Analyzes voice")
                    
                    with col3:
                        st.markdown("**Video XCeption:**")
                        st.write(individual.get('video_xception', 'N/A').title())
                        st.caption("Analyzes facial expressions")
            else:
                st.info("No multimodal prediction available for this video")
        
        st.markdown("---")
        
        # Temporal emotion analysis
        if 'temporal_predictions' in results and len(results['temporal_predictions']) > 0:
            st.subheader("Temporal Emotion Analysis")
            
            temporal_data = results['temporal_predictions']
            
            # Create time series data for all emotions
            import plotly.graph_objects as go
            
            fig = go.Figure()
            
            emotion_labels = list(temporal_data[0]['confidences'].keys())
            
            for emotion in emotion_labels:
                timestamps = [pred['timestamp'] for pred in temporal_data]
                confidences = [pred['confidences'][emotion] * 100 for pred in temporal_data]
                
                fig.add_trace(go.Scatter(
                    x=timestamps,
                    y=confidences,
                    mode='lines+markers',
                    name=emotion.capitalize(),
                    hovertemplate='Time: %{x:.1f}s<br>Confidence: %{y:.1f}%<extra></extra>'
                ))
            
            fig.update_layout(
                title="Emotion Distribution Over Time",
                xaxis_title="Time (seconds)",
                yaxis_title="Confidence (%)",
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show dominant emotion timeline
            st.markdown("**Dominant Emotion Timeline:**")
            timeline_text = ""
            for pred in temporal_data:
                time_str = f"{pred['timestamp']:.1f}s"
                emotion = pred['emotion'].capitalize()
                confidence = pred['confidences'][pred['emotion']] * 100
                timeline_text += f"- **{time_str}**: {emotion} ({confidence:.1f}%)\n"
            
            st.markdown(timeline_text)
            
            # Mental health analysis (FER-based)
            if 'mental_health_analysis' in results:
                st.markdown("---")
                st.subheader("Mental Health Analysis (FER-based)")
                
                mh = results['mental_health_analysis']
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Mental Health Score", f"{mh['mental_health_score']:.1f}/100")
                    st.caption("Based on facial expression analysis")
                
                with col2:
                    st.metric("Average Confidence", f"{mh['avg_confidence']:.1%}")
                    st.caption(f"Across {mh['num_frames']} frames")
                
                with col3:
                    st.metric("Dominant Emotion", mh['dominant_emotion'].capitalize())
                    st.caption("Most frequent emotion")
                
                # Emotion distribution
                st.markdown("**Emotion Distribution:**")
                emotion_dist = mh['emotion_distribution']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("*Positive Emotions:*")
                    for emotion in ['happy', 'surprise', 'neutral']:
                        if emotion in emotion_dist:
                            st.text(f"{emotion.capitalize()}: {emotion_dist[emotion]:.1f}%")
                    st.info(f"Total Positive: {mh['positive_percentage']:.1f}%")
                
                with col2:
                    st.markdown("*Negative Emotions:*")
                    for emotion in ['angry', 'disgust', 'fear', 'sad']:
                        if emotion in emotion_dist:
                            st.text(f"{emotion.capitalize()}: {emotion_dist[emotion]:.1f}%")
                    st.warning(f"Total Negative: {mh['negative_percentage']:.1f}%")
                
                # Mental health interpretation
                score = mh['mental_health_score']
                if score >= 70:
                    st.success("Mental Health Status: Good - Predominantly positive emotional expressions")
                elif score >= 50:
                    st.info("Mental Health Status: Moderate - Balanced emotional expressions")
                elif score >= 30:
                    st.warning("Mental Health Status: Concerning - Elevated negative emotional expressions")
                else:
                    st.error("Mental Health Status: At Risk - Predominantly negative emotional expressions. Consider professional consultation.")
        
        st.markdown("---")
        
        # AI Agent Analysis (NEW SECTION)
        if 'ai_analysis' in results and results['ai_analysis']:
            st.header("AI Meeting Analysis")
            
            ai_analysis = results['ai_analysis']
            
            if not ai_analysis.get('agent_available', False):
                st.info("AI Agent running in limited mode (OpenAI API not configured). For full LLM-powered analysis, add your OPENAI_API_KEY to .env file.")
            
            # Executive Summary
            if ai_analysis.get('summary'):
                st.subheader("Executive Summary")
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write(ai_analysis['summary'])
                
                # TTS for Summary
                with col2:
                    if TTS_AVAILABLE:
                        st.markdown("### 🔊")
                        if st.button("🎵 Listen to Summary", use_container_width=True, key="tts_summary"):
                            with st.spinner("Generating speech..."):
                                try:
                                    tts_engine = get_tts_engine()
                                    if tts_engine.is_ready():
                                        summary_text = ai_analysis['summary'][:500]  # Limit length
                                        audio_bytes = text_to_speech(
                                            text=summary_text,
                                            emotion="neutral",
                                            speaker="8051"
                                        )
                                        if audio_bytes:
                                            st.audio(audio_bytes, format="audio/wav")
                                            st.success("Audio generated!")
                                        else:
                                            st.error("Failed to generate audio")
                                    else:
                                        st.warning("TTS engine not ready")
                                except Exception as e:
                                    st.error(f"TTS error: {e}")
                                    logger.error(f"TTS error: {e}", exc_info=True)
            
            # Key Insights
            if ai_analysis.get('key_insights'):
                st.subheader("Key Insights")
                for insight in ai_analysis['key_insights']:
                    st.markdown(f"- {insight}")
            
            # Emotional Dynamics
            if ai_analysis.get('emotional_dynamics'):
                st.subheader("Emotional Dynamics")
                ed = ai_analysis['emotional_dynamics']
                if isinstance(ed, dict) and 'analysis' in ed:
                    st.write(ed['analysis'])
                elif isinstance(ed, dict):
                    for key, value in ed.items():
                        st.write(f"**{key.replace('_', ' ').title()}**: {value}")
                else:
                    st.write(ed)
            
            # Recommendations
            if ai_analysis.get('recommendations'):
                st.subheader("Recommendations")
                for i, rec in enumerate(ai_analysis['recommendations'], 1):
                    st.markdown(f"{i}. {rec}")
            
            # Knowledge Base Context
            if ai_analysis.get('knowledge_base_context'):
                with st.expander("Knowledge Base Context Used"):
                    for i, ctx in enumerate(ai_analysis['knowledge_base_context'], 1):
                        st.markdown(f"**Context {i}** (Relevance: {ctx['similarity_score']:.0%})")
                        st.text(ctx['content'][:300] + "...")
                        st.caption(f"Document ID: {ctx['document_id']}, Page: {ctx.get('page_number', 'N/A')}")
                        st.markdown("---")
            
            # Detailed Analysis
            if ai_analysis.get('detailed_analysis'):
                with st.expander("Detailed Analysis"):
                    st.write(ai_analysis['detailed_analysis'])
            
            # Raw LLM Output (Full Transparency)
            if ai_analysis.get('raw_llm_response'):
                st.subheader("LLM Transparency")
                st.info(f"**Model Used**: {ai_analysis.get('llm_model', 'Unknown')}")
                
                # Separate expanders at same level (no nesting)
                with st.expander("View Raw LLM Response", expanded=False):
                    st.markdown("**Complete unprocessed output from the LLM:**")
                    st.text_area(
                        "Raw Response",
                        value=ai_analysis['raw_llm_response'],
                        height=400,
                        disabled=True,
                        label_visibility="collapsed"
                    )
                
                with st.expander("View Prompt Sent to LLM", expanded=False):
                    st.markdown("**Full prompt that was sent to the LLM:**")
                    st.text_area(
                        "Prompt",
                        value=ai_analysis.get('llm_prompt', 'Prompt not available'),
                        height=600,
                        disabled=True,
                        label_visibility="collapsed"
                    )
            
            # Error display
            if 'error' in ai_analysis:
                st.error(f"Analysis error: {ai_analysis['error']}")
            
            st.markdown("---")
        
        # Transcription
        if 'transcription' in results:
            st.subheader("Video Transcript")
            transcript_text = results['transcription'].get('text', 'No speech detected')
            
            if transcript_text and transcript_text != 'No speech detected':
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.text_area(
                        "Full Transcript",
                        transcript_text,
                        height=150
                    )
                    
                    # Show word count
                    word_count = len(transcript_text.split())
                    st.caption(f"Total words: {word_count}")
                
                # TTS for Transcript
                with col2:
                    st.markdown("### 🔊 Text-to-Speech")
                    if TTS_AVAILABLE:
                        # Get detected emotion for TTS emotion
                        detected_emotion = "neutral"
                        if 'prediction' in results:
                            detected_emotion = results['prediction'].get('predicted_emotion', 'neutral')
                        
                        # Map emotion labels to TTS emotions
                        emotion_map = {
                            'happy': 'happy',
                            'sad': 'sad',
                            'angry': 'angry',
                            'surprise': 'surprised',
                            'fear': 'worried',
                            'disgust': 'sad',
                            'neutral': 'neutral'
                        }
                        tts_emotion = emotion_map.get(detected_emotion.lower(), 'neutral')
                        
                        if st.button("🎵 Listen to Transcript", use_container_width=True, key="tts_transcript"):
                            with st.spinner("Generating speech..."):
                                try:
                                    tts_engine = get_tts_engine()
                                    if tts_engine.is_ready():
                                        audio_bytes = text_to_speech(
                                            text=transcript_text[:500],  # Limit length
                                            emotion=tts_emotion,
                                            speaker="8051"
                                        )
                                        if audio_bytes:
                                            st.audio(audio_bytes, format="audio/wav")
                                            st.success("Audio generated!")
                                        else:
                                            st.error("Failed to generate audio")
                                    else:
                                        st.warning("TTS engine not ready. Please wait...")
                                except Exception as e:
                                    st.error(f"TTS error: {e}")
                                    logger.error(f"TTS error: {e}", exc_info=True)
                        
                        st.caption(f"Emotion: {tts_emotion.title()}")
                    else:
                        st.info("""
                        **TTS Not Available**
                        
                        To enable text-to-speech:
                        1. Run: `python tts/setup_emotivoice.py`
                        2. Restart the app
                        """)
            else:
                st.info("No speech detected in the video")
        
        st.markdown("---")
        
        # Extracted frames display
        if 'frame_paths' in results and len(results['frame_paths']) > 0:
            st.subheader("Extracted Frames (LlamaIndex Approach)")
            st.caption(f"Extracted {len(results['frame_paths'])} frames at 0.2 FPS (1 frame every 5 seconds)")
            
            # Display sample frames
            num_display = min(6, len(results['frame_paths']))
            st.markdown(f"**Showing {num_display} sample frames:**")
            
            cols = st.columns(3)
            for idx in range(num_display):
                frame_path = results['frame_paths'][idx]
                if os.path.exists(frame_path):
                    with cols[idx % 3]:
                        # Calculate timestamp
                        timestamp = idx * 5.0  # 5 seconds per frame
                        st.image(frame_path, caption=f"Frame at {timestamp:.1f}s", use_column_width=True)
            
            # Show frame storage info
            if 'frames_folder' in results:
                st.info(f"All frames saved to: `{results['frames_folder']}`")
        
        st.markdown("---")
        
        # Feature information
        st.subheader("Extracted Features")
        
        features = results.get('features', {})
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'audio' in features:
                st.success(f"Audio: {len(features['audio'])} features")
            else:
                st.warning("Audio: Not extracted")
        
        with col2:
            if 'visual' in features:
                st.success(f"Visual: {len(features['visual'])} features")
            else:
                st.warning("Visual: Not extracted")
        
        with col3:
            if 'text' in features:
                st.success(f"Text: {len(features['text'])} features")
            else:
                st.warning("Text: Not extracted")
        
        # Export options
        st.markdown("---")
        st.subheader("Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Download JSON", use_container_width=True):
                json_str = json.dumps(
                    {k: str(v) for k, v in results.items()},
                    indent=2
                )
                st.download_button(
                    "Click to Download",
                    json_str,
                    "emotion_results.json",
                    "application/json"
                )
        
        with col2:
            col2a, col2b = st.columns(2)
            
            with col2a:
                if st.button("Download Report", use_container_width=True):
                    with st.spinner("Generating report..."):
                        try:
                            report_text = generate_comprehensive_report(results)
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"emotion_report_{timestamp}.txt"
                            
                            st.download_button(
                                "📥 Download Report",
                                report_text,
                                filename,
                                "text/plain",
                                key="download_report"
                            )
                            st.success("Report generated! Click the download button above.")
                        except Exception as e:
                            st.error(f"Error generating report: {e}")
                            logger.error(f"Report generation error: {e}", exc_info=True)
            
            with col2b:
                if st.button("🤖 AI-Commented Report", use_container_width=True):
                    with st.spinner("Generating AI-commented report with LLaMA3..."):
                        try:
                            ai_report_text = generate_ai_commented_report(results)
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"emotion_report_ai_{timestamp}.txt"
                            
                            st.download_button(
                                "📥 Download AI Report",
                                ai_report_text,
                                filename,
                                "text/plain",
                                key="download_ai_report"
                            )
                            st.success("AI-commented report generated! Click the download button above.")
                        except Exception as e:
                            st.error(f"Error generating AI report: {e}")
                            logger.error(f"AI report generation error: {e}", exc_info=True)
        
        with col3:
            if st.button("Clear Results", use_container_width=True):
                del st.session_state['results']
                st.experimental_rerun()
