"""Results Tab"""

import streamlit as st
import pandas as pd
import json
import os


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
        
        # Prediction results
        if 'prediction' in results:
            st.subheader("Detected Emotion")
            
            pred = results['prediction']
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown(f"### **{pred['predicted_emotion'].upper()}**")
                st.metric("Confidence", f"{pred['confidence']:.1%}")
            
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
        
        # Transcription
        if 'transcription' in results:
            st.subheader("Video Transcript")
            transcript_text = results['transcription'].get('text', 'No speech detected')
            
            if transcript_text and transcript_text != 'No speech detected':
                st.text_area(
                    "Full Transcript",
                    transcript_text,
                    height=150
                )
                
                # Show word count
                word_count = len(transcript_text.split())
                st.caption(f"Total words: {word_count}")
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
            if st.button("Download Report", use_container_width=True):
                st.info("Report generation coming soon!")
        
        with col3:
            if st.button("Clear Results", use_container_width=True):
                del st.session_state['results']
                st.experimental_rerun()
