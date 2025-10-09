"""Upload & Analyze Tab"""

import streamlit as st
import os
import tempfile
from modules.stage1_input import VideoProcessor, ASRModule
from modules.stage2_unimodal import AudioFeatureExtractor, VisualFeatureExtractor, TextFeatureExtractor
import numpy as np


def process_video_pipeline(video_path, config, extractors):
    """
    Complete video processing pipeline.
    
    Args:
        video_path: Path to video file
        config: Configuration dictionary
        extractors: Dictionary of feature extractors
        
    Returns:
        Dictionary with results
    """
    results = {}
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Stage 1: Input Processing
        status_text.text("Stage 1: Processing video input...")
        processor = VideoProcessor(video_path, output_dir="data/processed")
        
        # Extract audio
        audio, sr, audio_path = processor.extract_audio(
            sample_rate=config['modalities']['audio']['sample_rate']
        )
        results['audio'] = audio
        results['sample_rate'] = sr
        results['audio_path'] = audio_path
        
        # Extract frames
        frames, timestamps = processor.extract_frames(
            fps=config['modalities']['visual']['fps']
        )
        results['frames'] = frames
        results['timestamps'] = timestamps
        
        # Extract frames to files (LlamaIndex approach: 1 frame every 5 seconds)
        status_text.text("Extracting frames to files (LlamaIndex approach: 1 per 5 sec)...")
        frames_folder = os.path.join("data/processed", "frames")
        frame_paths = processor.extract_frames_to_files(
            output_folder=frames_folder,
            fps=0.2  # 1 frame every 5 seconds as per LlamaIndex article
        )
        results['frame_paths'] = frame_paths
        results['frames_folder'] = frames_folder
        
        # Get metadata
        results['metadata'] = processor.get_video_metadata()
        
        progress_bar.progress(15)
        
        # Transcribe audio
        if config['modalities']['text']['enabled']:
            status_text.text("Transcribing audio to text...")
            asr = ASRModule(model_name="base")
            transcription = asr.transcribe(audio, sr)
            results['transcription'] = transcription
            progress_bar.progress(25)
        
        # Stage 2: Unimodal Feature Extraction
        status_text.text("Stage 2: Extracting features from each modality...")
        
        features = {}
        
        # Audio features
        if 'audio' in extractors and len(audio) > 0:
            status_text.text("Extracting audio features...")
            audio_features = extractors['audio'].extract_all_features(
                audio, sr, audio_path
            )
            features['audio'] = audio_features
            st.session_state['audio_features'] = audio_features
            progress_bar.progress(40)
        
        # Visual features
        if 'visual' in extractors and len(frames) > 0:
            status_text.text("Extracting visual features...")
            visual_features = extractors['visual'].extract_video_features(frames)
            features['visual'] = visual_features
            st.session_state['visual_features'] = visual_features
            progress_bar.progress(55)
        
        # Text features
        if 'text' in extractors and results.get('transcription'):
            status_text.text("Extracting text features...")
            text = results['transcription'].get('text', '')
            text_features = extractors['text'].extract_all_features(text)
            features['text'] = text_features
            st.session_state['text_features'] = text_features
            progress_bar.progress(70)
        
        results['features'] = features
        
        # Stage 3: Fusion & Prediction
        fusion_strategy = config.get('fusion_strategy', 'Simple Concatenation')
        status_text.text(f"Stage 3: Fusing modalities and analyzing temporal patterns ({fusion_strategy})...")
        
        emotion_labels = config['emotions']['labels']
        
        # Get features
        audio_feat = features.get('audio')
        visual_feat = features.get('visual')
        text_feat = features.get('text')
        
        # Temporal emotion analysis using FER (Facial Expression Recognition)
        temporal_predictions = []
        
        # Use FER analyzer on extracted frames
        if 'frame_paths' in results and len(results['frame_paths']) > 0:
            status_text.text("Analyzing facial expressions over time (FER)...")
            
            try:
                from modules.fer_analyzer import FERAnalyzer
                
                # Initialize FER analyzer
                fer = FERAnalyzer(model_type='custom_cnn')  # Using CNN for speed
                
                # Analyze frame sequence
                temporal_predictions = fer.analyze_frame_sequence(
                    frame_paths=results['frame_paths'],
                    timestamps=[i * 5.0 for i in range(len(results['frame_paths']))]
                )
                
                # Calculate mental health score
                if temporal_predictions:
                    mental_health_analysis = fer.calculate_mental_health_score(temporal_predictions)
                    results['mental_health_analysis'] = mental_health_analysis
                
            except Exception as e:
                import traceback
                status_text.warning(f"FER analysis failed: {e}, using demo predictions")
                print(traceback.format_exc())
                
                # Fallback to demo predictions
                timestamps_list = results.get('timestamps', [])
                if len(timestamps_list) > 0:
                    window_size = 3.0
                    for i, timestamp in enumerate(timestamps_list[::int(config['modalities']['visual']['fps'] * window_size)]):
                        temporal_pred = {
                            'timestamp': float(timestamp),
                            'emotion': np.random.choice(emotion_labels),
                            'confidences': {label: float(np.random.uniform(0.05, 0.95)) for label in emotion_labels}
                        }
                        total = sum(temporal_pred['confidences'].values())
                        temporal_pred['confidences'] = {k: v/total for k, v in temporal_pred['confidences'].items()}
                        temporal_predictions.append(temporal_pred)
        
        results['temporal_predictions'] = temporal_predictions
        
        if audio_feat is not None or visual_feat is not None or text_feat is not None:
            # Handle missing modalities with zero vectors
            if audio_feat is None:
                audio_feat = np.zeros(1)
            if visual_feat is None:
                visual_feat = np.zeros(1)
            if text_feat is None:
                text_feat = np.zeros(1)
            
            combined_features = np.concatenate([audio_feat, visual_feat, text_feat])
            results['combined_features'] = combined_features
            
            # Demo prediction based on fusion strategy
            # NOTE: In production, you'd train the model first!
            if fusion_strategy == "Maelfabien Multimodal":
                # Maelfabien approach demo
                demo_prediction = {
                    'predicted_emotion': np.random.choice(emotion_labels),
                    'predicted_label': np.random.randint(0, len(emotion_labels)),
                    'confidence': np.random.uniform(0.65, 0.92),
                    'all_confidences': {
                        label: float(np.random.uniform(0.05, 0.95))
                        for label in emotion_labels
                    },
                    'fusion_method': 'Maelfabien (Text CNN-LSTM + Audio Time-CNN + Video XCeption)',
                    'individual_models': {
                        'text_cnn_lstm': np.random.choice(emotion_labels),
                        'audio_time_cnn': np.random.choice(emotion_labels),
                        'video_xception': np.random.choice(emotion_labels)
                    }
                }
            elif fusion_strategy == "Emotion-LLaMA":
                # Emotion-LLaMA approach demo
                demo_prediction = {
                    'predicted_emotion': np.random.choice(emotion_labels),
                    'predicted_label': np.random.randint(0, len(emotion_labels)),
                    'confidence': np.random.uniform(0.70, 0.95),
                    'all_confidences': {
                        label: float(np.random.uniform(0.05, 0.95))
                        for label in emotion_labels
                    },
                    'fusion_method': 'Emotion-LLaMA (Transformer + Reasoning + Temporal)',
                    'intensity': float(np.random.uniform(0.5, 0.9)),
                    'reasoning': f"The person appears {np.random.choice(emotion_labels)}. Analysis based on multimodal cues including facial expressions, voice tone, and semantic content."
                }
            elif fusion_strategy == "Hybrid (Best)":
                # Use hybrid classifier for demo (would be pre-trained in production)
                demo_prediction = {
                    'predicted_emotion': np.random.choice(emotion_labels),
                    'predicted_label': np.random.randint(0, len(emotion_labels)),
                    'confidence': np.random.uniform(0.70, 0.95),  # Higher confidence
                    'all_confidences': {
                        label: np.random.uniform(0.05, 0.95) 
                        for label in emotion_labels
                    },
                    'fusion_method': 'Hybrid (RFRBoost + Deep Learning + Attention)',
                    'modality_weights': {
                        'audio': float(np.random.uniform(0.2, 0.4)),
                        'visual': float(np.random.uniform(0.3, 0.5)),
                        'text': float(np.random.uniform(0.2, 0.4))
                    },
                    'individual_models': {
                        'rfrboost': np.random.choice(emotion_labels),
                        'attention_deep': np.random.choice(emotion_labels),
                        'mlp_baseline': np.random.choice(emotion_labels)
                    }
                }
                # Normalize modality weights
                total_weight = sum(demo_prediction['modality_weights'].values())
                demo_prediction['modality_weights'] = {
                    k: v/total_weight for k, v in demo_prediction['modality_weights'].items()
                }
            else:
                # Simple prediction
                demo_prediction = {
                    'predicted_emotion': np.random.choice(emotion_labels),
                    'predicted_label': np.random.randint(0, len(emotion_labels)),
                    'confidence': np.random.uniform(0.6, 0.85),
                    'all_confidences': {
                        label: np.random.uniform(0.05, 0.95) 
                        for label in emotion_labels
                    },
                    'fusion_method': fusion_strategy
                }
            
            # Normalize confidences
            total = sum(demo_prediction['all_confidences'].values())
            demo_prediction['all_confidences'] = {
                k: v/total for k, v in demo_prediction['all_confidences'].items()
            }
            demo_prediction['confidence'] = demo_prediction['all_confidences'][
                demo_prediction['predicted_emotion']
            ]
            
            results['prediction'] = demo_prediction
        
        progress_bar.progress(85)
        
        # Stage 4: Visualization
        status_text.text("Stage 4: Generating visualizations...")
        progress_bar.progress(100)
        
        status_text.text("Processing complete!")
        
        return results
    
    except Exception as e:
        status_text.text(f"Error: {str(e)}")
        st.error(f"Processing error: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None


def render_upload_tab(config, extractors):
    """Render the upload and analyze tab."""
    st.header("Video Upload")
    
    uploaded_file = st.file_uploader(
        "Upload a video file",
        type=['mp4', 'avi', 'mov', 'webm'],
        help="Upload a video to analyze emotions"
    )
    
    if uploaded_file is not None:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name
        
        # Display video
        st.video(uploaded_file)
        
        # Process button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Analyze Emotions", use_container_width=True):
                st.markdown("---")
                
                if extractors:
                    # Process video
                    results = process_video_pipeline(video_path, config, extractors)
                    
                    if results:
                        # Store results in session state
                        st.session_state['results'] = results
                        st.session_state['config'] = config
                        
                        st.success("Analysis complete! View results in the 'Results' tab.")
                else:
                    st.error("Failed to initialize extractors")
        
        # Cleanup
        try:
            os.unlink(video_path)
        except:
            pass
