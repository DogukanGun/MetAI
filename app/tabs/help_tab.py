"""Help & Information Tab"""

import streamlit as st


def render_help_tab(config):
    """Render the help and information tab."""
    st.header("System Information")
    
    st.markdown("""
    ### How It Works
    
    The system processes videos through 4 stages:
    
    **Stage 1: Input Processing**
    - Extracts audio stream
    - Extracts video frames (standard rate for analysis)
    - Saves frames to files (LlamaIndex approach: 1 frame per 5 seconds)
    - Transcribes speech to text
    
    **Stage 2: Unimodal Feature Extraction & FER Analysis**
    - **Audio**: Prosodic, spectral, and voice quality features
    - **Visual**: Facial landmarks, action units, head pose
    - **Text**: Semantic embeddings, sentiment, lexical features
    - **FER (Facial Expression Recognition)**: Time-series emotion analysis from extracted frames
      - Uses Swin Transformer / Custom CNN models
      - Detects 7 emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
      - Generates mental health score based on emotion distribution
    
    **Stage 3: Multimodal Fusion**
    - **Hybrid (Recommended)**: Combines RFRBoost + Deep Learning + Attention
    - **RFRBoost Only**: Uses gradient boosting with SWIM features
    - **Simple Concatenation**: Baseline feature combination
    
    **Stage 4: Results & Visualization**
    - Displays detected emotions
    - Provides confidence scores
    - Exports results in multiple formats
    
    ### Fusion Strategies
    
    **Hybrid Fusion** (Best Performance):
    - Ensemble of 3 models: RFRBoost (40%) + Attention+Deep (35%) + MLP (25%)
    - Learns modality importance weights dynamically
    - Shows which models agree/disagree
    - Provides interpretable predictions
    
    **RFRBoost**:
    - Random Feature Representation Boosting
    - Gradient-based boosting with SWIM features
    - Excellent for structured/tabular features
    - Robust to overfitting
    
    ### Supported Emotions
    """)
    
    emotion_labels = config['emotions']['labels']
    emotion_colors = config['emotions']['colors']
    
    cols = st.columns(len(emotion_labels))
    for i, (emotion, col) in enumerate(zip(emotion_labels, cols)):
        with col:
            color = emotion_colors.get(emotion, '#95a5a6')
            st.markdown(f"""
            <div style="background-color: {color}; padding: 10px; border-radius: 5px; text-align: center;">
                <strong>{emotion.upper()}</strong>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    ### Requirements
    
    - Video files in MP4, AVI, MOV, or WebM format
    - Clear audio track for speech recognition
    - Visible faces for facial analysis
    - Reasonable video quality
    
    ### Tips for Best Results
    
    - Use videos with clear facial expressions
    - Ensure good audio quality
    - Avoid excessive background noise
    - Keep video duration under 2 minutes for faster processing
    
    ### Model Training
    
    **Note**: The current demo uses random predictions for demonstration purposes.
    
    To train the model on your own data:
    
    1. **Prepare labeled dataset**: Collect videos with emotion labels
    2. **Extract features**: Use the feature extractors on your dataset
    3. **Train model**: Use HybridMultimodalClassifier.fit()
    4. **Save model**: Save the trained model to pretrained/ directory
    5. **Load for inference**: Load the model in the app for real predictions
    
    See the documentation for detailed training examples.
    
    ### Technical Details
    
    **Feature Dimensions**:
    - Audio: ~200-300 features (MFCCs, prosody, spectral, OpenSMILE)
    - Visual: ~300-500 features (landmarks, action units, geometric)
    - Text: ~400-800 features (SBERT embeddings, sentiment, lexical)
    
    **Hybrid Model Components**:
    - **GradientRFRBoostClassifier**: From rfr/models directory
    - **Multi-head Attention**: 4 attention heads across modalities
    - **Deep Networks**: Multi-layer perceptrons with dropout
    - **Ensemble Voting**: Weighted combination of all models
    
    ### References
    
    This system integrates concepts from:
    - Random Feature Representation Boosting (this repository)
    - USDM (Unsupervised Self-Distillation for Multimodal Emotion Recognition)
    - Emotion-LLaMA
    - Various multimodal emotion recognition papers
    """)
