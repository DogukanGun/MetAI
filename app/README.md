# Multimodal Emotion Recognition System

A comprehensive system for detecting emotions from videos using audio, visual, and text modalities.

**Status**: Complete and ready for use  
**Architecture**: 4-stage pipeline (Input -> Features -> Fusion -> Export)  
**Modalities**: Audio (prosody, spectral) + Visual (facial expressions) + Text (semantics)  

## Overview

This system analyzes videos to recognize human emotions by combining three modalities:
- **Audio**: Prosodic, spectral, and voice quality features
- **Visual**: Facial expressions, action units, and head pose
- **Text**: Semantic embeddings, sentiment, and linguistic features

### Available Fusion Approaches

The system includes multiple state-of-the-art fusion approaches:

1. **RFRBoost**: Random Feature Representation Boosting with SWIM features
2. **Hybrid**: Ensemble combining RFRBoost + Deep Learning + Attention  
   - RFRBoost (40%) + Attention+Deep (35%) + MLP Baseline (25%)
3. **Maelfabien Multimodal**: Implementation from [maelfabien/Multimodal-Emotion-Recognition](https://github.com/maelfabien/Multimodal-Emotion-Recognition)
   - Text: Word2Vec + CNN + LSTM  
   - Audio: Time-Distributed CNN on mel-spectrograms  
   - Video: XCeption for facial emotion recognition
4. **Emotion-LLaMA**: Implementation inspired by [Emotion-LLaMA](https://github.com/ZebangCheng/Emotion-LLaMA)
   - Transformer-based multi-modal encoder  
   - Emotion reasoning and explanation generation  
   - Temporal emotion tracking with smoothing

## System Architecture

The system follows a 4-stage pipeline:

```
Stage 1: Input Processing
    ├── Video ingestion
    ├── Audio extraction  
    ├── Frame extraction
    └── Speech-to-text transcription

Stage 2: Unimodal Processing
    ├── Audio features (MFCCs, prosody, spectral, OpenSMILE)
    ├── Visual features (landmarks, action units, geometric)
    └── Text features (embeddings, sentiment, lexical)

Stage 3: Multimodal Fusion
    ├── RFRBoost (original implementation)
    ├── Hybrid (ensemble of 3 models)
    ├── Maelfabien (3 specialized models)
    └── Emotion-LLaMA (transformer + reasoning)

Stage 4: Results & Export
    ├── Temporal emotion analysis
    ├── Video transcript
    ├── Visualizations
    └── Export (JSON, CSV, PDF)
```

## Key Features

### Frame Extraction (LlamaIndex Approach)
- Automatic frame extraction at 0.2 FPS (1 frame every 5 seconds)
- Based on [LlamaIndex multimodal RAG](https://www.llamaindex.ai/blog/multimodal-rag-for-advanced-video-processing-with-llamaindex-lancedb-33be4804822e)
- Frames saved as individual PNG files with sequential naming (frame0001.png, frame0002.png, etc.)
- Visual preview of extracted frames in results tab
- Suitable for multimodal RAG and video analysis pipelines

### Facial Expression Recognition (FER) for Mental Health
- Based on [FER for Mental Health Detection](https://github.com/mujiyantosvc/Facial-Expression-Recognition-FER-for-Mental-Health-Detection-)
- Analyzes extracted frames using Swin Transformer or Custom CNN models
- Detects 7 emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- Generates mental health score (0-100) based on emotion distribution
- Time-series emotion tracking with confidence scores
- Real-time face detection using Haar Cascade
- Emotion distribution analysis (positive vs negative emotions)

### Temporal Emotion Analysis
- Time-series emotion tracking at 3-second intervals
- Interactive plotly timeline showing all emotions over time
- Dominant emotion timeline with timestamps and confidence scores
- Example: "43.0s: Happy (87.3%)"

### Video Transcription
- Automatic speech recognition using Whisper
- Full video transcript with word count
- Integrated with text emotion analysis

### Multiple Model Options
- Choose between 5 different fusion strategies
- Each with unique strengths and characteristics
- Compare model outputs and confidence levels

### Rich Visualizations
- Interactive emotion timeline graphs
- Confidence distribution charts
- Temporal emotion heatmaps
- Model comparison views

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- FFmpeg (for video processing)

### Setup

1. Navigate to app directory:
```bash
cd app/
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download NLTK data (automatic on first run):
```python
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
```

4. Run the application:
```bash
streamlit run app.py
```

5. Open browser at `http://localhost:8501`

## Project Structure

```
app/
├── app.py                          # Main application entry point
├── requirements.txt                # Python dependencies
├── config/
│   ├── config.yaml                 # System configuration
│   └── emotion_labels.json         # Emotion class definitions
├── tabs/                           # Modular UI tabs
│   ├── upload_tab.py              # Upload & analyze interface
│   ├── results_tab.py             # Results display
│   └── help_tab.py                # Help & information
├── modules/
│   ├── stage1_input/              # Video processing & ASR
│   ├── stage2_unimodal/           # Feature extractors
│   ├── stage3_fusion/             # Multimodal fusion strategies
│   └── stage4_output/             # Metrics & visualization
├── models/                        # Pre-trained model implementations
│   ├── maelfabien_model.py       # Maelfabien approach
│   └── emotion_llama_model.py    # Emotion-LLaMA approach
├── utils/                         # Helper functions
└── data/                          # Data storage
    ├── raw/                       # Input videos
    ├── processed/                 # Extracted features
    └── results/                   # Output files
```

## Usage

### Web Interface

1. **Launch the app**: `streamlit run app.py`
2. **Select fusion strategy**: Choose from dropdown in sidebar
   - Hybrid (Best) - Recommended for best accuracy
   - RFRBoost Only - Fast and robust
   - Maelfabien Multimodal - Specialized models
   - Emotion-LLaMA - With reasoning
   - Simple Concatenation - Baseline
3. **Configure modalities**: Enable/disable audio, visual, text
4. **Upload video**: Use file uploader (MP4, AVI, MOV, WebM)
5. **Analyze**: Click "Analyze Emotions" button
6. **View results**: 
   - Temporal emotion timeline
   - Video transcript
   - Confidence distribution
   - Model-specific analysis
7. **Export**: Download results as JSON or PDF

### Results Output

The system provides:

1. **Temporal Emotion Analysis**
   - Interactive timeline graph
   - Emotion distribution over time
   - Dominant emotion at each timestamp
   - Confidence scores for all emotions

2. **Video Transcript**
   - Full text transcription
   - Word count
   - Integration with text emotion analysis

3. **Model-Specific Outputs**
   - **Hybrid**: Modality weights + Model agreement
   - **Maelfabien**: Individual model predictions (Text/Audio/Video)
   - **Emotion-LLaMA**: Reasoning explanation + Intensity score

4. **Feature Extraction Summary**
   - Audio features count
   - Visual features count
   - Text features count

5. **Extracted Frames** (LlamaIndex Approach)
   - Visual preview of extracted frames
   - Frame paths and timestamps
   - 1 frame per 5 seconds saved as PNG files

6. **Mental Health Analysis** (FER-based)
   - Mental health score (0-100 scale)
   - Emotion distribution breakdown
   - Positive vs negative emotion percentages
   - Dominant emotion identification
   - Status interpretation with recommendations

### Programmatic Usage

#### FER Analysis Example

```python
from modules.fer_analyzer import FERAnalyzer

# Initialize FER analyzer
fer = FERAnalyzer(model_type='custom_cnn')  # or 'swin_transformer'

# Analyze frame sequence
temporal_predictions = fer.analyze_frame_sequence(
    frame_paths=['frame0001.png', 'frame0002.png', ...],
    timestamps=[0.0, 5.0, 10.0, ...]
)

# Calculate mental health score
mental_health = fer.calculate_mental_health_score(temporal_predictions)
print(f"Mental Health Score: {mental_health['mental_health_score']:.1f}/100")
print(f"Dominant Emotion: {mental_health['dominant_emotion']}")
```

#### Frame Extraction Example

```python
from modules.stage1_input import VideoProcessor

# Initialize processor
processor = VideoProcessor("path/to/video.mp4")

# Extract frames to files (1 frame every 5 seconds)
frame_paths = processor.extract_frames_to_files(
    output_folder="./extracted_frames",
    fps=0.2  # 0.2 FPS = 1 frame per 5 seconds
)

print(f"Extracted {len(frame_paths)} frames")
# Output files: frame0001.png, frame0002.png, ...
```

Run the example script:
```bash
python examples/frame_extraction_example.py path/to/video.mp4
```

## Configuration

Edit `config/config.yaml` to customize:

### Modalities
```yaml
modalities:
  audio:
    enabled: true
    sample_rate: 16000
    n_mfcc: 40
  visual:
    enabled: true
    fps: 5
    extract_action_units: true
  text:
    enabled: true
    asr_model: "openai/whisper-base"
```

### RFRBoost Parameters
```yaml
rfrboost:
  n_layers: 6
  hidden_dim: 256
  boost_lr: 0.5
  feature_type: "SWIM"
```

### Emotions
```yaml
emotions:
  labels: ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']
```

## Model Implementations

### 1. RFRBoost (Original)
From this repository's core implementation:
- Gradient-based boosting with SWIM features
- Excellent for structured/tabular features
- Robust to overfitting

### 2. Hybrid Fusion
Custom ensemble combining:
- RFRBoost (40%): Tabular learning
- Attention+Deep (35%): Modality fusion with multi-head attention
- MLP Baseline (25%): Simple patterns

### 3. Maelfabien Multimodal
Implementation from: https://github.com/maelfabien/Multimodal-Emotion-Recognition

**Text Model**:
- 300-dim Word2Vec embeddings
- 3x Conv1D blocks (128, 256, 512 filters)
- 3x LSTM layers (180 units each)
- Dense classification head

**Audio Model**:
- Time-Distributed CNN on mel-spectrograms
- 4x Local Feature Learning Blocks (LFLBs)
- 2x LSTM layers for temporal context
- Captures prosody and acoustic patterns

**Video Model**:
- XCeption architecture with DepthWise Separable Convolutions
- Optimized for 48x48 facial images
- Class activation maps for interpretability

**Fusion**: Weighted ensemble (33.3% each modality)

### 4. Emotion-LLaMA
Implementation inspired by: https://github.com/ZebangCheng/Emotion-LLaMA

**Encoder**:
- Multi-modal transformer with 6 layers
- Emotion-specific prompt embeddings
- Cross-modal attention mechanisms

**Reasoning Module**:
- LSTM-based decoder for explanation generation
- Natural language emotion reasoning
- Explainable predictions

**Temporal Tracker**:
- Temporal LSTM for emotion sequence modeling
- Transition smoothing with Conv1D
- Consistent emotion trajectories

**Features**:
- Emotion intensity estimation
- Multi-task learning (classification + reasoning)
- Context-aware emotion understanding

### 5. FER (Facial Expression Recognition)
Implementation inspired by: https://github.com/mujiyantosvc/Facial-Expression-Recognition-FER-for-Mental-Health-Detection-

**Swin Transformer Model**:
- Hierarchical transformer architecture
- Patch size: 4x4
- Depths: [2, 2, 6, 2] layers
- Attention heads: [3, 6, 12, 24]
- Optimized for 48x48 facial images
- Enhanced dropout and layer-wise unfreezing

**Custom CNN Model**:
- 4 convolutional blocks (64, 128, 256, 512 filters)
- Batch normalization after each conv layer
- MaxPooling for spatial reduction
- 3 fully connected layers (1024, 512, 7 outputs)
- Dropout regularization (0.3)
- Lightweight and fast for real-time analysis

**Face Detection**:
- Haar Cascade classifier for face localization
- Automatic face cropping with padding
- Handles multi-face scenarios (selects largest)

**Mental Health Scoring**:
- Score range: 0-100
- Formula: `50 + (positive% - negative%) / 2`
- Positive emotions: Happy, Surprise, Neutral
- Negative emotions: Angry, Disgust, Fear, Sad
- Status interpretation:
  - 70-100: Good mental health
  - 50-70: Moderate
  - 30-50: Concerning
  - 0-30: At risk (recommend consultation)

**Emotion Categories** (FER2013 standard):
1. Angry
2. Disgust
3. Fear
4. Happy
5. Sad
6. Surprise
7. Neutral

## Technical Details

### Feature Dimensions

**Audio Features** (100-300 dimensions):
- 40 MFCCs with delta and delta-delta
- Prosodic features (pitch, energy, speaking rate)
- Spectral features (centroid, rolloff, bandwidth)
- Voice quality (HNR proxy)
- OpenSMILE eGeMAPS (88 features)

**Visual Features** (100-500 dimensions):
- 468 facial landmarks (MediaPipe)
- Geometric features (eye/mouth aspect ratios)
- Head pose estimation (yaw, pitch, roll)
- Facial Action Units (FAUs via Py-feat)
- Temporal aggregations (mean, std, max, min)

**Text Features** (300-768 dimensions):
- SBERT embeddings (384-dim, all-MiniLM-L6-v2)
- Sentiment scores (VADER)
- Lexical features (word count, sentence length)
- Emotion keyword matching
- Part-of-speech statistics

### Performance Expectations

Expected performance on standard datasets (7-class emotion recognition):

| Approach | Accuracy | F1-Score | Processing Time |
|----------|----------|----------|-----------------|
| Simple Concatenation | 55-60% | 0.52-0.58 | Fast (~1min) |
| RFRBoost Only | 60-70% | 0.58-0.68 | Medium (~5min) |
| Maelfabien | 62-72% | 0.60-0.70 | Medium (~8min) |
| Emotion-LLaMA | 65-75% | 0.63-0.73 | Medium (~10min) |
| **Hybrid** | **65-75%** | **0.63-0.73** | Slower (~15min) |

*Note: Results vary by dataset, feature quality, and training data size*

## Environment Variables

**No environment variables required!** All models use free, open-source implementations:
- Whisper (speech recognition) - Downloads from OpenAI's public repo
- SBERT (text embeddings) - Downloads from HuggingFace public models
- MediaPipe (face detection) - Google's free library
- VADER (sentiment) - Included in package

Optional environment variables for advanced configurations are documented but not required for basic operation.

## Supported Datasets

The system has been tested with:
- **RAVDESS**: Ryerson Audio-Visual Database
- **IEMOCAP**: Interactive Emotional Dyadic Motion Capture
- **CREMA-D**: Crowd-sourced Emotional Multimodal Actors Dataset
- **MELD**: Multimodal EmotionLines Dataset

## Troubleshooting

### Common Issues

**FFmpeg not found**:
```bash
# macOS
brew install ffmpeg

# Ubuntu
sudo apt-get install ffmpeg
```

**Out of memory errors**:
- Reduce video resolution
- Lower FPS (e.g., fps: 3 instead of 5)
- Process shorter video segments

**No faces detected**:
- Ensure faces are clearly visible
- Check lighting conditions
- Verify video quality

## References

### Implementations
- **RFRBoost**: Original implementation from `/rfr/models/`
- **Maelfabien**: https://github.com/maelfabien/Multimodal-Emotion-Recognition
- **Emotion-LLaMA**: https://github.com/ZebangCheng/Emotion-LLaMA

### Related Work
- **SWIM Features**: Self-Organizing Invariant Mappings
- **OpenSMILE**: eGeMAPS feature set for audio
- **MediaPipe**: Face mesh for facial landmarks
- **Whisper**: OpenAI speech recognition

## License

This project builds upon the RFRBoost implementation and follows its licensing terms.

## Acknowledgments

- RFRBoost implementation from the base repository
- Maelfabien's multimodal emotion recognition approach
- Emotion-LLaMA for transformer-based emotion understanding
- OpenSMILE toolkit for audio features
- MediaPipe for facial landmarks
- OpenAI Whisper for speech recognition
- MELD dataset creators for multimodal emotion research

## Contact

For questions or issues, please open an issue on the repository.

---

**Built with PyTorch, Streamlit, and Multiple State-of-the-Art Approaches**