# Multimodal Emotion Recognition System - Usage Guide

## Table of Contents
1. [Installation](#installation)
2. [Basic Usage](#basic-usage)
3. [Advanced Configuration](#advanced-configuration)
4. [Training Custom Models](#training-custom-models)
5. [API Reference](#api-reference)
6. [Examples](#examples)
7. [Troubleshooting](#troubleshooting)

---

## Installation

### Step 1: Environment Setup

```bash
# Create virtual environment (recommended)
python -m venv emotion_env
source emotion_env/bin/activate  # On Windows: emotion_env\Scripts\activate

# Navigate to app directory
cd app/
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Install System Dependencies

**FFmpeg** (required for video processing):

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get update
sudo apt-get install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html and add to PATH
```

### Step 4: Verify Installation

```bash
python -c "import streamlit; import torch; import librosa; print('✅ Installation successful!')"
```

---

## Basic Usage

### Web Interface (Recommended for Beginners)

1. **Start the application**:
```bash
streamlit run app.py
```

2. **Access in browser**: Navigate to `http://localhost:8501`

3. **Upload video**:
   - Click "Browse files" in the Upload tab
   - Select a video (MP4, AVI, MOV, WebM)
   - Wait for upload to complete

4. **Configure settings** (optional):
   - Use sidebar to enable/disable modalities
   - Adjust RFRBoost parameters
   - Default settings work well for most cases

5. **Analyze**:
   - Click "🚀 Analyze Emotions"
   - Wait for processing (20-30s per minute of video)
   - View progress bar and status updates

6. **View results**:
   - Switch to "Results" tab
   - See detected emotion and confidence scores
   - View feature extraction statistics
   - Read transcription (if available)

7. **Export**:
   - Click "Download JSON" for structured data
   - Click "Download Report" for detailed analysis
   - Results saved to `data/results/`

### Command Line Interface

Process a single video:

```bash
python -c "
from modules.stage1_input import VideoProcessor
from modules.stage2_unimodal import AudioFeatureExtractor
from utils.helpers import load_config

config = load_config('config/config.yaml')
processor = VideoProcessor('path/to/video.mp4')
audio, sr, path = processor.extract_audio()
print(f'Processed: {len(audio)} samples at {sr} Hz')
"
```

---

## Advanced Configuration

### config/config.yaml Structure

```yaml
system:
  name: "Multimodal Emotion Recognition System"
  version: "1.0.0"

modalities:
  audio:
    enabled: true              # Enable/disable audio processing
    sample_rate: 16000         # Audio sample rate in Hz
    window_size: 3.0           # Feature window in seconds
    hop_length: 1.5            # Hop length in seconds
    n_mfcc: 40                 # Number of MFCC coefficients
    extract_opensmile: true    # Use OpenSMILE features
  
  visual:
    enabled: true              # Enable/disable visual processing
    fps: 5                     # Frames per second to extract
    face_detection: "mediapipe"  # Face detection method
    extract_action_units: true   # Extract facial action units
    extract_landmarks: true      # Extract facial landmarks
    
  text:
    enabled: true              # Enable/disable text processing
    asr_model: "openai/whisper-base"  # ASR model name
    embedding_model: "all-MiniLM-L6-v2"  # Text embedding model
    extract_sentiment: true    # Extract sentiment features

fusion:
  strategy: "early"           # Fusion strategy: early, late, attention
  model: "rfrboost"           # Fusion model type

rfrboost:
  n_layers: 6                 # Number of boosting layers
  hidden_dim: 256             # Hidden layer dimension
  randfeat_xt_dim: 512        # Random features from current representation
  randfeat_x0_dim: 512        # Random features from input
  boost_lr: 0.5               # Boosting learning rate
  feature_type: "SWIM"        # Random feature type (SWIM, RFF, ORF)
  upscale_type: "SWIM"        # Upscaling layer type
  activation: "tanh"          # Activation function
  use_batchnorm: true         # Use batch normalization
  do_linesearch: true         # Optimize step size
  l2_cls: 0.001               # L2 regularization for classifier
  l2_ghat: 0.001              # L2 regularization for gradient estimator

emotions:
  labels: ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']
  colors:
    neutral: "#95a5a6"
    happy: "#f1c40f"
    sad: "#3498db"
    angry: "#e74c3c"
    fear: "#9b59b6"
    disgust: "#16a085"
    surprise: "#e67e22"

output:
  save_json: true
  save_csv: true
  save_video: false
  save_report: true
```

### Customizing for Specific Use Cases

**High Quality Processing** (slower, more features):
```yaml
modalities:
  audio:
    n_mfcc: 40
    extract_opensmile: true
  visual:
    fps: 10
    extract_action_units: true
  text:
    asr_model: "openai/whisper-large"
```

**Fast Processing** (faster, fewer features):
```yaml
modalities:
  audio:
    n_mfcc: 13
    extract_opensmile: false
  visual:
    fps: 3
    extract_action_units: false
  text:
    asr_model: "openai/whisper-tiny"
```

**Audio-Only Mode**:
```yaml
modalities:
  audio:
    enabled: true
  visual:
    enabled: false
  text:
    enabled: true  # Still useful for audio transcription
```

---

## Training Custom Models

### Preparing Your Dataset

1. **Organize videos**:
```
data/raw/
├── train/
│   ├── happy/
│   │   ├── video1.mp4
│   │   └── video2.mp4
│   ├── sad/
│   └── angry/
└── test/
    ├── happy/
    ├── sad/
    └── angry/
```

2. **Extract features**:
```python
import os
import torch
import numpy as np
from pathlib import Path
from modules.stage1_input import VideoProcessor
from modules.stage2_unimodal import AudioFeatureExtractor, VisualFeatureExtractor, TextFeatureExtractor
from utils.helpers import load_config

config = load_config("config/config.yaml")

# Initialize extractors
audio_extractor = AudioFeatureExtractor(config['modalities']['audio'])
visual_extractor = VisualFeatureExtractor(config['modalities']['visual'])
text_extractor = TextFeatureExtractor(config['modalities']['text'])

def extract_features_from_video(video_path):
    """Extract all features from a video."""
    processor = VideoProcessor(video_path)
    
    # Extract modalities
    audio, sr, audio_path = processor.extract_audio()
    frames, timestamps = processor.extract_frames()
    
    # Extract features
    audio_feat = audio_extractor.extract_all_features(audio, sr, audio_path)
    visual_feat = visual_extractor.extract_video_features(frames)
    text_feat = np.zeros(413)  # Placeholder if no ASR
    
    # Concatenate
    features = np.concatenate([audio_feat, visual_feat, text_feat])
    return features

# Process dataset
emotion_to_label = {'neutral': 0, 'happy': 1, 'sad': 2, 'angry': 3, 
                   'fear': 4, 'disgust': 5, 'surprise': 6}

train_features = []
train_labels = []

for emotion in emotion_to_label.keys():
    emotion_dir = Path(f"data/raw/train/{emotion}")
    for video_path in emotion_dir.glob("*.mp4"):
        try:
            features = extract_features_from_video(str(video_path))
            train_features.append(features)
            train_labels.append(emotion_to_label[emotion])
            print(f"Processed: {video_path.name}")
        except Exception as e:
            print(f"Error processing {video_path}: {e}")

# Convert to tensors
X_train = torch.tensor(np.array(train_features), dtype=torch.float32)
y_train = torch.tensor(np.array(train_labels), dtype=torch.long)

print(f"Training data: {X_train.shape}")
print(f"Labels: {y_train.shape}")
```

3. **Train the model**:
```python
from modules.stage3_fusion import MultimodalFusionRFRBoost

emotion_labels = ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']

# Initialize model
model = MultimodalFusionRFRBoost(config, emotion_labels)

# Train
print("Training RFRBoost model...")
model.fit(X_train, y_train)

# Save
model.save("pretrained/my_emotion_model.pth")
print("Model saved!")
```

4. **Evaluate**:
```python
# Load test data (same process as above)
X_test = ...  # Your test features
y_test = ...  # Your test labels

# Predict
predictions, confidences = model.predict(X_test)

# Evaluate
from modules.stage4_output import EmotionMetrics

metrics_calculator = EmotionMetrics(emotion_labels)
metrics = metrics_calculator.compute_all_metrics(y_test.numpy(), predictions)

print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1-Score: {metrics['weighted_f1']:.4f}")
```

---

## API Reference

### Stage 1: Input Processing

#### VideoProcessor
```python
processor = VideoProcessor(video_path, output_dir="data/processed")

# Extract audio
audio, sr, audio_path = processor.extract_audio(sample_rate=16000)

# Extract frames
frames, timestamps = processor.extract_frames(fps=5)

# Get metadata
metadata = processor.get_video_metadata()
```

#### ASRModule
```python
asr = ASRModule(model_name="base")

# Transcribe audio
transcription = asr.transcribe(audio, sample_rate)
# Returns: {'text': str, 'segments': list, 'language': str}
```

### Stage 2: Unimodal Processing

#### AudioFeatureExtractor
```python
audio_extractor = AudioFeatureExtractor(config)

# Extract all features
features = audio_extractor.extract_all_features(audio, sr, audio_path)
# Returns: numpy array of shape (n_features,)

# Extract specific feature types
prosodic = audio_extractor.extract_prosodic_features(audio, sr)
spectral = audio_extractor.extract_spectral_features(audio, sr)
```

#### VisualFeatureExtractor
```python
visual_extractor = VisualFeatureExtractor(config)

# Extract from all frames
features = visual_extractor.extract_video_features(frames)
# Returns: numpy array of shape (n_features,)

# Extract from single frame
frame_features = visual_extractor.extract_frame_features(frame)
```

#### TextFeatureExtractor
```python
text_extractor = TextFeatureExtractor(config)

# Extract all features
features = text_extractor.extract_all_features(text)
# Returns: numpy array of shape (n_features,)

# Extract specific features
sentiment = text_extractor.extract_sentiment(text)
embeddings = text_extractor.extract_semantic_embeddings(text)
```

### Stage 3: Multimodal Fusion

#### MultimodalFusionRFRBoost
```python
model = MultimodalFusionRFRBoost(config, emotion_labels)

# Train
model.fit(X_train, y_train)

# Predict batch
predictions, confidences = model.predict(X_test)

# Predict single sample
result = model.predict_single(
    audio_feat=audio_features,
    visual_feat=visual_features,
    text_feat=text_features
)
# Returns: {'predicted_emotion': str, 'confidence': float, 'all_confidences': dict}

# Save/load
model.save("path/to/model.pth")
model.load("path/to/model.pth")
```

### Stage 4: Metrics & Export

#### EmotionMetrics
```python
metrics_calc = EmotionMetrics(emotion_labels)

# Compute all metrics
metrics = metrics_calc.compute_all_metrics(y_true, y_pred, y_prob)

# Individual metrics
accuracy = metrics_calc.compute_accuracy(y_true, y_pred)
cm = metrics_calc.compute_confusion_matrix(y_true, y_pred)
```

#### ResultVisualizer
```python
visualizer = ResultVisualizer(emotion_labels, config)

# Plot confusion matrix
fig = visualizer.plot_confusion_matrix(confusion_matrix)

# Plot emotion timeline
fig = visualizer.plot_emotion_timeline(timestamps, emotions, confidences)

# Create summary plot
fig = visualizer.create_summary_plot(metrics)
```

#### ResultExporter
```python
exporter = ResultExporter(output_dir="data/results")

# Export to JSON
json_path = exporter.export_to_json(results)

# Export to CSV
csv_path = exporter.export_to_csv(timeline_data)

# Create detailed report
report_path = exporter.create_detailed_report(results, metrics)
```

---

## Examples

### Example 1: Process Single Video

```python
from pathlib import Path
from modules.stage1_input import VideoProcessor, ASRModule
from modules.stage2_unimodal import AudioFeatureExtractor, VisualFeatureExtractor, TextFeatureExtractor
from modules.stage3_fusion import MultimodalFusionRFRBoost
from utils.helpers import load_config

# Setup
config = load_config("config/config.yaml")
video_path = "data/raw/my_video.mp4"

# Stage 1: Input
processor = VideoProcessor(video_path)
audio, sr, audio_path = processor.extract_audio()
frames, timestamps = processor.extract_frames()

asr = ASRModule()
transcription = asr.transcribe(audio, sr)

# Stage 2: Features
audio_extractor = AudioFeatureExtractor(config['modalities']['audio'])
visual_extractor = VisualFeatureExtractor(config['modalities']['visual'])
text_extractor = TextFeatureExtractor(config['modalities']['text'])

audio_feat = audio_extractor.extract_all_features(audio, sr, audio_path)
visual_feat = visual_extractor.extract_video_features(frames)
text_feat = text_extractor.extract_all_features(transcription['text'])

# Stage 3: Prediction (assuming trained model)
emotion_labels = config['emotions']['labels']
model = MultimodalFusionRFRBoost(config, emotion_labels)
model.load("pretrained/my_model.pth")

result = model.predict_single(audio_feat, visual_feat, text_feat)

print(f"Detected: {result['predicted_emotion']} ({result['confidence']:.1%})")
```

### Example 2: Batch Processing

```python
import torch
from pathlib import Path
from tqdm import tqdm

video_dir = Path("data/raw/videos")
results = []

for video_path in tqdm(list(video_dir.glob("*.mp4"))):
    try:
        # Process (same as Example 1)
        ...
        result = model.predict_single(audio_feat, visual_feat, text_feat)
        results.append({
            'filename': video_path.name,
            'emotion': result['predicted_emotion'],
            'confidence': result['confidence']
        })
    except Exception as e:
        print(f"Error: {video_path.name} - {e}")

# Save results
import pandas as pd
df = pd.DataFrame(results)
df.to_csv("batch_results.csv", index=False)
```

### Example 3: Real-time Webcam Processing

```python
import cv2
import numpy as np

# Initialize camera
cap = cv2.VideoCapture(0)

# Initialize extractors
visual_extractor = VisualFeatureExtractor(config['modalities']['visual'])

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Extract features from single frame
    features_dict = visual_extractor.extract_frame_features(frame_rgb)
    
    if features_dict:
        # Display emotion (would need full pipeline for actual prediction)
        cv2.putText(frame, "Processing...", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Emotion Recognition', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'opensmile'"

**Solution**:
```bash
pip uninstall opensmile
pip install opensmile --no-cache-dir
```

### Issue: "No faces detected in video"

**Possible causes**:
- Poor video quality
- Faces too small or occluded
- Lighting issues

**Solutions**:
- Use higher resolution video
- Ensure faces are clearly visible
- Improve lighting
- Adjust MediaPipe confidence threshold in code

### Issue: "Out of memory" error

**Solutions**:
1. Reduce FPS: Change `fps: 5` to `fps: 3` in config
2. Process shorter clips: Split long videos
3. Disable resource-intensive features:
```yaml
modalities:
  visual:
    extract_action_units: false  # Disable AU extraction
```

### Issue: Slow processing speed

**Optimizations**:
1. Use GPU if available (automatic with PyTorch)
2. Reduce feature dimensions
3. Use faster ASR model: `whisper-tiny` instead of `whisper-base`
4. Disable OpenSMILE if not needed

### Issue: Poor accuracy

**Improvements**:
1. Train on more data
2. Enable all modalities
3. Increase RFRBoost layers: `n_layers: 8-10`
4. Use higher quality features:
```yaml
modalities:
  audio:
    n_mfcc: 40
    extract_opensmile: true
```

---

For more help, open an issue on the repository or check the main README.md.
