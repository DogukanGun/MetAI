# Emotion Recognition API

FastAPI backend for multimodal emotion recognition with parallel model execution.

## Overview

This API provides advanced emotion recognition from video files using 5 state-of-the-art models running in parallel. It processes videos through audio, visual, and text modalities and returns comprehensive emotion analysis suitable for LLM integration.

## Architecture

```
api/
├── main.py                  # FastAPI application entry point
├── controllers/             # HTTP route handlers
│   └── emotion_controller.py
├── services/                # Business logic
│   └── emotion_analysis_service.py
├── models/                  # Pydantic models
│   └── response_models.py
├── utils/                   # Utility functions
│   └── video_validator.py
└── data/                    # Data access
    └── config_loader.py
```

## Features

### Emotion Recognition
- **5 Parallel Models**: Hybrid, RFRBoost, Maelfabien, Emotion-LLaMA, Simple Concatenation
- **Multimodal Analysis**: Audio + Visual + Text feature extraction
- **Temporal Analysis**: Emotion tracking over time (1 frame per 5 seconds)
- **Mental Health Scoring**: AI-based mental health assessment (0-100 scale)
- **Frame Extraction**: LlamaIndex approach (0.2 FPS)
- **FER Analysis**: Facial Expression Recognition with 7 emotions
- **LLM-Friendly Output**: Structured JSON responses
- **Automatic Validation**: File type, size, duration validation
- **Error Handling**: Comprehensive error responses

### Knowledge Base (NEW)
- **Document Ingestion**: PDF, TXT, DOCX, Images
- **Multimodal RAG**: Text and image retrieval using CLIP
- **Vector Storage**: FAISS-based similarity search
- **Separate Pipelines**: Independent document processing and querying
- **Semantic Search**: Find relevant content using natural language
- **LLM Integration**: Perfect for RAG-based AI agents

## Installation

### 1. Install API Dependencies

```bash
pip install -r requirements_api.txt
```

### 2. Install Core Dependencies

```bash
cd ..
pip install -r requirements.txt
```

### 3. Install Knowledge Base Dependencies

```bash
cd knowledge_base
pip install -r requirements_knowledge.txt
```

## Usage

### Start the API Server

```bash
# From api/ directory
python main.py

# Or using uvicorn directly
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### API Endpoints

#### 1. Analyze Video

**POST** `/api/v1/emotion/analyze`

Upload a video file for emotion analysis.

**Request**:
```bash
curl -X POST "http://localhost:8000/api/v1/emotion/analyze" \
  -H "Content-Type: multipart/form-data" \
  -F "video=@path/to/video.mp4"
```

**Response**:
```json
{
  "success": true,
  "message": "Successfully analyzed video with 5 models",
  "results": [
    {
      "model_name": "Hybrid (Best)",
      "fusion_strategy": "RFRBoost + Deep Learning + Attention",
      "processing_time": 15.3,
      "overall_prediction": {
        "predicted_emotion": "happy",
        "confidence": 0.87,
        "all_confidences": {
          "angry": 0.02,
          "disgust": 0.01,
          "fear": 0.03,
          "happy": 0.87,
          "sad": 0.02,
          "surprise": 0.03,
          "neutral": 0.02
        }
      },
      "temporal_predictions": [
        {
          "timestamp": 0.0,
          "emotion": "happy",
          "confidence": 0.85,
          "all_confidences": {...}
        }
      ],
      "mental_health_analysis": {
        "mental_health_score": 72.5,
        "avg_confidence": 0.82,
        "num_frames": 56,
        "dominant_emotion": "happy",
        "positive_percentage": 75.0,
        "negative_percentage": 25.0,
        "emotion_distribution": {...},
        "status": "Good",
        "recommendation": "Maintaining predominantly positive emotional expressions"
      },
      "video_metadata": {
        "filename": "video.mp4",
        "duration": 280.0,
        "fps": 30.0,
        "width": 1920,
        "height": 1080,
        "frame_count": 8400
      },
      "transcription": {
        "text": "Full video transcription...",
        "word_count": 150,
        "language": "en"
      },
      "features": {
        "audio_features": 116,
        "visual_features": 300,
        "text_features": 404
      },
      "modality_weights": {
        "audio": 0.32,
        "visual": 0.41,
        "text": 0.27
      },
      "model_agreement": {
        "rfrboost": "happy",
        "attention_deep": "happy",
        "mlp_baseline": "neutral",
        "agreement_status": "majority_agree"
      }
    }
  ],
  "total_processing_time": 45.6
}
```

#### 2. List Available Models

**GET** `/api/v1/emotion/models`

Get information about all available models.

**Response**:
```json
{
  "success": true,
  "models": [
    {
      "name": "Hybrid (Best)",
      "type": "hybrid",
      "description": "Ensemble combining RFRBoost + Deep Learning + Attention",
      "features": [...]
    }
  ]
}
```

#### 3. Health Check

**GET** `/api/v1/emotion/health`

Check API health status.

**Response**:
```json
{
  "status": "healthy",
  "service": "Emotion Recognition API",
  "version": "1.0.0"
}
```

## Request Validation

### Supported Video Formats
- MP4 (`.mp4`)
- AVI (`.avi`)
- MOV (`.mov`)
- WebM (`.webm`)
- MKV (`.mkv`)
- FLV (`.flv`)

### File Constraints
- **Max File Size**: 500 MB
- **Min Duration**: 1 second
- **Max Duration**: 10 minutes (600 seconds)

## Response Format

All responses follow a consistent structure:

### Success Response
```json
{
  "success": true,
  "message": "Success message",
  "results": [...],
  "total_processing_time": 45.6
}
```

### Error Response
```json
{
  "success": false,
  "error": "ErrorType",
  "message": "Error description",
  "details": {...}
}
```

## Model-Specific Data

### Hybrid Model
- `modality_weights`: Importance of each modality (audio, visual, text)
- `model_agreement`: Predictions from sub-models and agreement status

### Maelfabien Model
- `maelfabien_predictions`: Individual predictions from Text CNN-LSTM, Audio Time-CNN, Video XCeption

### Emotion-LLaMA Model
- `emotion_llama_details`: Emotion intensity and natural language reasoning

## Mental Health Analysis

Available for all models when FER analysis is successful:

```json
{
  "mental_health_score": 72.5,
  "avg_confidence": 0.82,
  "num_frames": 56,
  "dominant_emotion": "happy",
  "positive_percentage": 75.0,
  "negative_percentage": 25.0,
  "emotion_distribution": {
    "happy": 45.5,
    "neutral": 20.2,
    "surprise": 9.3,
    "sad": 12.1,
    "angry": 8.0,
    "fear": 3.5,
    "disgust": 1.4
  },
  "status": "Good",
  "recommendation": "Maintaining predominantly positive emotional expressions"
}
```

### Score Interpretation
- **70-100**: Good mental health
- **50-70**: Moderate
- **30-50**: Concerning
- **0-30**: At risk (recommend consultation)

## Parallel Processing

All 5 models run simultaneously using `ThreadPoolExecutor`:
- Each model processes independently
- Common features extracted once
- Results aggregated and returned together
- Typical total time: 45-60 seconds for 5-minute video

## Error Handling

### Validation Errors (400)
- Invalid file extension
- Invalid MIME type
- File too large/small
- Duration out of range

### Processing Errors (500)
- Model execution failure
- Feature extraction failure
- Video processing errors

## Integration with LLMs

The API response format is designed for easy LLM integration:

```python
# Example: Using with OpenAI
import openai

response = requests.post("http://localhost:8000/api/v1/emotion/analyze", ...)
analysis = response.json()

# Feed to LLM
prompt = f"""
Analyze this emotion recognition data:
- Overall Emotion: {analysis['results'][0]['overall_prediction']['predicted_emotion']}
- Confidence: {analysis['results'][0]['overall_prediction']['confidence']:.1%}
- Mental Health Score: {analysis['results'][0]['mental_health_analysis']['mental_health_score']}/100
- Transcript: {analysis['results'][0]['transcription']['text']}

Provide insights and recommendations.
"""

llm_response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}]
)
```

## Performance

Expected processing times (5-minute video):
- Feature Extraction: ~10-15 seconds
- Per Model: ~8-12 seconds
- Total (5 models parallel): ~45-60 seconds

## Development

### Project Structure
```
api/
├── main.py                      # FastAPI app
├── controllers/
│   └── emotion_controller.py    # Endpoints
├── services/
│   └── emotion_analysis_service.py  # Core logic
├── models/
│   └── response_models.py       # Pydantic models
├── utils/
│   └── video_validator.py       # Validation
└── data/
    └── config_loader.py         # Configuration
```

### Adding New Models

1. Update `emotion_analysis_service.py` - add model to `models_to_run`
2. Implement `_get_model_prediction()` logic for new model
3. Add model-specific response fields in `response_models.py`
4. Update documentation

## Logging

Logs are output to stdout with the format:
```
2025-10-10 00:00:00 - module - LEVEL - message
```

## Security Considerations

**For Production**:
1. Configure CORS properly (restrict origins)
2. Add authentication/authorization
3. Rate limiting
4. File upload size limits
5. Virus scanning for uploads
6. Use HTTPS
7. Validate content types strictly

## License

Same as main project license.

## Support

For issues or questions, refer to the main project README or open an issue.
