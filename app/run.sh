#!/bin/bash
# Quick run script for the Emotion Recognition System

# Activate virtual environment if it exists
if [ -d "emotion_env" ]; then
    source emotion_env/bin/activate
fi

# Run the Streamlit app
echo "🎭 Starting Multimodal Emotion Recognition System..."
echo "Opening browser at http://localhost:8501"
echo "Press Ctrl+C to stop the server"
echo ""

streamlit run app.py
