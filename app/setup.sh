#!/bin/bash
# Setup script for Multimodal Emotion Recognition System

echo "🎭 Multimodal Emotion Recognition System - Setup"
echo "================================================"
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then 
    echo "✅ Python $python_version detected"
else
    echo "❌ Python 3.8+ required (found $python_version)"
    exit 1
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv emotion_env
source emotion_env/bin/activate
echo "✅ Virtual environment created"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip
echo "✅ pip upgraded"

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt
echo "✅ Dependencies installed"

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p data/raw data/processed data/results pretrained notebooks
touch data/raw/.gitkeep data/processed/.gitkeep data/results/.gitkeep pretrained/.gitkeep
echo "✅ Directories created"

# Download NLTK data
echo ""
echo "Downloading NLTK data..."
python3 -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('averaged_perceptron_tagger', quiet=True); nltk.download('stopwords', quiet=True)"
echo "✅ NLTK data downloaded"

echo ""
echo "================================================"
echo "✅ Setup complete!"
echo ""
echo "To start the application:"
echo "  1. Activate the environment: source emotion_env/bin/activate"
echo "  2. Run the app: streamlit run app.py"
echo ""
echo "For more information, see README.md and USAGE_GUIDE.md"
echo "================================================"
