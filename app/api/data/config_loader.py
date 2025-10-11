"""
Configuration Loader

Loads and manages API configuration.
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any


def load_api_config() -> Dict[str, Any]:
    """
    Load API configuration.
    
    Returns:
        Configuration dictionary
    """
    # Get base directory
    base_dir = Path(__file__).parent.parent.parent
    
    # Load main config
    config_path = base_dir / "config" / "config.yaml"
    
    if not config_path.exists():
        # Return default config if file doesn't exist
        return get_default_config()
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load emotion labels
    emotion_labels_path = base_dir / "config" / "emotion_labels.json"
    if emotion_labels_path.exists():
        with open(emotion_labels_path, 'r') as f:
            emotion_data = json.load(f)
            config['emotions'] = emotion_data
    else:
        config['emotions'] = {
            'labels': ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise'],
            'colors': {
                'neutral': '#95a5a6',
                'happy': '#2ecc71',
                'sad': '#3498db',
                'angry': '#e74c3c',
                'fear': '#9b59b6',
                'disgust': '#16a085',
                'surprise': '#f39c12'
            }
        }
    
    return config


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration.
    
    Returns:
        Default configuration dictionary
    """
    return {
        'modalities': {
            'audio': {
                'enabled': True,
                'sample_rate': 16000,
                'n_mfcc': 40,
                'hop_length': 512,
                'n_fft': 2048
            },
            'visual': {
                'enabled': True,
                'fps': 5,
                'extract_action_units': True,
                'face_detection_confidence': 0.5
            },
            'text': {
                'enabled': True,
                'asr_model': 'openai/whisper-base',
                'max_length': 512,
                'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2'
            }
        },
        'rfrboost': {
            'n_layers': 6,
            'hidden_dim': 256,
            'boost_lr': 0.5,
            'feature_type': 'SWIM',
            'activation': 'tanh',
            'line_search_max_iter': 20
        },
        'emotions': {
            'labels': ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise'],
            'colors': {
                'neutral': '#95a5a6',
                'happy': '#2ecc71',
                'sad': '#3498db',
                'angry': '#e74c3c',
                'fear': '#9b59b6',
                'disgust': '#16a085',
                'surprise': '#f39c12'
            }
        }
    }
