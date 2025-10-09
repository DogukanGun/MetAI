"""Stage 2: Unimodal Processing Module

This module handles feature extraction from individual modalities:
- Audio: Prosodic, spectral, and voice quality features
- Visual: Facial landmarks, action units, and deep visual features
- Text: Semantic embeddings, sentiment, and linguistic features
"""

from .audio_features import AudioFeatureExtractor
from .visual_features import VisualFeatureExtractor
from .text_features import TextFeatureExtractor

__all__ = ['AudioFeatureExtractor', 'VisualFeatureExtractor', 'TextFeatureExtractor']
