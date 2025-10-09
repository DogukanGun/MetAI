"""
Pre-trained Models Package

Contains implementations from research papers:
- Maelfabien: Multimodal Emotion Recognition
- Emotion-LLaMA: Language Model for Emotion Understanding
"""

from .maelfabien_model import MaelfabienMultimodalModel
from .emotion_llama_model import EmotionLLaMAModel

__all__ = ['MaelfabienMultimodalModel', 'EmotionLLaMAModel']
