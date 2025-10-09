"""Late Fusion Strategy

Train separate classifiers for each modality and combine their predictions.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
import logging


class LateFusion:
    """Late fusion by combining predictions from unimodal classifiers."""
    
    def __init__(self, config: dict, emotion_labels: List[str]):
        """
        Initialize late fusion.
        
        Args:
            config: Configuration dictionary
            emotion_labels: List of emotion labels
        """
        self.config = config
        self.emotion_labels = emotion_labels
        self.n_classes = len(emotion_labels)
        self.logger = logging.getLogger(__name__)
        
        # Classifiers for each modality
        self.audio_classifier = None
        self.visual_classifier = None
        self.text_classifier = None
        
        # Fusion weights (can be learned or set manually)
        self.weights = {
            'audio': 0.33,
            'visual': 0.33,
            'text': 0.34
        }
    
    def set_fusion_weights(self, weights: Dict[str, float]):
        """
        Set fusion weights for each modality.
        
        Args:
            weights: Dictionary mapping modality names to weights
        """
        # Normalize weights to sum to 1
        total = sum(weights.values())
        self.weights = {k: v / total for k, v in weights.items()}
        self.logger.info(f"Fusion weights set: {self.weights}")
    
    def fuse_predictions(self,
                        audio_pred: Optional[np.ndarray] = None,
                        visual_pred: Optional[np.ndarray] = None,
                        text_pred: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fuse predictions from multiple modalities using weighted voting.
        
        Args:
            audio_pred: Audio predictions (probability distribution)
            visual_pred: Visual predictions (probability distribution)
            text_pred: Text predictions (probability distribution)
            
        Returns:
            Tuple of (final_predictions, confidence_scores)
        """
        predictions = []
        weights = []
        
        if audio_pred is not None and self.config.get('modalities', {}).get('audio', {}).get('enabled', True):
            predictions.append(audio_pred)
            weights.append(self.weights['audio'])
        
        if visual_pred is not None and self.config.get('modalities', {}).get('visual', {}).get('enabled', True):
            predictions.append(visual_pred)
            weights.append(self.weights['visual'])
        
        if text_pred is not None and self.config.get('modalities', {}).get('text', {}).get('enabled', True):
            predictions.append(text_pred)
            weights.append(self.weights['text'])
        
        if not predictions:
            raise ValueError("No predictions to fuse")
        
        # Weighted sum of predictions
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize
        
        fused_probs = np.zeros_like(predictions[0])
        for pred, weight in zip(predictions, weights):
            fused_probs += weight * pred
        
        # Get final predictions
        final_preds = np.argmax(fused_probs, axis=-1)
        
        self.logger.info(f"Late fusion: {len(predictions)} modalities with weights {weights}")
        
        return final_preds, fused_probs
