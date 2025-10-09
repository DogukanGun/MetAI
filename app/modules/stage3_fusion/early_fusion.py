"""Early Fusion Strategy

Simple concatenation of features from all modalities before classification.
"""

import numpy as np
import torch
from typing import Optional, List
import logging


class EarlyFusion:
    """Early fusion by feature concatenation."""
    
    def __init__(self, config: dict):
        """
        Initialize early fusion.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def fuse(self,
             audio_feat: Optional[np.ndarray] = None,
             visual_feat: Optional[np.ndarray] = None,
             text_feat: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Fuse features by concatenation.
        
        Args:
            audio_feat: Audio features
            visual_feat: Visual features
            text_feat: Text features
            
        Returns:
            Concatenated feature vector
        """
        features = []
        
        if audio_feat is not None and self.config.get('modalities', {}).get('audio', {}).get('enabled', True):
            features.append(audio_feat)
            self.logger.debug(f"Added audio features: {audio_feat.shape}")
        
        if visual_feat is not None and self.config.get('modalities', {}).get('visual', {}).get('enabled', True):
            features.append(visual_feat)
            self.logger.debug(f"Added visual features: {visual_feat.shape}")
        
        if text_feat is not None and self.config.get('modalities', {}).get('text', {}).get('enabled', True):
            features.append(text_feat)
            self.logger.debug(f"Added text features: {text_feat.shape}")
        
        if not features:
            raise ValueError("No features to fuse")
        
        # Concatenate
        fused = np.concatenate(features, axis=-1)
        self.logger.info(f"Early fusion: {len(features)} modalities -> {fused.shape} features")
        
        return fused
