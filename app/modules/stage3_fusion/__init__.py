"""Stage 3: Multimodal Fusion Module

This module handles fusion of features from multiple modalities using various strategies:
- Early fusion: Concatenate features before classification
- Late fusion: Combine predictions from unimodal classifiers
- Attention fusion: Learn attention weights over modalities
- RFRBoost-based fusion for superior representation learning
- Hybrid fusion: Combines RFRBoost + Deep Learning + Attention (BEST)
"""

from .rfrboost_classifier import MultimodalFusionRFRBoost
from .early_fusion import EarlyFusion
from .late_fusion import LateFusion
from .hybrid_classifier import HybridMultimodalClassifier

__all__ = [
    'MultimodalFusionRFRBoost', 
    'EarlyFusion', 
    'LateFusion',
    'HybridMultimodalClassifier'  # ⭐ NEW: Best hybrid approach
]
