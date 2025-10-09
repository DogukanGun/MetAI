"""RFRBoost-based Multimodal Fusion Classifier

Uses Random Feature Representation Boosting for emotion classification.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import sys
import os

# Add rfr directory to path to import models
rfr_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../rfr'))
if rfr_dir not in sys.path:
    sys.path.insert(0, rfr_dir)

try:
    from models.random_feature_representation_boosting import GradientRFRBoostClassifier
    RFRBOOST_AVAILABLE = True
except ImportError as e:
    logging.error(f"Could not import RFRBoost models. Make sure the rfr/models directory exists. Error: {e}")
    RFRBOOST_AVAILABLE = False
    # Create dummy class for development
    class GradientRFRBoostClassifier:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("RFRBoost models not available")


class MultimodalFusionRFRBoost:
    """Multimodal fusion using RFRBoost classifier."""
    
    def __init__(self, config: Dict, emotion_labels: List[str]):
        """
        Initialize multimodal fusion model.
        
        Args:
            config: Configuration dictionary
            emotion_labels: List of emotion labels
        """
        self.config = config
        self.emotion_labels = emotion_labels
        self.n_classes = len(emotion_labels)
        self.logger = logging.getLogger(__name__)
        
        self.model = None
        self.feature_dim = None
        self.is_fitted = False
    
    def prepare_features(self, 
                        audio_feat: Optional[np.ndarray] = None,
                        visual_feat: Optional[np.ndarray] = None,
                        text_feat: Optional[np.ndarray] = None) -> torch.Tensor:
        """
        Concatenate features from enabled modalities.
        
        Args:
            audio_feat: Audio features
            visual_feat: Visual features
            text_feat: Text features
            
        Returns:
            Concatenated feature tensor
        """
        features = []
        
        # Handle single sample or batch
        if audio_feat is not None and self.config.get('audio', {}).get('enabled', True):
            audio_tensor = torch.tensor(audio_feat, dtype=torch.float32)
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            features.append(audio_tensor)
        
        if visual_feat is not None and self.config.get('visual', {}).get('enabled', True):
            visual_tensor = torch.tensor(visual_feat, dtype=torch.float32)
            if visual_tensor.dim() == 1:
                visual_tensor = visual_tensor.unsqueeze(0)
            features.append(visual_tensor)
        
        if text_feat is not None and self.config.get('text', {}).get('enabled', True):
            text_tensor = torch.tensor(text_feat, dtype=torch.float32)
            if text_tensor.dim() == 1:
                text_tensor = text_tensor.unsqueeze(0)
            features.append(text_tensor)
        
        if not features:
            raise ValueError("No features provided or all modalities are disabled")
        
        # Concatenate along feature dimension
        fused = torch.cat(features, dim=-1)
        return fused
    
    def initialize_model(self, feature_dim: int):
        """
        Initialize RFRBoost model with specified feature dimension.
        
        Args:
            feature_dim: Total feature dimension after fusion
        """
        self.feature_dim = feature_dim
        
        rfrboost_config = self.config.get('rfrboost', {})
        
        self.logger.info(f"Initializing RFRBoost classifier with {feature_dim} input features")
        
        try:
            self.model = GradientRFRBoostClassifier(
                in_dim=feature_dim,
                n_classes=self.n_classes,
                hidden_dim=rfrboost_config.get('hidden_dim', 256),
                n_layers=rfrboost_config.get('n_layers', 6),
                randfeat_xt_dim=rfrboost_config.get('randfeat_xt_dim', 512),
                randfeat_x0_dim=rfrboost_config.get('randfeat_x0_dim', 512),
                l2_cls=rfrboost_config.get('l2_cls', 0.001),
                l2_ghat=rfrboost_config.get('l2_ghat', 0.001),
                boost_lr=rfrboost_config.get('boost_lr', 0.5),
                feature_type=rfrboost_config.get('feature_type', 'SWIM'),
                upscale_type=rfrboost_config.get('upscale_type', 'SWIM'),
                activation=rfrboost_config.get('activation', 'tanh'),
                use_batchnorm=rfrboost_config.get('use_batchnorm', True),
                do_linesearch=rfrboost_config.get('do_linesearch', True),
            )
            self.logger.info("RFRBoost classifier initialized successfully")
        
        except Exception as e:
            self.logger.error(f"Error initializing RFRBoost: {e}")
            raise
    
    def fit(self, X: torch.Tensor, y: torch.Tensor):
        """
        Train the RFRBoost classifier.
        
        Args:
            X: Input features [n_samples, feature_dim]
            y: Target labels [n_samples] (integer labels)
        """
        if self.model is None:
            self.initialize_model(X.shape[1])
        
        # Convert labels to one-hot encoding
        if y.dim() == 1:
            y_onehot = torch.nn.functional.one_hot(
                y.long(),
                num_classes=self.n_classes
            ).float()
        else:
            y_onehot = y.float()
        
        self.logger.info(f"Training RFRBoost on {X.shape[0]} samples with {X.shape[1]} features")
        
        try:
            self.model.fit(X, y_onehot)
            self.is_fitted = True
            self.logger.info("Training complete")
        except Exception as e:
            self.logger.error(f"Error during training: {e}")
            raise
    
    def predict(self, X: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict emotions from features.
        
        Args:
            X: Input features [n_samples, feature_dim]
            
        Returns:
            Tuple of (predicted_labels, confidence_scores)
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        
        with torch.no_grad():
            logits = self.model(X)
            predictions = torch.argmax(logits, dim=1)
            confidences = torch.softmax(logits, dim=1)
        
        return predictions.numpy(), confidences.numpy()
    
    def predict_single(self, 
                      audio_feat: Optional[np.ndarray] = None,
                      visual_feat: Optional[np.ndarray] = None,
                      text_feat: Optional[np.ndarray] = None) -> Dict:
        """
        Predict emotion from a single sample.
        
        Args:
            audio_feat: Audio features
            visual_feat: Visual features
            text_feat: Text features
            
        Returns:
            Dictionary with prediction results
        """
        # Prepare features
        X = self.prepare_features(audio_feat, visual_feat, text_feat)
        
        # Get predictions
        pred_labels, confidences = self.predict(X)
        
        # Format result
        result = {
            'predicted_emotion': self.emotion_labels[pred_labels[0]],
            'predicted_label': int(pred_labels[0]),
            'confidence': float(confidences[0, pred_labels[0]]),
            'all_confidences': {
                self.emotion_labels[i]: float(confidences[0, i])
                for i in range(self.n_classes)
            }
        }
        
        return result
    
    def save(self, path: str):
        """
        Save model to file.
        
        Args:
            path: Path to save model
        """
        if not self.is_fitted:
            self.logger.warning("Saving unfitted model")
        
        torch.save({
            'model_state': self.model.state_dict() if hasattr(self.model, 'state_dict') else None,
            'config': self.config,
            'emotion_labels': self.emotion_labels,
            'feature_dim': self.feature_dim,
            'is_fitted': self.is_fitted
        }, path)
        
        self.logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """
        Load model from file.
        
        Args:
            path: Path to load model from
        """
        checkpoint = torch.load(path)
        
        self.config = checkpoint['config']
        self.emotion_labels = checkpoint['emotion_labels']
        self.feature_dim = checkpoint['feature_dim']
        self.is_fitted = checkpoint['is_fitted']
        self.n_classes = len(self.emotion_labels)
        
        if self.feature_dim:
            self.initialize_model(self.feature_dim)
            if checkpoint['model_state']:
                self.model.load_state_dict(checkpoint['model_state'])
        
        self.logger.info(f"Model loaded from {path}")
