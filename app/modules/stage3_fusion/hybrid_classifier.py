"""Hybrid Multimodal Fusion Classifier

Combines multiple approaches for emotion recognition:
1. RFRBoost (from rfr/ directory) - Random Feature Representation Boosting
2. Deep Neural Network fusion (inspired by USDM, Emotion-LLaMA)
3. Attention-based fusion (from multimodal emotion recognition literature)
4. Ensemble methods (combining multiple classifiers)

This creates a powerful hybrid system that leverages the best of all approaches.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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
    logging.error(f"Could not import RFRBoost: {e}")
    RFRBOOST_AVAILABLE = False


class MultimodalAttentionFusion(nn.Module):
    """
    Attention-based fusion mechanism for multimodal features.
    Inspired by transformer-based multimodal fusion from recent papers.
    """
    
    def __init__(self, 
                 audio_dim: int,
                 visual_dim: int, 
                 text_dim: int,
                 hidden_dim: int = 256,
                 num_heads: int = 4):
        """
        Initialize attention fusion module.
        
        Args:
            audio_dim: Audio feature dimension
            visual_dim: Visual feature dimension
            text_dim: Text feature dimension
            hidden_dim: Hidden dimension for attention
            num_heads: Number of attention heads
        """
        super().__init__()
        
        # Project each modality to same dimension
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # Multi-head attention
        self.multihead_attn = nn.MultiheadAttention(
            hidden_dim, 
            num_heads, 
            dropout=0.1,
            batch_first=True
        )
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Modality-specific attention weights
        self.modality_attention = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 3),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, audio_feat, visual_feat, text_feat):
        """
        Forward pass with attention fusion.
        
        Args:
            audio_feat: Audio features [batch_size, audio_dim]
            visual_feat: Visual features [batch_size, visual_dim]
            text_feat: Text features [batch_size, text_dim]
            
        Returns:
            Fused features [batch_size, hidden_dim]
        """
        # Project to common space
        audio_h = self.audio_proj(audio_feat)
        visual_h = self.visual_proj(visual_feat)
        text_h = self.text_proj(text_feat)
        
        # Stack modalities for attention
        # [batch_size, 3, hidden_dim]
        modalities = torch.stack([audio_h, visual_h, text_h], dim=1)
        
        # Self-attention across modalities
        attn_output, attn_weights = self.multihead_attn(
            modalities, modalities, modalities
        )
        
        # Residual connection and layer norm
        modalities = self.layer_norm(modalities + attn_output)
        
        # Compute modality-specific attention weights
        concat_feat = modalities.view(modalities.size(0), -1)
        modality_weights = self.modality_attention(concat_feat)
        
        # Weighted combination of modalities
        modality_weights = modality_weights.unsqueeze(-1)  # [batch_size, 3, 1]
        fused = (modalities * modality_weights).sum(dim=1)  # [batch_size, hidden_dim]
        
        return fused, attn_weights, modality_weights.squeeze(-1)


class DeepEmotionNetwork(nn.Module):
    """
    Deep neural network for emotion classification.
    Inspired by USDM and Emotion-LLaMA architectures.
    """
    
    def __init__(self, 
                 input_dim: int,
                 n_classes: int,
                 hidden_dims: List[int] = [512, 256, 128]):
        """
        Initialize deep emotion network.
        
        Args:
            input_dim: Input feature dimension
            n_classes: Number of emotion classes
            hidden_dims: List of hidden layer dimensions
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, n_classes)
    
    def forward(self, x):
        """Forward pass."""
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits, features


class HybridMultimodalClassifier:
    """
    Hybrid classifier combining multiple approaches:
    1. RFRBoost for robust tabular feature learning
    2. Deep neural networks for representation learning
    3. Attention-based multimodal fusion
    4. Ensemble prediction aggregation
    """
    
    def __init__(self, 
                 config: Dict,
                 emotion_labels: List[str],
                 modality_dims: Dict[str, int]):
        """
        Initialize hybrid classifier.
        
        Args:
            config: Configuration dictionary
            emotion_labels: List of emotion labels
            modality_dims: Dictionary with dimensions for each modality
                          {'audio': audio_dim, 'visual': visual_dim, 'text': text_dim}
        """
        self.config = config
        self.emotion_labels = emotion_labels
        self.n_classes = len(emotion_labels)
        self.modality_dims = modality_dims
        self.logger = logging.getLogger(__name__)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize models
        self.models = {}
        self.is_fitted = False
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all sub-models."""
        total_dim = sum(self.modality_dims.values())
        
        # 1. RFRBoost model (for robust tabular learning)
        if RFRBOOST_AVAILABLE:
            try:
                rfrboost_config = self.config.get('rfrboost', {})
                self.models['rfrboost'] = GradientRFRBoostClassifier(
                    in_dim=total_dim,
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
                self.logger.info("✓ RFRBoost model initialized")
            except Exception as e:
                self.logger.warning(f"Could not initialize RFRBoost: {e}")
        
        # 2. Attention-based fusion network
        self.models['attention_fusion'] = MultimodalAttentionFusion(
            audio_dim=self.modality_dims['audio'],
            visual_dim=self.modality_dims['visual'],
            text_dim=self.modality_dims['text'],
            hidden_dim=256,
            num_heads=4
        ).to(self.device)
        self.logger.info("✓ Attention fusion model initialized")
        
        # 3. Deep emotion network (on fused features)
        self.models['deep_net'] = DeepEmotionNetwork(
            input_dim=256,  # From attention fusion
            n_classes=self.n_classes,
            hidden_dims=[256, 128, 64]
        ).to(self.device)
        self.logger.info("✓ Deep emotion network initialized")
        
        # 4. Simple MLP baseline (on concatenated features)
        self.models['mlp_baseline'] = DeepEmotionNetwork(
            input_dim=total_dim,
            n_classes=self.n_classes,
            hidden_dims=[512, 256, 128]
        ).to(self.device)
        self.logger.info("✓ MLP baseline initialized")
        
        # Ensemble weights (learned or fixed)
        self.ensemble_weights = {
            'rfrboost': 0.4,
            'attention_deep': 0.35,
            'mlp_baseline': 0.25
        }
    
    def fit(self, 
            audio_feat: torch.Tensor,
            visual_feat: torch.Tensor,
            text_feat: torch.Tensor,
            labels: torch.Tensor,
            epochs: int = 50,
            batch_size: int = 32,
            lr: float = 0.001):
        """
        Train all models in the hybrid system.
        
        Args:
            audio_feat: Audio features [n_samples, audio_dim]
            visual_feat: Visual features [n_samples, visual_dim]
            text_feat: Text features [n_samples, text_dim]
            labels: Labels [n_samples]
            epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
        """
        self.logger.info("Training hybrid multimodal classifier...")
        
        n_samples = audio_feat.shape[0]
        
        # 1. Train RFRBoost on concatenated features
        if 'rfrboost' in self.models:
            self.logger.info("Training RFRBoost...")
            X_concat = torch.cat([audio_feat, visual_feat, text_feat], dim=-1)
            y_onehot = F.one_hot(labels.long(), num_classes=self.n_classes).float()
            self.models['rfrboost'].fit(X_concat, y_onehot)
            self.logger.info("✓ RFRBoost training complete")
        
        # 2. Train attention fusion + deep network
        self.logger.info("Training attention-based fusion network...")
        self._train_attention_network(
            audio_feat, visual_feat, text_feat, labels,
            epochs=epochs, batch_size=batch_size, lr=lr
        )
        
        # 3. Train MLP baseline
        self.logger.info("Training MLP baseline...")
        X_concat = torch.cat([audio_feat, visual_feat, text_feat], dim=-1)
        self._train_mlp_baseline(
            X_concat, labels,
            epochs=epochs, batch_size=batch_size, lr=lr
        )
        
        self.is_fitted = True
        self.logger.info("✓ Hybrid classifier training complete!")
    
    def _train_attention_network(self, audio_feat, visual_feat, text_feat, labels,
                                 epochs, batch_size, lr):
        """Train attention fusion + deep network."""
        # Combine models for training
        model_params = list(self.models['attention_fusion'].parameters()) + \
                      list(self.models['deep_net'].parameters())
        
        optimizer = torch.optim.Adam(model_params, lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        dataset = torch.utils.data.TensorDataset(
            audio_feat, visual_feat, text_feat, labels
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        
        self.models['attention_fusion'].train()
        self.models['deep_net'].train()
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_audio, batch_visual, batch_text, batch_labels in dataloader:
                batch_audio = batch_audio.to(self.device)
                batch_visual = batch_visual.to(self.device)
                batch_text = batch_text.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                fused_feat, _, _ = self.models['attention_fusion'](
                    batch_audio, batch_visual, batch_text
                )
                logits, _ = self.models['deep_net'](fused_feat)
                
                # Loss and backward
                loss = criterion(logits, batch_labels.long())
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(dataloader)
                self.logger.info(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    def _train_mlp_baseline(self, X, labels, epochs, batch_size, lr):
        """Train MLP baseline."""
        optimizer = torch.optim.Adam(
            self.models['mlp_baseline'].parameters(), 
            lr=lr, weight_decay=1e-4
        )
        criterion = nn.CrossEntropyLoss()
        
        dataset = torch.utils.data.TensorDataset(X, labels)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        
        self.models['mlp_baseline'].train()
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_labels in dataloader:
                batch_X = batch_X.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                optimizer.zero_grad()
                
                logits, _ = self.models['mlp_baseline'](batch_X)
                loss = criterion(logits, batch_labels.long())
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(dataloader)
                self.logger.info(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    def predict(self, 
                audio_feat: torch.Tensor,
                visual_feat: torch.Tensor,
                text_feat: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Predict emotions using ensemble of all models.
        
        Args:
            audio_feat: Audio features
            visual_feat: Visual features
            text_feat: Text features
            
        Returns:
            Tuple of (predictions, confidences, detailed_results)
        """
        if not self.is_fitted:
            raise RuntimeError("Models not fitted. Call fit() first.")
        
        all_predictions = {}
        
        # 1. RFRBoost prediction
        if 'rfrboost' in self.models:
            with torch.no_grad():
                X_concat = torch.cat([audio_feat, visual_feat, text_feat], dim=-1)
                logits = self.models['rfrboost'](X_concat)
                probs = F.softmax(logits, dim=-1)
                all_predictions['rfrboost'] = probs.cpu().numpy()
        
        # 2. Attention + Deep network prediction
        self.models['attention_fusion'].eval()
        self.models['deep_net'].eval()
        
        with torch.no_grad():
            audio_feat_gpu = audio_feat.to(self.device)
            visual_feat_gpu = visual_feat.to(self.device)
            text_feat_gpu = text_feat.to(self.device)
            
            fused_feat, attn_weights, modality_weights = self.models['attention_fusion'](
                audio_feat_gpu, visual_feat_gpu, text_feat_gpu
            )
            logits, _ = self.models['deep_net'](fused_feat)
            probs = F.softmax(logits, dim=-1)
            all_predictions['attention_deep'] = probs.cpu().numpy()
        
        # 3. MLP baseline prediction
        self.models['mlp_baseline'].eval()
        
        with torch.no_grad():
            X_concat = torch.cat([audio_feat, visual_feat, text_feat], dim=-1).to(self.device)
            logits, _ = self.models['mlp_baseline'](X_concat)
            probs = F.softmax(logits, dim=-1)
            all_predictions['mlp_baseline'] = probs.cpu().numpy()
        
        # Ensemble prediction (weighted voting)
        ensemble_probs = np.zeros_like(all_predictions['mlp_baseline'])
        
        for model_name, probs in all_predictions.items():
            weight = self.ensemble_weights.get(model_name, 0.33)
            ensemble_probs += weight * probs
        
        # Normalize
        ensemble_probs = ensemble_probs / ensemble_probs.sum(axis=-1, keepdims=True)
        
        # Final predictions
        predictions = np.argmax(ensemble_probs, axis=-1)
        
        # Detailed results
        detailed = {
            'ensemble_probabilities': ensemble_probs,
            'individual_predictions': all_predictions,
            'modality_weights': modality_weights.cpu().numpy() if isinstance(modality_weights, torch.Tensor) else None,
            'ensemble_weights': self.ensemble_weights
        }
        
        return predictions, ensemble_probs, detailed
    
    def predict_single(self,
                      audio_feat: np.ndarray,
                      visual_feat: np.ndarray,
                      text_feat: np.ndarray) -> Dict:
        """
        Predict emotion for a single sample.
        
        Args:
            audio_feat: Audio features
            visual_feat: Visual features  
            text_feat: Text features
            
        Returns:
            Dictionary with prediction results
        """
        # Convert to tensors and add batch dimension
        audio_tensor = torch.tensor(audio_feat, dtype=torch.float32).unsqueeze(0)
        visual_tensor = torch.tensor(visual_feat, dtype=torch.float32).unsqueeze(0)
        text_tensor = torch.tensor(text_feat, dtype=torch.float32).unsqueeze(0)
        
        # Get predictions
        pred_labels, confidences, detailed = self.predict(
            audio_tensor, visual_tensor, text_tensor
        )
        
        # Format result
        result = {
            'predicted_emotion': self.emotion_labels[pred_labels[0]],
            'predicted_label': int(pred_labels[0]),
            'confidence': float(confidences[0, pred_labels[0]]),
            'all_confidences': {
                self.emotion_labels[i]: float(confidences[0, i])
                for i in range(self.n_classes)
            },
            'individual_model_predictions': {
                model: self.emotion_labels[np.argmax(probs[0])]
                for model, probs in detailed['individual_predictions'].items()
            },
            'ensemble_weights': detailed['ensemble_weights'],
            'modality_attention_weights': {
                'audio': float(detailed['modality_weights'][0, 0]) if detailed['modality_weights'] is not None else None,
                'visual': float(detailed['modality_weights'][0, 1]) if detailed['modality_weights'] is not None else None,
                'text': float(detailed['modality_weights'][0, 2]) if detailed['modality_weights'] is not None else None,
            } if detailed['modality_weights'] is not None else None
        }
        
        return result
    
    def save(self, path: str):
        """Save all models."""
        save_dict = {
            'config': self.config,
            'emotion_labels': self.emotion_labels,
            'modality_dims': self.modality_dims,
            'ensemble_weights': self.ensemble_weights,
            'is_fitted': self.is_fitted
        }
        
        # Save neural network states
        save_dict['attention_fusion_state'] = self.models['attention_fusion'].state_dict()
        save_dict['deep_net_state'] = self.models['deep_net'].state_dict()
        save_dict['mlp_baseline_state'] = self.models['mlp_baseline'].state_dict()
        
        # Save RFRBoost if available
        if 'rfrboost' in self.models and hasattr(self.models['rfrboost'], 'state_dict'):
            save_dict['rfrboost_state'] = self.models['rfrboost'].state_dict()
        
        torch.save(save_dict, path)
        self.logger.info(f"Hybrid model saved to {path}")
    
    def load(self, path: str):
        """Load all models."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.config = checkpoint['config']
        self.emotion_labels = checkpoint['emotion_labels']
        self.modality_dims = checkpoint['modality_dims']
        self.ensemble_weights = checkpoint['ensemble_weights']
        self.is_fitted = checkpoint['is_fitted']
        self.n_classes = len(self.emotion_labels)
        
        # Reinitialize models
        self._initialize_models()
        
        # Load neural network states
        self.models['attention_fusion'].load_state_dict(checkpoint['attention_fusion_state'])
        self.models['deep_net'].load_state_dict(checkpoint['deep_net_state'])
        self.models['mlp_baseline'].load_state_dict(checkpoint['mlp_baseline_state'])
        
        # Load RFRBoost if available
        if 'rfrboost_state' in checkpoint and 'rfrboost' in self.models:
            self.models['rfrboost'].load_state_dict(checkpoint['rfrboost_state'])
        
        self.logger.info(f"Hybrid model loaded from {path}")
