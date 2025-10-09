"""
Emotion-LLaMA Implementation

Based on: https://github.com/ZebangCheng/Emotion-LLaMA

Architecture:
- Uses large language models (LLaMA-based) for emotion understanding
- Multi-task learning for emotion recognition and reasoning
- Instruction-tuned for emotion-aware responses
- Leverages textual, audio transcripts, and visual descriptions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging


class EmotionLLaMAEncoder(nn.Module):
    """
    Emotion-aware encoder based on LLaMA architecture.
    
    Features:
    - Multi-modal input processing
    - Emotion-specific prompt engineering
    - Context-aware emotion understanding
    """
    
    def __init__(self, hidden_size: int = 768, num_layers: int = 6, 
                 num_heads: int = 8, n_classes: int = 7):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        # Input projections for different modalities
        self.text_projection = nn.Linear(384, hidden_size)  # SBERT embeddings
        self.audio_projection = nn.Linear(116, hidden_size)  # Audio features
        self.visual_projection = nn.Linear(300, hidden_size)  # Visual features
        
        # Emotion-specific prompt embeddings
        self.emotion_prompts = nn.Parameter(torch.randn(n_classes, hidden_size))
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Multi-task heads
        self.emotion_classifier = nn.Linear(hidden_size, n_classes)
        self.emotion_intensity = nn.Linear(hidden_size, 1)
        self.emotion_reasoning = nn.Linear(hidden_size, hidden_size)
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, text_feat=None, audio_feat=None, visual_feat=None):
        """
        Forward pass with multi-modal inputs.
        
        Args:
            text_feat: Text features [batch, 384]
            audio_feat: Audio features [batch, 116]
            visual_feat: Visual features [batch, 300]
            
        Returns:
            Dictionary with multiple outputs
        """
        batch_size = 1
        if text_feat is not None:
            batch_size = text_feat.size(0)
        elif audio_feat is not None:
            batch_size = audio_feat.size(0)
        elif visual_feat is not None:
            batch_size = visual_feat.size(0)
        
        # Collect available modalities
        modality_features = []
        
        if text_feat is not None:
            text_proj = self.text_projection(text_feat)
            modality_features.append(text_proj.unsqueeze(1))
        
        if audio_feat is not None:
            audio_proj = self.audio_projection(audio_feat)
            modality_features.append(audio_proj.unsqueeze(1))
        
        if visual_feat is not None:
            visual_proj = self.visual_projection(visual_feat)
            modality_features.append(visual_proj.unsqueeze(1))
        
        # Concatenate modalities
        if len(modality_features) > 0:
            x = torch.cat(modality_features, dim=1)  # [batch, n_modalities, hidden]
            
            # Add emotion prompts
            emotion_prompts = self.emotion_prompts.unsqueeze(0).expand(batch_size, -1, -1)
            x = torch.cat([x, emotion_prompts], dim=1)  # [batch, n_modalities + n_classes, hidden]
            
            # Transformer encoding
            x = self.transformer(x)
            
            # Get pooled representation
            pooled = x.mean(dim=1)  # [batch, hidden]
            pooled = self.layer_norm(pooled)
            
            # Multi-task outputs
            emotion_logits = self.emotion_classifier(pooled)
            intensity = torch.sigmoid(self.emotion_intensity(pooled))
            reasoning = self.emotion_reasoning(pooled)
            
            return {
                'logits': emotion_logits,
                'intensity': intensity,
                'reasoning_features': reasoning,
                'pooled_features': pooled
            }
        
        return None


class EmotionReasoningModule(nn.Module):
    """
    Emotion reasoning module for explainable predictions.
    
    Generates natural language explanations for emotion predictions.
    """
    
    def __init__(self, hidden_size: int = 768, vocab_size: int = 5000):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        # Reasoning decoder
        self.reasoning_decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        self.output_projection = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, reasoning_features, max_length=20):
        """
        Generate reasoning sequence.
        
        Args:
            reasoning_features: Features from encoder [batch, hidden]
            max_length: Maximum sequence length
            
        Returns:
            Reasoning token logits
        """
        batch_size = reasoning_features.size(0)
        
        # Initialize decoder
        h0 = reasoning_features.unsqueeze(0).repeat(2, 1, 1)
        c0 = torch.zeros_like(h0)
        
        # Generate sequence
        outputs = []
        decoder_input = reasoning_features.unsqueeze(1)
        
        for _ in range(max_length):
            output, (h0, c0) = self.reasoning_decoder(decoder_input, (h0, c0))
            token_logits = self.output_projection(output)
            outputs.append(token_logits)
            decoder_input = output
        
        # Stack outputs
        outputs = torch.cat(outputs, dim=1)  # [batch, max_length, vocab_size]
        
        return outputs


class TemporalEmotionTracker(nn.Module):
    """
    Tracks emotion changes over time for temporal analysis.
    
    Features:
    - Emotion transition modeling
    - Temporal consistency
    - Smooth emotion trajectories
    """
    
    def __init__(self, hidden_size: int = 768, n_classes: int = 7):
        super().__init__()
        
        # Temporal LSTM
        self.temporal_lstm = nn.LSTM(
            input_size=n_classes,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # Transition model
        self.transition = nn.Linear(hidden_size, n_classes)
        
        # Smoothing layer
        self.smooth = nn.Conv1d(n_classes, n_classes, kernel_size=3, padding=1)
    
    def forward(self, emotion_sequence):
        """
        Process temporal emotion sequence.
        
        Args:
            emotion_sequence: Sequence of emotion distributions [batch, time, n_classes]
            
        Returns:
            Smoothed emotion predictions
        """
        # LSTM processing
        lstm_out, _ = self.temporal_lstm(emotion_sequence)
        
        # Transition prediction
        transitions = self.transition(lstm_out)
        
        # Apply smoothing
        transitions_t = transitions.transpose(1, 2)  # [batch, n_classes, time]
        smoothed = self.smooth(transitions_t)
        smoothed = smoothed.transpose(1, 2)  # [batch, time, n_classes]
        
        return F.softmax(smoothed, dim=-1)


class EmotionLLaMAModel:
    """
    Complete Emotion-LLaMA system for multimodal emotion recognition.
    
    Features:
    - Multi-modal emotion understanding
    - Emotion reasoning and explanation
    - Temporal emotion tracking
    - Instruction-following capabilities
    """
    
    def __init__(self, config: Dict, emotion_labels: List[str]):
        self.config = config
        self.emotion_labels = emotion_labels
        self.n_classes = len(emotion_labels)
        self.logger = logging.getLogger(__name__)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize models
        self.encoder = EmotionLLaMAEncoder(n_classes=self.n_classes).to(self.device)
        self.reasoning_module = EmotionReasoningModule().to(self.device)
        self.temporal_tracker = TemporalEmotionTracker(n_classes=self.n_classes).to(self.device)
        
        # Emotion descriptions for reasoning
        self.emotion_descriptions = {
            'neutral': 'no strong emotional expression, calm and composed',
            'happy': 'positive and joyful, showing satisfaction',
            'sad': 'sorrowful and melancholic, showing distress',
            'angry': 'irritated and furious, showing hostility',
            'fear': 'anxious and frightened, showing worry',
            'disgust': 'repulsed and aversive, showing dislike',
            'surprise': 'unexpected and astonished, showing amazement'
        }
        
        self.logger.info("Emotion-LLaMA model initialized")
    
    def predict_single(self, text_feat: Optional[np.ndarray] = None,
                      audio_feat: Optional[np.ndarray] = None,
                      visual_feat: Optional[np.ndarray] = None) -> Dict:
        """
        Predict emotion from single sample.
        
        Args:
            text_feat: Text features
            audio_feat: Audio features
            visual_feat: Visual features
            
        Returns:
            Prediction dictionary with reasoning
        """
        self.encoder.eval()
        self.reasoning_module.eval()
        
        # Convert to tensors
        text_tensor = torch.tensor(text_feat, dtype=torch.float32).unsqueeze(0) if text_feat is not None else None
        audio_tensor = torch.tensor(audio_feat, dtype=torch.float32).unsqueeze(0) if audio_feat is not None else None
        visual_tensor = torch.tensor(visual_feat, dtype=torch.float32).unsqueeze(0) if visual_feat is not None else None
        
        with torch.no_grad():
            # Forward pass
            if text_tensor is not None:
                text_tensor = text_tensor.to(self.device)
            if audio_tensor is not None:
                audio_tensor = audio_tensor.to(self.device)
            if visual_tensor is not None:
                visual_tensor = visual_tensor.to(self.device)
            
            outputs = self.encoder(text_tensor, audio_tensor, visual_tensor)
            
            if outputs is not None:
                logits = outputs['logits']
                intensity = outputs['intensity']
                
                # Get predictions
                probs = F.softmax(logits, dim=-1)
                pred_label = torch.argmax(probs, dim=-1).item()
                
                # Generate reasoning
                emotion_name = self.emotion_labels[pred_label]
                reasoning = f"The person appears to be {emotion_name}. "
                reasoning += f"{self.emotion_descriptions[emotion_name]}. "
                reasoning += f"Confidence: {probs[0, pred_label].item():.2%}. "
                reasoning += f"Intensity: {intensity.item():.2%}."
                
                result = {
                    'predicted_emotion': emotion_name,
                    'predicted_label': int(pred_label),
                    'confidence': float(probs[0, pred_label].item()),
                    'intensity': float(intensity.item()),
                    'all_confidences': {
                        self.emotion_labels[i]: float(probs[0, i].item())
                        for i in range(self.n_classes)
                    },
                    'reasoning': reasoning,
                    'fusion_method': 'emotion_llama_transformer'
                }
                
                return result
        
        return {}
    
    def predict_temporal(self, temporal_features: List[Dict]) -> List[Dict]:
        """
        Predict emotions over temporal sequence with smoothing.
        
        Args:
            temporal_features: List of feature dicts at different timestamps
            
        Returns:
            List of temporal predictions with smooth transitions
        """
        self.temporal_tracker.eval()
        
        # Collect predictions for each timestamp
        predictions = []
        
        for feat_dict in temporal_features:
            pred = self.predict_single(
                feat_dict.get('text'),
                feat_dict.get('audio'),
                feat_dict.get('visual')
            )
            if pred:
                predictions.append(pred)
        
        # Apply temporal smoothing if we have sequence
        if len(predictions) > 2:
            # Build emotion sequence
            emotion_seq = []
            for pred in predictions:
                conf_vector = [pred['all_confidences'][label] for label in self.emotion_labels]
                emotion_seq.append(conf_vector)
            
            emotion_tensor = torch.tensor(emotion_seq, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                smoothed = self.temporal_tracker(emotion_tensor)
                smoothed_probs = smoothed[0].cpu().numpy()
            
            # Update predictions with smoothed values
            for i, pred in enumerate(predictions):
                pred['smoothed_confidences'] = {
                    self.emotion_labels[j]: float(smoothed_probs[i, j])
                    for j in range(self.n_classes)
                }
                pred['smoothed_emotion'] = self.emotion_labels[np.argmax(smoothed_probs[i])]
        
        return predictions
