"""
Maelfabien Multimodal Emotion Recognition Implementation

Based on: https://github.com/maelfabien/Multimodal-Emotion-Recognition

Architecture:
- Text: Word2Vec + CNN + LSTM
- Audio: Time-Distributed CNN on mel-spectrograms  
- Video: XCeption for facial emotion recognition
- Ensemble: Weighted fusion of all three modalities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging


class TextEmotionCNN(nn.Module):
    """
    Text emotion recognition using CNN + LSTM.
    
    Architecture from maelfabien's repo:
    - Word2Vec embeddings (300 dim)
    - 3x Conv1D blocks (128, 256, 512 filters)
    - 3x LSTM layers (180 units each)
    - Dense layer (128 units)
    - Output layer (7 emotions)
    """
    
    def __init__(self, vocab_size: int = 10000, embedding_dim: int = 300, 
                 n_classes: int = 7, max_length: int = 100):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Conv1D blocks
        self.conv1 = nn.Conv1d(embedding_dim, 128, kernel_size=8, padding=4)
        self.bn1 = nn.BatchNorm1d(128)
        self.pool1 = nn.MaxPool1d(2)
        self.dropout1 = nn.Dropout(0.3)
        
        self.conv2 = nn.Conv1d(128, 256, kernel_size=8, padding=4)
        self.bn2 = nn.BatchNorm1d(256)
        self.pool2 = nn.MaxPool1d(2)
        self.dropout2 = nn.Dropout(0.3)
        
        self.conv3 = nn.Conv1d(256, 512, kernel_size=8, padding=4)
        self.bn3 = nn.BatchNorm1d(512)
        self.pool3 = nn.MaxPool1d(2)
        self.dropout3 = nn.Dropout(0.3)
        
        # LSTM layers
        self.lstm1 = nn.LSTM(512, 180, batch_first=True, bidirectional=False)
        self.lstm2 = nn.LSTM(180, 180, batch_first=True, bidirectional=False)
        self.lstm3 = nn.LSTM(180, 180, batch_first=True, bidirectional=False)
        
        # Dense layers
        self.fc1 = nn.Linear(180, 128)
        self.dropout_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, n_classes)
    
    def forward(self, x):
        # x: [batch, seq_len]
        x = self.embedding(x)  # [batch, seq_len, embedding_dim]
        x = x.transpose(1, 2)  # [batch, embedding_dim, seq_len]
        
        # Conv block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Conv block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Conv block 3
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # Prepare for LSTM
        x = x.transpose(1, 2)  # [batch, seq_len, features]
        
        # LSTM layers
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, (h, _) = self.lstm3(x)
        
        # Use last hidden state
        x = h[-1]  # [batch, 180]
        
        # Dense layers
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        
        return x


class TimeDistributedCNN(nn.Module):
    """
    Audio emotion recognition using Time-Distributed CNN.
    
    Architecture from maelfabien's repo:
    - Rolling window over mel-spectrogram
    - 4x LFLB (Local Feature Learning Blocks)
    - 2x LSTM layers
    - Dense layer + Softmax
    """
    
    def __init__(self, n_mels: int = 128, n_classes: int = 7):
        super().__init__()
        
        # Local Feature Learning Blocks (LFLB)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2)
        
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(2)
        
        # Calculate feature size after conv layers
        self.feature_size = 512 * (n_mels // 16) * (n_mels // 16)
        
        # LSTM layers
        self.lstm1 = nn.LSTM(self.feature_size, 256, batch_first=True)
        self.lstm2 = nn.LSTM(256, 128, batch_first=True)
        
        # Output layer
        self.fc = nn.Linear(128, n_classes)
    
    def forward(self, x):
        # x: [batch, time_steps, 1, n_mels, n_mels]
        batch_size, time_steps = x.size(0), x.size(1)
        
        # Process each time step
        features = []
        for t in range(time_steps):
            xt = x[:, t]  # [batch, 1, n_mels, n_mels]
            
            # LFLB 1
            xt = F.relu(self.bn1(self.conv1(xt)))
            xt = self.pool1(xt)
            
            # LFLB 2
            xt = F.relu(self.bn2(self.conv2(xt)))
            xt = self.pool2(xt)
            
            # LFLB 3
            xt = F.relu(self.bn3(self.conv3(xt)))
            xt = self.pool3(xt)
            
            # LFLB 4
            xt = F.relu(self.bn4(self.conv4(xt)))
            xt = self.pool4(xt)
            
            # Flatten
            xt = xt.view(batch_size, -1)
            features.append(xt)
        
        # Stack temporal features
        x = torch.stack(features, dim=1)  # [batch, time_steps, features]
        
        # LSTM layers
        x, _ = self.lstm1(x)
        x, (h, _) = self.lstm2(x)
        
        # Use last hidden state
        x = h[-1]  # [batch, 128]
        
        # Output
        x = self.fc(x)
        
        return x


class XCeptionFacialEmotion(nn.Module):
    """
    Video emotion recognition using XCeption architecture.
    
    Architecture from maelfabien's repo:
    - Modified XCeption with DepthWise Separable Convolutions
    - Optimized for 48x48 facial images
    - Outputs 7 emotion classes
    """
    
    def __init__(self, n_classes: int = 7):
        super().__init__()
        
        # Entry flow
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Depthwise separable convolutions
        self.depthwise1 = nn.Conv2d(64, 64, 3, padding=1, groups=64)
        self.pointwise1 = nn.Conv2d(64, 128, 1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.depthwise2 = nn.Conv2d(128, 128, 3, padding=1, groups=128)
        self.pointwise2 = nn.Conv2d(128, 256, 1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.depthwise3 = nn.Conv2d(256, 256, 3, padding=1, groups=256)
        self.pointwise3 = nn.Conv2d(256, 512, 1)
        self.bn5 = nn.BatchNorm2d(512)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classification head
        self.fc = nn.Linear(512, n_classes)
        
        # Regularization
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # Entry flow
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        # Depthwise separable conv 1
        x = self.depthwise1(x)
        x = F.relu(self.bn3(self.pointwise1(x)))
        x = F.max_pool2d(x, 2)
        
        # Depthwise separable conv 2
        x = self.depthwise2(x)
        x = F.relu(self.bn4(self.pointwise2(x)))
        x = F.max_pool2d(x, 2)
        
        # Depthwise separable conv 3
        x = self.depthwise3(x)
        x = F.relu(self.bn5(self.pointwise3(x)))
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Classification
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


class MaelfabienMultimodalModel:
    """
    Complete multimodal emotion recognition system from maelfabien's repository.
    
    Ensemble approach with weighted fusion.
    """
    
    def __init__(self, config: Dict, emotion_labels: List[str]):
        self.config = config
        self.emotion_labels = emotion_labels
        self.n_classes = len(emotion_labels)
        self.logger = logging.getLogger(__name__)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize models
        self.text_model = TextEmotionCNN(n_classes=self.n_classes).to(self.device)
        self.audio_model = TimeDistributedCNN(n_classes=self.n_classes).to(self.device)
        self.video_model = XCeptionFacialEmotion(n_classes=self.n_classes).to(self.device)
        
        # Fusion weights (from paper)
        self.fusion_weights = {
            'text': 0.33,
            'audio': 0.33,
            'video': 0.34
        }
        
        self.logger.info("Maelfabien multimodal model initialized")
    
    def predict(self, text_input: Optional[torch.Tensor] = None,
                audio_input: Optional[torch.Tensor] = None,
                video_input: Optional[torch.Tensor] = None) -> Dict:
        """
        Predict emotion from multimodal inputs.
        
        Args:
            text_input: Tokenized text [batch, seq_len]
            audio_input: Mel-spectrogram windows [batch, time_steps, 1, n_mels, n_mels]
            video_input: Facial images [batch, 3, 48, 48]
            
        Returns:
            Dictionary with predictions and confidences
        """
        self.text_model.eval()
        self.audio_model.eval()
        self.video_model.eval()
        
        predictions = {}
        
        with torch.no_grad():
            # Text prediction
            if text_input is not None:
                text_logits = self.text_model(text_input.to(self.device))
                text_probs = F.softmax(text_logits, dim=-1)
                predictions['text'] = text_probs.cpu().numpy()
            
            # Audio prediction
            if audio_input is not None:
                audio_logits = self.audio_model(audio_input.to(self.device))
                audio_probs = F.softmax(audio_logits, dim=-1)
                predictions['audio'] = audio_probs.cpu().numpy()
            
            # Video prediction
            if video_input is not None:
                video_logits = self.video_model(video_input.to(self.device))
                video_probs = F.softmax(video_logits, dim=-1)
                predictions['video'] = video_probs.cpu().numpy()
        
        # Ensemble fusion
        if len(predictions) > 0:
            ensemble_probs = np.zeros((1, self.n_classes))
            total_weight = 0
            
            for modality, probs in predictions.items():
                weight = self.fusion_weights[modality]
                ensemble_probs += weight * probs
                total_weight += weight
            
            ensemble_probs /= total_weight
            
            # Final prediction
            pred_label = np.argmax(ensemble_probs, axis=-1)[0]
            
            result = {
                'predicted_emotion': self.emotion_labels[pred_label],
                'predicted_label': int(pred_label),
                'confidence': float(ensemble_probs[0, pred_label]),
                'all_confidences': {
                    self.emotion_labels[i]: float(ensemble_probs[0, i])
                    for i in range(self.n_classes)
                },
                'individual_predictions': predictions,
                'fusion_method': 'maelfabien_weighted_ensemble'
            }
            
            return result
        
        return {}
