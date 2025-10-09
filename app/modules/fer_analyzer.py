"""
Facial Expression Recognition (FER) Analyzer

Based on: https://github.com/mujiyantosvc/Facial-Expression-Recognition-FER-for-Mental-Health-Detection-

Uses Swin Transformer, CNN, and ViT models for detecting emotions from facial expressions.
Analyzes extracted frames to provide time-series emotion predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
import logging
from PIL import Image
import torchvision.transforms as transforms


class SwinTransformerFER(nn.Module):
    """
    Swin Transformer for Facial Expression Recognition.
    
    Architecture based on the FER repository using Swin Transformer
    for mental health detection through facial expressions.
    """
    
    def __init__(self, num_classes: int = 7, pretrained: bool = False):
        """
        Initialize Swin Transformer FER model.
        
        Args:
            num_classes: Number of emotion classes (default: 7 for FER2013)
            pretrained: Whether to use pretrained weights
        """
        super().__init__()
        
        # Swin Transformer base configuration
        self.patch_size = 4
        self.embed_dim = 96
        self.depths = [2, 2, 6, 2]
        self.num_heads = [3, 6, 12, 24]
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(3, self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        
        # Transformer blocks (simplified)
        self.layer1 = self._make_layer(self.embed_dim, self.num_heads[0], self.depths[0])
        self.layer2 = self._make_layer(self.embed_dim * 2, self.num_heads[1], self.depths[1])
        self.layer3 = self._make_layer(self.embed_dim * 4, self.num_heads[2], self.depths[2])
        self.layer4 = self._make_layer(self.embed_dim * 8, self.num_heads[3], self.depths[3])
        
        # Global pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.embed_dim * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    
    def _make_layer(self, dim, num_heads, depth):
        """Create a Swin Transformer layer."""
        layers = []
        for _ in range(depth):
            layers.append(nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=num_heads,
                dim_feedforward=dim * 4,
                dropout=0.1,
                batch_first=True
            ))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass."""
        # Patch embedding
        x = self.patch_embed(x)
        b, c, h, w = x.shape
        
        # Reshape for transformer
        x = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        
        # Transformer layers (simplified without hierarchical structure)
        x = self.layer1(x)
        
        # Reshape back for pooling
        x = x.transpose(1, 2).view(b, -1, h, w)
        
        # Global pooling
        x = self.avgpool(x)
        x = x.flatten(1)
        
        # Classification
        x = self.classifier(x)
        
        return x


class CustomCNNFER(nn.Module):
    """
    Custom CNN for Facial Expression Recognition.
    
    Lightweight CNN model optimized for real-time emotion detection.
    """
    
    def __init__(self, num_classes: int = 7):
        super().__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        
        # Fully connected layers
        self.fc1 = nn.Linear(512 * 3 * 3, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        # Conv block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        # Conv block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        # Conv block 3
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        
        # Conv block 4
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x


class FERAnalyzer:
    """
    Facial Expression Recognition Analyzer for time-series emotion detection.
    
    Uses multiple models (Swin Transformer, Custom CNN) to analyze facial
    expressions from extracted frames and generate temporal emotion predictions.
    """
    
    def __init__(self, model_type: str = 'swin_transformer', device: str = None):
        """
        Initialize FER Analyzer.
        
        Args:
            model_type: Type of model ('swin_transformer', 'custom_cnn')
            device: Computing device ('cuda' or 'cpu')
        """
        self.logger = logging.getLogger(__name__)
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # FER2013 emotion labels
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        
        # Initialize model
        self.model_type = model_type
        if model_type == 'swin_transformer':
            self.model = SwinTransformerFER(num_classes=len(self.emotion_labels))
        elif model_type == 'custom_cnn':
            self.model = CustomCNNFER(num_classes=len(self.emotion_labels))
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Face detection (Haar Cascade)
        try:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
        except Exception as e:
            self.logger.warning(f"Could not load face cascade: {e}")
            self.face_cascade = None
        
        self.logger.info(f"FER Analyzer initialized with {model_type} on {self.device}")
    
    def detect_face(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect face in image and return cropped face region.
        
        Args:
            image: Input image (RGB or BGR)
            
        Returns:
            Cropped face image or None if no face detected
        """
        if self.face_cascade is None:
            return image
        
        # Convert to grayscale for face detection
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            return None
        
        # Get largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        
        # Add padding
        padding = int(0.2 * max(w, h))
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)
        
        # Crop face
        face = image[y:y+h, x:x+w]
        
        return face
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for model input.
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            Preprocessed tensor
        """
        # Detect and crop face
        face = self.detect_face(image)
        if face is None:
            face = image
        
        # Convert to PIL Image
        if face.dtype == np.uint8:
            pil_image = Image.fromarray(face)
        else:
            face = (face * 255).astype(np.uint8)
            pil_image = Image.fromarray(face)
        
        # Convert to RGB if needed
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Apply transforms
        tensor = self.transform(pil_image)
        
        return tensor
    
    def predict_emotion(self, image: np.ndarray) -> Dict:
        """
        Predict emotion from a single image.
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            Dictionary with emotion predictions
        """
        try:
            # Preprocess
            tensor = self.preprocess_image(image).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                logits = self.model(tensor)
                probs = F.softmax(logits, dim=1)
            
            # Get prediction
            probs_np = probs.cpu().numpy()[0]
            pred_idx = np.argmax(probs_np)
            
            result = {
                'emotion': self.emotion_labels[pred_idx],
                'confidence': float(probs_np[pred_idx]),
                'all_confidences': {
                    label: float(prob) 
                    for label, prob in zip(self.emotion_labels, probs_np)
                }
            }
            
            return result
        
        except Exception as e:
            self.logger.error(f"Error predicting emotion: {e}")
            return {
                'emotion': 'neutral',
                'confidence': 0.0,
                'all_confidences': {label: 0.0 for label in self.emotion_labels}
            }
    
    def analyze_frame_sequence(self, frame_paths: List[str], timestamps: List[float] = None) -> List[Dict]:
        """
        Analyze a sequence of frames and generate temporal emotion predictions.
        
        Args:
            frame_paths: List of paths to frame images
            timestamps: List of timestamps (optional)
            
        Returns:
            List of temporal emotion predictions
        """
        self.logger.info(f"Analyzing {len(frame_paths)} frames for temporal emotions")
        
        temporal_predictions = []
        
        for idx, frame_path in enumerate(frame_paths):
            try:
                # Load image
                image = cv2.imread(frame_path)
                if image is None:
                    self.logger.warning(f"Could not load frame: {frame_path}")
                    continue
                
                # Convert BGR to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Predict emotion
                prediction = self.predict_emotion(image_rgb)
                
                # Add timestamp
                if timestamps and idx < len(timestamps):
                    timestamp = timestamps[idx]
                else:
                    timestamp = idx * 5.0  # Default: 5 seconds per frame
                
                temporal_pred = {
                    'timestamp': float(timestamp),
                    'emotion': prediction['emotion'],
                    'confidence': prediction['confidence'],
                    'confidences': prediction['all_confidences'],
                    'frame_path': frame_path
                }
                
                temporal_predictions.append(temporal_pred)
                
            except Exception as e:
                self.logger.error(f"Error processing frame {frame_path}: {e}")
                continue
        
        self.logger.info(f"Generated {len(temporal_predictions)} temporal emotion predictions")
        
        return temporal_predictions
    
    def calculate_mental_health_score(self, temporal_predictions: List[Dict]) -> Dict:
        """
        Calculate mental health score based on temporal emotions.
        
        Based on the approach from the FER repository.
        
        Args:
            temporal_predictions: List of temporal emotion predictions
            
        Returns:
            Mental health analysis dictionary
        """
        if not temporal_predictions:
            return {}
        
        # Calculate average confidence
        avg_confidence = np.mean([p['confidence'] for p in temporal_predictions])
        
        # Count emotions
        emotion_counts = {}
        for pred in temporal_predictions:
            emotion = pred['emotion']
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        # Calculate percentages
        total = len(temporal_predictions)
        emotion_percentages = {
            emotion: (count / total) * 100
            for emotion, count in emotion_counts.items()
        }
        
        # Mental health score (0-100)
        # Higher positive emotions = higher score
        # Higher negative emotions = lower score
        positive_emotions = ['happy', 'surprise', 'neutral']
        negative_emotions = ['angry', 'disgust', 'fear', 'sad']
        
        positive_pct = sum(emotion_percentages.get(e, 0) for e in positive_emotions)
        negative_pct = sum(emotion_percentages.get(e, 0) for e in negative_emotions)
        
        # Base score is 50, adjusted by positive/negative balance
        mental_health_score = 50 + (positive_pct - negative_pct) / 2
        mental_health_score = max(0, min(100, mental_health_score))
        
        result = {
            'avg_confidence': float(avg_confidence),
            'num_frames': total,
            'mental_health_score': float(mental_health_score),
            'emotion_distribution': emotion_percentages,
            'dominant_emotion': max(emotion_counts, key=emotion_counts.get),
            'positive_percentage': float(positive_pct),
            'negative_percentage': float(negative_pct)
        }
        
        return result
