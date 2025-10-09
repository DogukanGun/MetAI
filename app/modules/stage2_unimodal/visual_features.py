"""Visual Feature Extraction Module

Extracts emotion-relevant features from facial expressions:
- Facial landmarks (468-point MediaPipe or 68-point dlib)
- Facial Action Units (FAUs)
- Geometric features (eye aspect ratio, mouth aspect ratio)
- Head pose and gaze
- Deep visual features from CNNs
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import cv2

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logging.warning("mediapipe not available")

try:
    from feat import Detector
    PYFEAT_AVAILABLE = True
except ImportError:
    PYFEAT_AVAILABLE = False
    logging.warning("py-feat not available")


class VisualFeatureExtractor:
    """Extracts emotion-relevant features from video frames."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize visual feature extractor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize MediaPipe Face Mesh
        self.face_mesh = None
        if MEDIAPIPE_AVAILABLE:
            try:
                self.mp_face_mesh = mp.solutions.face_mesh
                self.face_mesh = self.mp_face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=1,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                self.logger.info("MediaPipe Face Mesh initialized")
            except Exception as e:
                self.logger.warning(f"Could not initialize MediaPipe: {e}")
        
        # Initialize Py-feat for Action Units
        self.detector = None
        if PYFEAT_AVAILABLE and self.config.get('extract_action_units', True):
            try:
                self.detector = Detector()
                self.logger.info("Py-feat detector initialized")
            except Exception as e:
                self.logger.warning(f"Could not initialize Py-feat: {e}")
    
    def extract_facial_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract facial landmarks from frame using MediaPipe.
        
        Args:
            frame: RGB image frame
            
        Returns:
            Flattened array of landmark coordinates (x, y, z) or None
        """
        if self.face_mesh is None:
            return None
        
        try:
            # Ensure RGB format
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            elif frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            
            results = self.face_mesh.process(frame)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]
                coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
                return coords.flatten()
            
        except Exception as e:
            self.logger.error(f"Error extracting landmarks: {e}")
        
        return None
    
    def extract_action_units(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract Facial Action Units using Py-feat.
        
        Args:
            frame: RGB image frame
            
        Returns:
            Array of AU intensities or None
        """
        if self.detector is None:
            return None
        
        try:
            # Py-feat expects BGR format
            if frame.shape[2] == 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame
            
            # Detect faces and AUs
            results = self.detector.detect_faces(frame_bgr)
            
            if len(results) > 0:
                aus = self.detector.detect_aus(frame_bgr, results[0])
                return np.array(aus)
        
        except Exception as e:
            self.logger.error(f"Error extracting action units: {e}")
        
        return None
    
    def compute_geometric_features(self, landmarks: np.ndarray) -> Dict[str, float]:
        """
        Compute geometric features from facial landmarks.
        
        Args:
            landmarks: Flattened landmark array (N*3,)
            
        Returns:
            Dictionary of geometric features
        """
        features = {}
        
        try:
            # Reshape to (N, 3)
            landmarks_3d = landmarks.reshape(-1, 3)
            
            # Eye Aspect Ratio (EAR) - simplified version
            # Using landmarks for left eye (indices 33, 133, 160, 144, 153, 145)
            # and right eye (362, 263, 387, 373, 380, 374)
            
            # Left eye
            if len(landmarks_3d) >= 468:  # MediaPipe has 468 landmarks
                left_eye_indices = [33, 133, 160, 144, 153, 145]
                right_eye_indices = [362, 263, 387, 373, 380, 374]
                
                left_eye = landmarks_3d[left_eye_indices, :2]  # Use only x, y
                right_eye = landmarks_3d[right_eye_indices, :2]
                
                # Calculate EAR for left eye
                left_ear = self._calculate_ear(left_eye)
                right_ear = self._calculate_ear(right_eye)
                
                features['left_eye_aspect_ratio'] = float(left_ear)
                features['right_eye_aspect_ratio'] = float(right_ear)
                features['eye_aspect_ratio_mean'] = float((left_ear + right_ear) / 2)
                
                # Mouth Aspect Ratio (MAR)
                mouth_indices = [61, 291, 0, 17, 269, 405]  # Simplified mouth landmarks
                mouth = landmarks_3d[mouth_indices, :2]
                mar = self._calculate_mar(mouth)
                features['mouth_aspect_ratio'] = float(mar)
                
                # Face width and height (normalized)
                face_width = np.max(landmarks_3d[:, 0]) - np.min(landmarks_3d[:, 0])
                face_height = np.max(landmarks_3d[:, 1]) - np.min(landmarks_3d[:, 1])
                features['face_width'] = float(face_width)
                features['face_height'] = float(face_height)
                features['face_aspect_ratio'] = float(face_width / face_height) if face_height > 0 else 0.0
        
        except Exception as e:
            self.logger.error(f"Error computing geometric features: {e}")
        
        return features
    
    def _calculate_ear(self, eye_landmarks: np.ndarray) -> float:
        """
        Calculate Eye Aspect Ratio.
        
        Args:
            eye_landmarks: Array of eye landmark coordinates (6, 2)
            
        Returns:
            EAR value
        """
        if len(eye_landmarks) < 6:
            return 0.0
        
        # Vertical distances
        v1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
        v2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
        
        # Horizontal distance
        h = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        
        if h > 0:
            ear = (v1 + v2) / (2.0 * h)
        else:
            ear = 0.0
        
        return ear
    
    def _calculate_mar(self, mouth_landmarks: np.ndarray) -> float:
        """
        Calculate Mouth Aspect Ratio.
        
        Args:
            mouth_landmarks: Array of mouth landmark coordinates
            
        Returns:
            MAR value
        """
        if len(mouth_landmarks) < 4:
            return 0.0
        
        # Vertical distance
        v = np.linalg.norm(mouth_landmarks[1] - mouth_landmarks[5])
        
        # Horizontal distance
        h = np.linalg.norm(mouth_landmarks[0] - mouth_landmarks[3])
        
        if h > 0:
            mar = v / h
        else:
            mar = 0.0
        
        return mar
    
    def estimate_head_pose(self, landmarks: np.ndarray, 
                          frame_shape: Tuple[int, int]) -> Dict[str, float]:
        """
        Estimate head pose (yaw, pitch, roll) from landmarks.
        
        Args:
            landmarks: Facial landmarks
            frame_shape: (height, width) of frame
            
        Returns:
            Dictionary with pose angles
        """
        pose = {'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0}
        
        try:
            # Simplified head pose estimation
            # This would require camera calibration for accurate results
            landmarks_3d = landmarks.reshape(-1, 3)
            
            # Calculate centroid
            centroid = np.mean(landmarks_3d[:, :2], axis=0)
            
            # Normalize by frame size
            normalized_x = (centroid[0] - 0.5) * 2  # Range [-1, 1]
            normalized_y = (centroid[1] - 0.5) * 2
            
            # Approximate yaw and pitch from face position
            pose['yaw'] = float(normalized_x * 45)  # Approximate angle
            pose['pitch'] = float(normalized_y * 30)
            
        except Exception as e:
            self.logger.error(f"Error estimating head pose: {e}")
        
        return pose
    
    def extract_frame_features(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Extract all visual features from a single frame.
        
        Args:
            frame: RGB image frame
            
        Returns:
            Dictionary of features
        """
        features_dict = {}
        
        # Extract landmarks
        landmarks = self.extract_facial_landmarks(frame)
        if landmarks is not None:
            features_dict['landmarks'] = landmarks
            
            # Compute geometric features
            geometric = self.compute_geometric_features(landmarks)
            features_dict['geometric'] = geometric
            
            # Estimate head pose
            pose = self.estimate_head_pose(landmarks, frame.shape[:2])
            features_dict['head_pose'] = pose
        
        # Extract action units
        aus = self.extract_action_units(frame)
        if aus is not None:
            features_dict['action_units'] = aus
        
        return features_dict if features_dict else None
    
    def extract_video_features(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Extract features from all frames and aggregate temporally.
        
        Args:
            frames: List of RGB image frames
            
        Returns:
            Aggregated feature vector
        """
        self.logger.info(f"Extracting visual features from {len(frames)} frames")
        
        all_frame_features = []
        
        for i, frame in enumerate(frames):
            features = self.extract_frame_features(frame)
            
            if features:
                # Flatten all features for this frame
                frame_vector = []
                
                # Add geometric features
                if 'geometric' in features:
                    frame_vector.extend(features['geometric'].values())
                
                # Add head pose
                if 'head_pose' in features:
                    frame_vector.extend(features['head_pose'].values())
                
                # Add subset of landmarks (to avoid too high dimensionality)
                if 'landmarks' in features:
                    # Use every 10th landmark
                    landmarks_subset = features['landmarks'][::30]
                    frame_vector.extend(landmarks_subset)
                
                # Add action units if available
                if 'action_units' in features:
                    frame_vector.extend(features['action_units'])
                
                all_frame_features.append(frame_vector)
        
        if not all_frame_features:
            self.logger.warning("No visual features extracted")
            return np.array([])
        
        # Convert to array
        all_frame_features = np.array(all_frame_features, dtype=np.float32)
        
        # Aggregate over time: concatenate mean and std
        mean_features = np.mean(all_frame_features, axis=0)
        std_features = np.std(all_frame_features, axis=0)
        max_features = np.max(all_frame_features, axis=0)
        min_features = np.min(all_frame_features, axis=0)
        
        # Concatenate statistics
        aggregated = np.concatenate([mean_features, std_features, max_features, min_features])
        
        self.logger.info(f"Extracted {len(aggregated)} visual features")
        return aggregated
    
    def close(self):
        """Close and cleanup resources."""
        if self.face_mesh:
            self.face_mesh.close()
            self.logger.info("Closed MediaPipe Face Mesh")
