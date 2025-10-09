"""Audio Feature Extraction Module

Extracts comprehensive audio features including:
- Prosodic features (pitch, energy, speaking rate)
- Spectral features (MFCCs, spectral statistics)
- Voice quality features (jitter, shimmer, HNR)
- OpenSMILE feature sets
"""

import numpy as np
from typing import Dict, Optional, List
import logging

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logging.warning("librosa not available")

try:
    import opensmile
    OPENSMILE_AVAILABLE = True
except ImportError:
    OPENSMILE_AVAILABLE = False
    logging.warning("opensmile not available")


class AudioFeatureExtractor:
    """Extracts emotion-relevant features from audio."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize audio feature extractor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize OpenSMILE if available
        self.smile = None
        if OPENSMILE_AVAILABLE and self.config.get('extract_opensmile', True):
            try:
                self.smile = opensmile.Smile(
                    feature_set=opensmile.FeatureSet.eGeMAPSv02,
                    feature_level=opensmile.FeatureLevel.Functionals,
                )
                self.logger.info("OpenSMILE initialized with eGeMAPSv02")
            except Exception as e:
                self.logger.warning(f"Could not initialize OpenSMILE: {e}")
    
    def extract_prosodic_features(self, audio: np.ndarray, sr: int) -> Dict[str, float]:
        """
        Extract prosodic features (pitch, energy, speaking rate).
        
        Args:
            audio: Audio signal
            sr: Sample rate
            
        Returns:
            Dictionary of prosodic features
        """
        if not LIBROSA_AVAILABLE:
            self.logger.warning("librosa not available, skipping prosodic features")
            return {}
        
        features = {}
        
        try:
            # Pitch (F0) using piptrack
            pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
            pitch_values = pitches[pitches > 0]
            
            if len(pitch_values) > 0:
                features['pitch_mean'] = float(np.mean(pitch_values))
                features['pitch_std'] = float(np.std(pitch_values))
                features['pitch_min'] = float(np.min(pitch_values))
                features['pitch_max'] = float(np.max(pitch_values))
                features['pitch_range'] = features['pitch_max'] - features['pitch_min']
            else:
                features.update({
                    'pitch_mean': 0.0, 'pitch_std': 0.0,
                    'pitch_min': 0.0, 'pitch_max': 0.0, 'pitch_range': 0.0
                })
            
            # Energy/RMS
            energy = librosa.feature.rms(y=audio)[0]
            features['energy_mean'] = float(np.mean(energy))
            features['energy_std'] = float(np.std(energy))
            features['energy_max'] = float(np.max(energy))
            
            # Zero crossing rate (proxy for speech rate)
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            features['zcr_mean'] = float(np.mean(zcr))
            features['zcr_std'] = float(np.std(zcr))
            
            # Spectral centroid (brightness)
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
            features['spectral_centroid_std'] = float(np.std(spectral_centroids))
            
        except Exception as e:
            self.logger.error(f"Error extracting prosodic features: {e}")
        
        return features
    
    def extract_spectral_features(self, audio: np.ndarray, sr: int, n_mfcc: int = 40) -> Dict[str, float]:
        """
        Extract spectral features including MFCCs.
        
        Args:
            audio: Audio signal
            sr: Sample rate
            n_mfcc: Number of MFCC coefficients
            
        Returns:
            Dictionary of spectral features
        """
        if not LIBROSA_AVAILABLE:
            self.logger.warning("librosa not available, skipping spectral features")
            return {}
        
        features = {}
        
        try:
            # MFCCs
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
            mfcc_mean = np.mean(mfccs, axis=1)
            mfcc_std = np.std(mfccs, axis=1)
            
            # Add MFCC statistics
            for i in range(n_mfcc):
                features[f'mfcc_{i}_mean'] = float(mfcc_mean[i])
                features[f'mfcc_{i}_std'] = float(mfcc_std[i])
            
            # Spectral rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
            features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
            features['spectral_rolloff_std'] = float(np.std(spectral_rolloff))
            
            # Spectral bandwidth
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
            features['spectral_bandwidth_mean'] = float(np.mean(spectral_bandwidth))
            features['spectral_bandwidth_std'] = float(np.std(spectral_bandwidth))
            
            # Spectral contrast
            spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
            for i in range(spectral_contrast.shape[0]):
                features[f'spectral_contrast_{i}_mean'] = float(np.mean(spectral_contrast[i]))
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
            for i in range(12):
                features[f'chroma_{i}_mean'] = float(np.mean(chroma[i]))
            
        except Exception as e:
            self.logger.error(f"Error extracting spectral features: {e}")
        
        return features
    
    def extract_voice_quality_features(self, audio: np.ndarray, sr: int) -> Dict[str, float]:
        """
        Extract voice quality features (jitter, shimmer, HNR).
        
        Args:
            audio: Audio signal
            sr: Sample rate
            
        Returns:
            Dictionary of voice quality features
        """
        features = {}
        
        # Note: Jitter and shimmer calculation requires specialized tools
        # This is a simplified implementation
        
        try:
            if LIBROSA_AVAILABLE:
                # Harmonic-to-Noise Ratio approximation using harmonic and percussive separation
                y_harmonic, y_percussive = librosa.effects.hpss(audio)
                
                # Calculate energy ratio as HNR proxy
                harmonic_energy = np.sum(y_harmonic ** 2)
                percussive_energy = np.sum(y_percussive ** 2)
                
                if percussive_energy > 0:
                    hnr_proxy = 10 * np.log10(harmonic_energy / percussive_energy)
                    features['hnr_proxy'] = float(hnr_proxy)
                else:
                    features['hnr_proxy'] = 0.0
        
        except Exception as e:
            self.logger.error(f"Error extracting voice quality features: {e}")
        
        return features
    
    def extract_opensmile_features(self, audio_path: str) -> Optional[np.ndarray]:
        """
        Extract features using OpenSMILE toolkit.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Feature vector as numpy array
        """
        if self.smile is None:
            self.logger.warning("OpenSMILE not initialized")
            return None
        
        try:
            features = self.smile.process_file(audio_path)
            return features.values[0]  # Return as numpy array
        except Exception as e:
            self.logger.error(f"Error extracting OpenSMILE features: {e}")
            return None
    
    def extract_temporal_features(self, audio: np.ndarray, sr: int, 
                                  window_size: float = 3.0, 
                                  hop_length: float = 1.5) -> List[Dict]:
        """
        Extract features from temporal windows.
        
        Args:
            audio: Audio signal
            sr: Sample rate
            window_size: Window size in seconds
            hop_length: Hop length in seconds
            
        Returns:
            List of feature dictionaries for each window
        """
        window_samples = int(window_size * sr)
        hop_samples = int(hop_length * sr)
        
        temporal_features = []
        
        for start in range(0, len(audio) - window_samples, hop_samples):
            end = start + window_samples
            window_audio = audio[start:end]
            
            # Extract features for this window
            window_features = {}
            window_features.update(self.extract_prosodic_features(window_audio, sr))
            window_features.update(self.extract_spectral_features(window_audio, sr, n_mfcc=13))
            window_features['timestamp'] = start / sr
            
            temporal_features.append(window_features)
        
        return temporal_features
    
    def extract_all_features(self, audio: np.ndarray, sr: int, 
                           audio_path: Optional[str] = None) -> np.ndarray:
        """
        Extract all audio features and return as a single vector.
        
        Args:
            audio: Audio signal
            sr: Sample rate
            audio_path: Path to audio file (for OpenSMILE)
            
        Returns:
            Feature vector as numpy array
        """
        self.logger.info("Extracting all audio features")
        
        all_features = []
        
        # Extract prosodic features
        prosodic = self.extract_prosodic_features(audio, sr)
        all_features.extend(prosodic.values())
        
        # Extract spectral features
        spectral = self.extract_spectral_features(audio, sr, n_mfcc=40)
        all_features.extend(spectral.values())
        
        # Extract voice quality features
        voice_quality = self.extract_voice_quality_features(audio, sr)
        all_features.extend(voice_quality.values())
        
        # Extract OpenSMILE features if available
        if audio_path and self.smile:
            opensmile_features = self.extract_opensmile_features(audio_path)
            if opensmile_features is not None:
                all_features.extend(opensmile_features)
        
        feature_vector = np.array(all_features, dtype=np.float32)
        
        self.logger.info(f"Extracted {len(feature_vector)} audio features")
        return feature_vector
    
    def get_feature_names(self, n_mfcc: int = 40) -> List[str]:
        """
        Get names of all extracted features.
        
        Args:
            n_mfcc: Number of MFCC coefficients
            
        Returns:
            List of feature names
        """
        names = [
            'pitch_mean', 'pitch_std', 'pitch_min', 'pitch_max', 'pitch_range',
            'energy_mean', 'energy_std', 'energy_max',
            'zcr_mean', 'zcr_std',
            'spectral_centroid_mean', 'spectral_centroid_std'
        ]
        
        # MFCC names
        for i in range(n_mfcc):
            names.append(f'mfcc_{i}_mean')
            names.append(f'mfcc_{i}_std')
        
        names.extend([
            'spectral_rolloff_mean', 'spectral_rolloff_std',
            'spectral_bandwidth_mean', 'spectral_bandwidth_std'
        ])
        
        # Spectral contrast
        for i in range(7):
            names.append(f'spectral_contrast_{i}_mean')
        
        # Chroma
        for i in range(12):
            names.append(f'chroma_{i}_mean')
        
        names.append('hnr_proxy')
        
        return names
