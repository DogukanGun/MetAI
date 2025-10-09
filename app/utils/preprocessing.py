"""Data Preprocessing Utilities"""

import numpy as np
from typing import List, Optional, Tuple
from sklearn.preprocessing import StandardScaler


def normalize_features(features: np.ndarray, 
                      scaler: Optional[StandardScaler] = None,
                      fit: bool = True) -> Tuple[np.ndarray, StandardScaler]:
    """
    Normalize features using StandardScaler.
    
    Args:
        features: Feature array [n_samples, n_features]
        scaler: Existing scaler (optional)
        fit: Whether to fit the scaler
        
    Returns:
        Tuple of (normalized_features, scaler)
    """
    if scaler is None:
        scaler = StandardScaler()
    
    if fit:
        normalized = scaler.fit_transform(features)
    else:
        normalized = scaler.transform(features)
    
    return normalized, scaler


def pad_sequences(sequences: List[np.ndarray], 
                 max_length: Optional[int] = None,
                 padding_value: float = 0.0) -> np.ndarray:
    """
    Pad sequences to same length.
    
    Args:
        sequences: List of sequences with varying lengths
        max_length: Maximum length (uses longest if None)
        padding_value: Value to use for padding
        
    Returns:
        Padded array [n_sequences, max_length, feature_dim]
    """
    if not sequences:
        return np.array([])
    
    # Determine max length
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)
    
    # Get feature dimension
    feature_dim = sequences[0].shape[-1] if sequences[0].ndim > 1 else 1
    
    # Create padded array
    padded = np.full((len(sequences), max_length, feature_dim), 
                     padding_value, dtype=np.float32)
    
    # Fill in sequences
    for i, seq in enumerate(sequences):
        seq_len = min(len(seq), max_length)
        padded[i, :seq_len] = seq[:seq_len]
    
    return padded


def chunk_audio(audio: np.ndarray, 
               sr: int,
               chunk_duration: float = 3.0,
               hop_duration: float = 1.5) -> List[np.ndarray]:
    """
    Split audio into overlapping chunks.
    
    Args:
        audio: Audio signal
        sr: Sample rate
        chunk_duration: Chunk duration in seconds
        hop_duration: Hop duration in seconds
        
    Returns:
        List of audio chunks
    """
    chunk_samples = int(chunk_duration * sr)
    hop_samples = int(hop_duration * sr)
    
    chunks = []
    for start in range(0, len(audio) - chunk_samples + 1, hop_samples):
        end = start + chunk_samples
        chunks.append(audio[start:end])
    
    # Add last chunk if needed
    if len(chunks) == 0 or len(chunks[-1]) < chunk_samples:
        if len(audio) >= chunk_samples:
            chunks.append(audio[-chunk_samples:])
    
    return chunks
