"""Automatic Speech Recognition Module

Uses Whisper or other ASR models to transcribe audio to text.
"""

import torch
import numpy as np
from typing import Optional, List, Dict
import logging

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    logging.warning("whisper not available")

try:
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    TRANSFORMERS_WHISPER_AVAILABLE = True
except ImportError:
    TRANSFORMERS_WHISPER_AVAILABLE = False
    logging.warning("transformers not available for Whisper")


class ASRModule:
    """Automatic Speech Recognition using Whisper."""
    
    def __init__(self, model_name: str = "base", use_transformers: bool = False):
        """
        Initialize ASR module.
        
        Args:
            model_name: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
            use_transformers: Whether to use HuggingFace transformers implementation
        """
        self.model_name = model_name
        self.use_transformers = use_transformers
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.processor = None
        
        self._load_model()
    
    def _load_model(self):
        """Load the ASR model."""
        self.logger.info(f"Loading Whisper model: {self.model_name}")
        
        if self.use_transformers and TRANSFORMERS_WHISPER_AVAILABLE:
            try:
                import torch
                model_id = f"openai/whisper-{self.model_name}"
                self.processor = WhisperProcessor.from_pretrained(model_id)
                
                # Load model with explicit device handling to avoid meta tensor issues
                self.model = WhisperForConditionalGeneration.from_pretrained(
                    model_id,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=False  # Avoid meta tensors
                )
                
                self.logger.info("Loaded Whisper from transformers")
            except Exception as e:
                self.logger.error(f"Error loading transformers Whisper: {e}")
                self.use_transformers = False
        
        if not self.use_transformers and WHISPER_AVAILABLE:
            try:
                self.model = whisper.load_model(self.model_name)
                self.logger.info("Loaded Whisper from openai-whisper")
            except Exception as e:
                self.logger.error(f"Error loading Whisper: {e}")
                self.model = None
    
    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> Dict:
        """
        Transcribe audio to text.
        
        Args:
            audio: Audio array
            sample_rate: Audio sample rate
            
        Returns:
            Dictionary with transcription results
        """
        if self.model is None:
            self.logger.error("No ASR model available")
            return {
                'text': "",
                'segments': [],
                'language': "en"
            }
        
        self.logger.info("Transcribing audio...")
        
        try:
            if self.use_transformers:
                return self._transcribe_transformers(audio, sample_rate)
            else:
                return self._transcribe_whisper(audio, sample_rate)
        except Exception as e:
            self.logger.error(f"Error during transcription: {e}")
            return {
                'text': "",
                'segments': [],
                'language': "en"
            }
    
    def _transcribe_whisper(self, audio: np.ndarray, sample_rate: int) -> Dict:
        """Transcribe using openai-whisper."""
        # Whisper expects float32 audio normalized to [-1, 1]
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Normalize if needed
        if np.abs(audio).max() > 1.0:
            audio = audio / np.abs(audio).max()
        
        result = self.model.transcribe(audio, fp16=False)
        
        return {
            'text': result['text'],
            'segments': result.get('segments', []),
            'language': result.get('language', 'en')
        }
    
    def _transcribe_transformers(self, audio: np.ndarray, sample_rate: int) -> Dict:
        """Transcribe using transformers Whisper."""
        # Ensure correct sample rate
        if sample_rate != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000
        
        # Process audio
        inputs = self.processor(
            audio,
            sampling_rate=sample_rate,
            return_tensors="pt"
        )
        
        # Generate transcription
        with torch.no_grad():
            predicted_ids = self.model.generate(inputs.input_features)
        
        # Decode
        transcription = self.processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True
        )[0]
        
        return {
            'text': transcription,
            'segments': [],  # Detailed segments not available with this method
            'language': 'en'
        }
    
    def transcribe_from_file(self, audio_path: str) -> Dict:
        """
        Transcribe audio from file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with transcription results
        """
        import librosa
        
        audio, sr = librosa.load(audio_path, sr=16000)
        return self.transcribe(audio, sr)
