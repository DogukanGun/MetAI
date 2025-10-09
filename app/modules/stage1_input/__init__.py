"""Stage 1: Input Processing Module

This module handles video ingestion, audio extraction, frame extraction,
and automatic speech recognition (ASR).
"""

from .video_processor import VideoProcessor
from .asr_module import ASRModule

__all__ = ['VideoProcessor', 'ASRModule']
