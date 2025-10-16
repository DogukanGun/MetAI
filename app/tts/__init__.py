"""
EmotiVoice TTS Package

Provides text-to-speech functionality using EmotiVoice engine.

Quick Start:
    from tts import text_to_speech
    
    audio_bytes = text_to_speech("Hello, world!", emotion="happy")
"""

from .model import (
    EmotiVoiceTTS,
    get_tts_engine,
    text_to_speech_simple as text_to_speech
)

__all__ = [
    'EmotiVoiceTTS',
    'get_tts_engine',
    'text_to_speech',
]

__version__ = '1.0.0'

