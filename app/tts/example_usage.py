"""
EmotiVoice TTS Usage Examples

Demonstrates various ways to use the TTS system.
"""

import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def example_1_simple_usage():
    """Example 1: Simple text-to-speech"""
    print("\n" + "="*60)
    print("Example 1: Simple Usage")
    print("="*60)
    
    from tts import text_to_speech
    
    text = "Hello! This is a simple text-to-speech example."
    audio_bytes = text_to_speech(text)
    
    if audio_bytes:
        output_path = Path("example_output_1.wav")
        with open(output_path, "wb") as f:
            f.write(audio_bytes)
        print(f"✅ Audio saved to: {output_path}")
    else:
        print("❌ TTS failed")


def example_2_with_emotion():
    """Example 2: TTS with emotion control"""
    print("\n" + "="*60)
    print("Example 2: Emotion Control")
    print("="*60)
    
    from tts import text_to_speech
    
    emotions = ["happy", "sad", "excited"]
    text = "This sentence has different emotions!"
    
    for emotion in emotions:
        print(f"Generating with emotion: {emotion}")
        audio_bytes = text_to_speech(text, emotion=emotion)
        
        if audio_bytes:
            output_path = Path(f"example_output_2_{emotion}.wav")
            with open(output_path, "wb") as f:
                f.write(audio_bytes)
            print(f"✅ Saved: {output_path}")


def example_3_different_voices():
    """Example 3: Try different voices"""
    print("\n" + "="*60)
    print("Example 3: Different Voices")
    print("="*60)
    
    from tts import get_tts_engine
    
    tts = get_tts_engine()
    
    if not tts.is_ready():
        print("❌ TTS not ready")
        return
    
    # Get available voices
    voices = tts.get_available_voices()
    print(f"Available voices: {list(voices.keys())}")
    
    text = "Each voice has a unique character."
    
    for speaker_id in list(voices.keys())[:2]:  # Test first 2 voices
        print(f"\nGenerating with voice {speaker_id}: {voices[speaker_id]}")
        audio_bytes = tts.text_to_speech(text, speaker=speaker_id)
        
        if audio_bytes:
            output_path = Path(f"example_output_3_voice_{speaker_id}.wav")
            with open(output_path, "wb") as f:
                f.write(audio_bytes)
            print(f"✅ Saved: {output_path}")


def example_4_speed_control():
    """Example 4: Control speech speed"""
    print("\n" + "="*60)
    print("Example 4: Speed Control")
    print("="*60)
    
    from tts import text_to_speech
    
    text = "This demonstrates speed control in text-to-speech."
    speeds = [0.5, 1.0, 1.5]
    
    for speed in speeds:
        print(f"Generating at speed: {speed}x")
        audio_bytes = text_to_speech(text, speed=speed)
        
        if audio_bytes:
            output_path = Path(f"example_output_4_speed_{speed}.wav")
            with open(output_path, "wb") as f:
                f.write(audio_bytes)
            print(f"✅ Saved: {output_path}")


def example_5_batch_processing():
    """Example 5: Batch process multiple texts"""
    print("\n" + "="*60)
    print("Example 5: Batch Processing")
    print("="*60)
    
    from tts import get_tts_engine
    
    tts = get_tts_engine()
    
    texts = [
        ("Welcome to the demo.", "happy"),
        ("This is very important.", "neutral"),
        ("Please be careful!", "worried"),
    ]
    
    for i, (text, emotion) in enumerate(texts, 1):
        print(f"Processing {i}/{len(texts)}: {text[:30]}...")
        audio_bytes = tts.text_to_speech(text, emotion=emotion)
        
        if audio_bytes:
            output_path = Path(f"example_output_5_batch_{i}.wav")
            with open(output_path, "wb") as f:
                f.write(audio_bytes)
            print(f"✅ Saved: {output_path}")


def example_6_check_availability():
    """Example 6: Check TTS availability"""
    print("\n" + "="*60)
    print("Example 6: Check Availability")
    print("="*60)
    
    from tts import get_tts_engine
    
    tts = get_tts_engine()
    
    print(f"TTS Ready: {tts.is_ready()}")
    print(f"Available Emotions: {tts.get_available_emotions()}")
    print(f"Available Voices: {len(tts.get_available_voices())} voices")
    
    for voice_id, description in list(tts.get_available_voices().items())[:3]:
        print(f"  - {voice_id}: {description}")


def example_7_error_handling():
    """Example 7: Proper error handling"""
    print("\n" + "="*60)
    print("Example 7: Error Handling")
    print("="*60)
    
    from tts import text_to_speech
    
    # Empty text
    audio = text_to_speech("")
    print(f"Empty text result: {audio}")
    
    # Very long text
    long_text = "A" * 1000
    audio = text_to_speech(long_text)
    print(f"Long text result: {'Success' if audio else 'Failed'}")
    
    # Invalid emotion
    audio = text_to_speech("Test", emotion="invalid_emotion")
    print(f"Invalid emotion result: {'Success' if audio else 'Failed'}")


def main():
    """Run all examples"""
    print("\n" + "="*60)
    print("EmotiVoice TTS Usage Examples")
    print("="*60)
    
    try:
        # Check if TTS is available
        from tts import get_tts_engine
        tts = get_tts_engine()
        
        if not tts.is_ready():
            print("\n⚠️  TTS not fully initialized.")
            print("Run: python tts/setup_emotivoice.py")
            return
        
        # Run examples
        example_1_simple_usage()
        example_2_with_emotion()
        example_3_different_voices()
        example_4_speed_control()
        example_5_batch_processing()
        example_6_check_availability()
        example_7_error_handling()
        
        print("\n" + "="*60)
        print("✅ All examples completed!")
        print("Check the generated WAV files in the current directory.")
        print("="*60 + "\n")
        
    except Exception as e:
        logger.error(f"Example failed: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        print("Make sure to run: python tts/setup_emotivoice.py first")


if __name__ == "__main__":
    main()

