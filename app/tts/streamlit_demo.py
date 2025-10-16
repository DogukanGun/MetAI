"""
Streamlit TTS Demo

Shows how to use EmotiVoice TTS in Streamlit applications.
This can be integrated into your existing Streamlit tabs.
"""

import streamlit as st
import logging

try:
    from tts import get_tts_engine, text_to_speech
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def render_tts_demo():
    """Render TTS demo interface in Streamlit."""
    
    st.header("Text-to-Speech Demo")
    st.caption("Convert text to speech with emotional control using EmotiVoice")
    
    if not TTS_AVAILABLE:
        st.error("""
        TTS engine not available. To enable TTS:
        1. Run: `python tts/setup_emotivoice.py`
        2. Restart the Streamlit app
        """)
        return
    
    # Get TTS engine
    try:
        tts = get_tts_engine()
        
        if not tts.is_ready():
            st.warning("TTS engine is initializing... Please wait.")
            return
        
    except Exception as e:
        st.error(f"Failed to initialize TTS: {e}")
        return
    
    # Get available options
    voices = tts.get_available_voices()
    emotions = tts.get_available_emotions()
    
    # Input section
    st.subheader("Input")
    
    col1, col2 = st.columns(2)
    
    with col1:
        text_input = st.text_area(
            "Text to synthesize",
            value="Hello! This is a demo of emotional text-to-speech.",
            height=150,
            max_chars=500
        )
        
        emotion = st.selectbox(
            "Emotion",
            options=emotions,
            index=emotions.index("neutral") if "neutral" in emotions else 0
        )
    
    with col2:
        speaker = st.selectbox(
            "Voice",
            options=list(voices.keys()),
            format_func=lambda x: f"{x} - {voices[x]}"
        )
        
        speed = st.slider(
            "Speed",
            min_value=0.5,
            max_value=2.0,
            value=1.0,
            step=0.1
        )
    
    # Generate button
    if st.button("Generate Speech", type="primary"):
        if not text_input.strip():
            st.error("Please enter some text")
            return
        
        with st.spinner("Generating speech..."):
            try:
                # Generate audio
                audio_bytes = text_to_speech(
                    text=text_input,
                    emotion=emotion,
                    speaker=speaker,
                    speed=speed
                )
                
                if audio_bytes:
                    st.success("Audio generated successfully!")
                    
                    # Display audio player
                    st.audio(audio_bytes, format="audio/wav")
                    
                    # Download button
                    st.download_button(
                        label="Download Audio",
                        data=audio_bytes,
                        file_name=f"speech_{emotion}.wav",
                        mime="audio/wav"
                    )
                else:
                    st.error("Audio generation failed")
                    
            except Exception as e:
                st.error(f"Error: {e}")
                logger.error(f"TTS error: {e}", exc_info=True)
    
    # Information section
    with st.expander("Available Voices"):
        for voice_id, description in voices.items():
            st.write(f"**{voice_id}**: {description}")
    
    with st.expander("Available Emotions"):
        st.write(", ".join(emotions))
    
    with st.expander("Usage in Code"):
        st.code("""
# Simple usage
from tts import text_to_speech

audio_bytes = text_to_speech(
    text="Hello, world!",
    emotion="happy",
    speaker="8051"
)

# In Streamlit
st.audio(audio_bytes, format="audio/wav")

# Save to file
with open("output.wav", "wb") as f:
    f.write(audio_bytes)

# In FastAPI
from fastapi.responses import Response

return Response(
    content=audio_bytes,
    media_type="audio/wav"
)
        """, language="python")


if __name__ == "__main__":
    st.set_page_config(
        page_title="TTS Demo",
        page_icon=":speaker:",
        layout="wide"
    )
    
    render_tts_demo()

