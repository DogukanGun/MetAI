"""
Multimodal Emotion Recognition System - Streamlit UI

Main application interface for video emotion recognition using RFRBoost.
"""

import streamlit as st
import os
import sys
import logging

# Add app directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import modules
from modules.stage2_unimodal import AudioFeatureExtractor, VisualFeatureExtractor, TextFeatureExtractor
from utils.helpers import load_config, setup_logging, format_duration
from tabs import render_upload_tab, render_results_tab, render_help_tab, render_knowledge_tab

# Page configuration
st.set_page_config(
    page_title="Multimodal Emotion Recognition",
    page_icon=":material/psychology:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_app_config():
    """Load application configuration."""
    try:
        config = load_config("config/config.yaml")
        return config
    except Exception as e:
        st.error(f"Error loading config: {e}")
        return None


@st.cache_resource
def initialize_extractors(config):
    """Initialize feature extractors."""
    logger = logging.getLogger(__name__)
    logger.info("Initializing feature extractors")
    
    extractors = {}
    
    try:
        if config['modalities']['audio']['enabled']:
            extractors['audio'] = AudioFeatureExtractor(config['modalities']['audio'])
        
        if config['modalities']['visual']['enabled']:
            extractors['visual'] = VisualFeatureExtractor(config['modalities']['visual'])
        
        if config['modalities']['text']['enabled']:
            extractors['text'] = TextFeatureExtractor(config['modalities']['text'])
            
        logger.info(f"Initialized {len(extractors)} feature extractors")
        return extractors
    
    except Exception as e:
        logger.error(f"Error initializing extractors: {e}")
        st.error(f"Error initializing extractors: {e}")
        return {}


def main():
    """Main application."""
    
    # Setup logging
    logger = setup_logging(log_level="INFO")
    
    # Load configuration
    config = load_app_config()
    if config is None:
        st.error("Failed to load configuration. Please check config/config.yaml")
        return
    
    # Header
    st.markdown('<div class="main-header">Multimodal Emotion Recognition System</div>', 
                unsafe_allow_html=True)
    st.markdown("**Powered by Random Feature Representation Boosting (RFRBoost)**")
    st.markdown("---")
    
    # Sidebar configuration
    with st.sidebar:
        st.title("Configuration")
        
        st.subheader("Fusion Strategy")
        fusion_strategy = st.selectbox(
            "Select Fusion Method",
            [
                "Hybrid (Best)",
                "RFRBoost Only",
                "Maelfabien Multimodal",
                "Emotion-LLaMA",
                "Simple Concatenation"
            ],
            help="Choose the emotion recognition approach"
        )
        
        st.subheader("Modalities")
        use_audio = st.checkbox("Audio", value=config['modalities']['audio']['enabled'])
        use_visual = st.checkbox("Visual", value=config['modalities']['visual']['enabled'])
        use_text = st.checkbox("Text", value=config['modalities']['text']['enabled'])
        
        # Update config
        config['modalities']['audio']['enabled'] = use_audio
        config['modalities']['visual']['enabled'] = use_visual
        config['modalities']['text']['enabled'] = use_text
        
        st.subheader("Model Parameters")
        n_layers = st.slider("Number of Layers", 1, 10, config['rfrboost']['n_layers'])
        hidden_dim = st.slider("Hidden Dimension", 64, 512, config['rfrboost']['hidden_dim'], step=64)
        boost_lr = st.slider("Boost Learning Rate", 0.1, 1.0, config['rfrboost']['boost_lr'], step=0.1)
        
        # Update config
        config['rfrboost']['n_layers'] = n_layers
        config['rfrboost']['hidden_dim'] = hidden_dim
        config['rfrboost']['boost_lr'] = boost_lr
        config['fusion_strategy'] = fusion_strategy
        
        st.markdown("---")
        st.subheader("AI Agent Configuration")
        
        # LLM Provider Selection
        llm_provider = st.selectbox(
            "LLM Provider",
            options=["Cloud (OpenAI)", "Local (Ollama/LM Studio)"],
            help="Choose between cloud-based OpenAI or local LLM"
        )
        
        # Store in session state for use in tabs
        st.session_state['llm_provider'] = llm_provider
        
        if llm_provider == "Cloud (OpenAI)":
            st.info("""
            **Cloud Provider (OpenAI)**
            - Uses GPT-4 or GPT-3.5-Turbo
            - Requires OPENAI_API_KEY in .env
            - Best quality analysis
            - Pay per use
            """)
        else:
            st.info("""
            **Local Provider (Ollama/LM Studio)**
            - Uses locally running models
            - Free and private
            - Requires local LLM server running
            - Configure LOCAL_LLM_BASE_URL in .env
            """)
        
        st.markdown("---")
        st.subheader("About")
        
        if fusion_strategy == "Hybrid (Best)":
            st.success("""
            **Hybrid Fusion** (Selected)
            
            Combines 3 powerful approaches:
            - RFRBoost (40%): Robust tabular learning
            - Attention+Deep (35%): Modality fusion
            - MLP Baseline (25%): Simple patterns
            
            Provides: Modality weights, model agreement, best accuracy!
            """)
        else:
            st.info("""
            This system analyzes videos to detect emotions using:
            - Audio features (prosody, spectral)
            - Visual features (facial expressions)
            - Text features (sentiment, semantics)
            
            Try **Hybrid Fusion** for best results!
            """)
    
    # Initialize extractors
    with st.spinner("Initializing feature extractors..."):
        extractors = initialize_extractors(config)
    
    # Main content - Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Upload & Analyze", "Results", "Knowledge Base", "Help"])
    
    with tab1:
        render_upload_tab(config, extractors)
    
    with tab2:
        render_results_tab()
    
    with tab3:
        render_knowledge_tab()
    
    with tab4:
        render_help_tab(config)


if __name__ == "__main__":
    main()