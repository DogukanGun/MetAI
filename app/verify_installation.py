#!/usr/bin/env python3
"""
Installation Verification Script
Checks if all dependencies and components are properly installed.
"""

import sys
from importlib import import_module

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'

def check_module(module_name, package_name=None):
    """Check if a Python module is available."""
    try:
        import_module(module_name)
        print(f"{GREEN}✓{RESET} {package_name or module_name}")
        return True
    except ImportError:
        print(f"{RED}✗{RESET} {package_name or module_name}")
        return False

def main():
    print("=" * 60)
    print("Multimodal Emotion Recognition System")
    print("Installation Verification")
    print("=" * 60)
    print()
    
    all_ok = True
    
    # Core dependencies
    print("Checking core dependencies...")
    core_deps = [
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("sklearn", "scikit-learn"),
    ]
    
    for module, name in core_deps:
        if not check_module(module, name):
            all_ok = False
    
    print()
    
    # Web interface
    print("Checking web interface...")
    if not check_module("streamlit", "Streamlit"):
        all_ok = False
    
    print()
    
    # Audio processing
    print("Checking audio processing libraries...")
    audio_deps = [
        ("librosa", "librosa"),
        ("soundfile", "soundfile"),
    ]
    
    for module, name in audio_deps:
        if not check_module(module, name):
            all_ok = False
    
    # OpenSMILE (optional)
    if check_module("opensmile", "opensmile (optional)"):
        pass
    else:
        print(f"{YELLOW}  Note: OpenSMILE not found (optional){RESET}")
    
    print()
    
    # Visual processing
    print("Checking visual processing libraries...")
    visual_deps = [
        ("cv2", "OpenCV"),
        ("PIL", "Pillow"),
    ]
    
    for module, name in visual_deps:
        if not check_module(module, name):
            all_ok = False
    
    # MediaPipe
    if check_module("mediapipe", "MediaPipe"):
        pass
    else:
        print(f"{YELLOW}  Note: MediaPipe not found (required for facial analysis){RESET}")
        all_ok = False
    
    print()
    
    # Text processing
    print("Checking text processing libraries...")
    text_deps = [
        ("transformers", "Transformers"),
        ("nltk", "NLTK"),
    ]
    
    for module, name in text_deps:
        if not check_module(module, name):
            all_ok = False
    
    # Sentence Transformers
    if check_module("sentence_transformers", "Sentence-Transformers"):
        pass
    else:
        print(f"{YELLOW}  Note: sentence-transformers not found{RESET}")
    
    # VADER Sentiment
    if check_module("vaderSentiment", "vaderSentiment"):
        pass
    else:
        print(f"{YELLOW}  Note: vaderSentiment not found{RESET}")
    
    print()
    
    # Video processing
    print("Checking video processing libraries...")
    if check_module("moviepy.editor", "MoviePy"):
        pass
    else:
        print(f"{YELLOW}  Note: MoviePy not found (will use fallback){RESET}")
    
    print()
    
    # Data processing & visualization
    print("Checking data processing libraries...")
    data_deps = [
        ("pandas", "Pandas"),
        ("matplotlib", "Matplotlib"),
        ("seaborn", "Seaborn"),
    ]
    
    for module, name in data_deps:
        if not check_module(module, name):
            all_ok = False
    
    # Plotly (optional)
    if check_module("plotly", "Plotly (optional)"):
        pass
    else:
        print(f"{YELLOW}  Note: Plotly not found (optional){RESET}")
    
    print()
    
    # Check custom modules
    print("Checking custom modules...")
    custom_modules = [
        "modules.stage1_input",
        "modules.stage2_unimodal",
        "modules.stage3_fusion",
        "modules.stage4_output",
        "utils",
    ]
    
    for module in custom_modules:
        if not check_module(module):
            all_ok = False
    
    print()
    print("=" * 60)
    
    if all_ok:
        print(f"{GREEN}✓ All required dependencies are installed!{RESET}")
        print()
        print("You can now run the application:")
        print(f"  {YELLOW}streamlit run app.py{RESET}")
        print()
        return 0
    else:
        print(f"{RED}✗ Some dependencies are missing{RESET}")
        print()
        print("Please run:")
        print(f"  {YELLOW}pip install -r requirements.txt{RESET}")
        print()
        return 1

if __name__ == "__main__":
    sys.exit(main())
