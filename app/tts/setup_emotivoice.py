"""
Setup script for EmotiVoice TTS engine.

This script clones EmotiVoice repository and downloads required models
to ensure the TTS system is self-contained and always available.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmotiVoiceSetup:
    """Setup EmotiVoice TTS system."""
    
    def __init__(self, base_dir: str = None):
        """
        Initialize setup.
        
        Args:
            base_dir: Base directory for EmotiVoice installation
        """
        if base_dir is None:
            base_dir = Path(__file__).parent / "EmotiVoice"
        self.base_dir = Path(base_dir)
        self.repo_url = "https://github.com/netease-youdao/EmotiVoice.git"
        self.models_url = "https://www.modelscope.cn/syq163/outputs.git"
        self.simbert_url = "https://www.modelscope.cn/syq163/WangZeJun.git"
    
    def is_installed(self) -> bool:
        """Check if EmotiVoice is already installed."""
        required_paths = [
            self.base_dir / "inference_am_vocoder_joint.py",
            self.base_dir / "outputs",
            self.base_dir / "WangZeJun",
        ]
        return all(path.exists() for path in required_paths)
    
    def run_command(self, cmd: list, cwd: str = None) -> bool:
        """
        Run shell command.
        
        Args:
            cmd: Command to run as list
            cwd: Working directory
            
        Returns:
            True if successful, False otherwise
        """
        try:
            result = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                check=True
            )
            logger.info(f"Command successful: {' '.join(cmd)}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed: {' '.join(cmd)}")
            logger.error(f"Error: {e.stderr}")
            return False
    
    def clone_repository(self) -> bool:
        """Clone EmotiVoice repository."""
        if self.base_dir.exists():
            logger.info(f"EmotiVoice directory already exists at {self.base_dir}")
            return True
        
        logger.info("Cloning EmotiVoice repository...")
        parent_dir = self.base_dir.parent
        parent_dir.mkdir(parents=True, exist_ok=True)
        
        return self.run_command(
            ["git", "clone", self.repo_url, str(self.base_dir)],
            cwd=str(parent_dir)
        )
    
    def download_models(self) -> bool:
        """Download pre-trained models."""
        logger.info("Downloading pre-trained models...")
        
        # Download main models
        models_dir = self.base_dir / "outputs"
        if not models_dir.exists():
            success = self.run_command(
                ["git", "clone", self.models_url, "outputs"],
                cwd=str(self.base_dir)
            )
            if not success:
                return False
        
        # Download SimBERT model
        simbert_dir = self.base_dir / "WangZeJun"
        if not simbert_dir.exists():
            success = self.run_command(
                ["git", "clone", self.simbert_url, "WangZeJun"],
                cwd=str(self.base_dir)
            )
            if not success:
                return False
        
        return True
    
    def install_dependencies(self) -> bool:
        """Install EmotiVoice dependencies."""
        logger.info("Installing EmotiVoice dependencies...")
        
        requirements = [
            "numpy",
            "numba",
            "scipy",
            "transformers",
            "soundfile",
            "yacs",
            "g2p_en",
            "jieba",
            "pypinyin",
            "pypinyin_dict",
        ]
        
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install"] + requirements,
                check=True,
                capture_output=True
            )
            
            # Download NLTK data
            import nltk
            nltk.download('averaged_perceptron_tagger_eng', quiet=True)
            
            logger.info("Dependencies installed successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to install dependencies: {e}")
            return False
    
    def setup(self) -> bool:
        """
        Run complete setup.
        
        Returns:
            True if setup successful, False otherwise
        """
        logger.info("Starting EmotiVoice setup...")
        
        # Check if already installed
        if self.is_installed():
            logger.info("EmotiVoice is already installed and ready!")
            return True
        
        # Clone repository
        if not self.clone_repository():
            logger.error("Failed to clone repository")
            return False
        
        # Install dependencies
        if not self.install_dependencies():
            logger.error("Failed to install dependencies")
            return False
        
        # Download models
        if not self.download_models():
            logger.error("Failed to download models")
            return False
        
        logger.info("EmotiVoice setup completed successfully!")
        return True


if __name__ == "__main__":
    setup = EmotiVoiceSetup()
    success = setup.setup()
    sys.exit(0 if success else 1)

