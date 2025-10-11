"""
Utility to download required NLTK data.

This script ensures all required NLTK resources are available.
Run this once after installation or if you encounter NLTK errors.
"""

import nltk
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_nltk_resources():
    """Download all required NLTK data packages."""
    
    resources = [
        'punkt_tab',
        'punkt',
        'vader_lexicon',
        'stopwords',
        'averaged_perceptron_tagger',
        'maxent_ne_chunker',
        'words'
    ]
    
    logger.info("Downloading NLTK resources...")
    
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
            logger.info(f"Downloaded/verified: {resource}")
        except Exception as e:
            logger.warning(f"Could not download {resource}: {e}")
    
    logger.info("NLTK resources ready!")


if __name__ == "__main__":
    download_nltk_resources()

