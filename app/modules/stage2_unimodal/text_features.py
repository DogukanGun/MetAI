"""Text Feature Extraction Module

Extracts emotion-relevant features from transcribed text:
- Lexical features (word count, sentence length)
- Sentiment scores
- Semantic embeddings (SBERT, BERT)
- Emotion lexicon features
- Part-of-speech statistics
"""

import numpy as np
from typing import Dict, List, Optional
import logging
import re

try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False
    logging.warning("sentence-transformers not available")

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    logging.warning("vaderSentiment not available")

try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    NLTK_AVAILABLE = True
    
    # Auto-download required NLTK resources if missing
    def _ensure_nltk_resources():
        resources = [
            'punkt_tab', 'punkt', 'vader_lexicon', 'stopwords', 
            'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng',
            'omw-1.4'
        ]
        for resource in resources:
            try:
                nltk.download(resource, quiet=True)
            except:
                pass
    
    _ensure_nltk_resources()
    
except ImportError:
    NLTK_AVAILABLE = False
    logging.warning("nltk not available")


class TextFeatureExtractor:
    """Extracts emotion-relevant features from text."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize text feature extractor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize SBERT model
        self.sbert_model = None
        if SBERT_AVAILABLE:
            try:
                model_name = self.config.get('embedding_model', 'all-MiniLM-L6-v2')
                self.sbert_model = SentenceTransformer(model_name)
                self.logger.info(f"Loaded SBERT model: {model_name}")
            except Exception as e:
                self.logger.warning(f"Could not load SBERT model: {e}")
        
        # Initialize VADER sentiment analyzer
        self.sia = None
        if VADER_AVAILABLE:
            try:
                self.sia = SentimentIntensityAnalyzer()
                self.logger.info("VADER sentiment analyzer initialized")
            except Exception as e:
                self.logger.warning(f"Could not initialize VADER: {e}")
        
        # Download required NLTK data
        if NLTK_AVAILABLE:
            try:
                nltk.download('punkt', quiet=True)
                nltk.download('averaged_perceptron_tagger', quiet=True)
                nltk.download('stopwords', quiet=True)
            except Exception as e:
                self.logger.warning(f"Could not download NLTK data: {e}")
        
        # Emotion lexicons
        self.emotion_keywords = {
            'happy': ['happy', 'joy', 'excited', 'delighted', 'pleased', 'cheerful', 'glad'],
            'sad': ['sad', 'unhappy', 'depressed', 'down', 'miserable', 'sorrowful'],
            'angry': ['angry', 'mad', 'furious', 'irritated', 'annoyed', 'rage'],
            'fear': ['afraid', 'scared', 'frightened', 'terrified', 'anxious', 'worried'],
            'disgust': ['disgusted', 'repulsed', 'revolted', 'sick', 'nauseated'],
            'surprise': ['surprised', 'amazed', 'astonished', 'shocked', 'startled']
        }
    
    def extract_lexical_features(self, text: str) -> Dict[str, float]:
        """
        Extract basic lexical features.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of lexical features
        """
        features = {}
        
        if not text or len(text.strip()) == 0:
            return {
                'word_count': 0,
                'char_count': 0,
                'sentence_count': 0,
                'avg_word_length': 0,
                'avg_sentence_length': 0
            }
        
        # Basic counts
        words = text.split()
        features['word_count'] = len(words)
        features['char_count'] = len(text)
        
        # Sentence count
        if NLTK_AVAILABLE:
            try:
                sentences = sent_tokenize(text)
                features['sentence_count'] = len(sentences)
                features['avg_sentence_length'] = len(words) / len(sentences) if sentences else 0
            except:
                features['sentence_count'] = text.count('.') + text.count('!') + text.count('?')
                features['avg_sentence_length'] = len(words) / features['sentence_count'] if features['sentence_count'] > 0 else 0
        else:
            features['sentence_count'] = text.count('.') + text.count('!') + text.count('?')
            features['avg_sentence_length'] = len(words) / features['sentence_count'] if features['sentence_count'] > 0 else 0
        
        # Average word length
        features['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0
        
        # Punctuation counts
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['comma_count'] = text.count(',')
        
        # Uppercase ratio
        uppercase_count = sum(1 for c in text if c.isupper())
        features['uppercase_ratio'] = uppercase_count / len(text) if len(text) > 0 else 0
        
        return features
    
    def extract_sentiment(self, text: str) -> Dict[str, float]:
        """
        Extract sentiment scores using VADER.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of sentiment scores
        """
        if not text or self.sia is None:
            return {
                'sentiment_neg': 0.0,
                'sentiment_neu': 0.0,
                'sentiment_pos': 0.0,
                'sentiment_compound': 0.0
            }
        
        try:
            scores = self.sia.polarity_scores(text)
            return {
                'sentiment_neg': scores['neg'],
                'sentiment_neu': scores['neu'],
                'sentiment_pos': scores['pos'],
                'sentiment_compound': scores['compound']
            }
        except Exception as e:
            self.logger.error(f"Error extracting sentiment: {e}")
            return {
                'sentiment_neg': 0.0,
                'sentiment_neu': 0.0,
                'sentiment_pos': 0.0,
                'sentiment_compound': 0.0
            }
    
    def extract_emotion_lexicon_features(self, text: str) -> Dict[str, float]:
        """
        Extract features based on emotion keyword matching.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of emotion keyword counts
        """
        text_lower = text.lower()
        features = {}
        
        for emotion, keywords in self.emotion_keywords.items():
            count = sum(text_lower.count(keyword) for keyword in keywords)
            features[f'emotion_keyword_{emotion}'] = float(count)
        
        # Total emotion words
        total_emotion_words = sum(features.values())
        features['total_emotion_keywords'] = float(total_emotion_words)
        
        return features
    
    def extract_pos_features(self, text: str) -> Dict[str, float]:
        """
        Extract Part-of-Speech features.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of POS tag counts
        """
        features = {
            'noun_count': 0,
            'verb_count': 0,
            'adj_count': 0,
            'adv_count': 0
        }
        
        if not NLTK_AVAILABLE or not text:
            return features
        
        try:
            tokens = word_tokenize(text)
            pos_tags = nltk.pos_tag(tokens)
            
            for word, tag in pos_tags:
                if tag.startswith('NN'):  # Nouns
                    features['noun_count'] += 1
                elif tag.startswith('VB'):  # Verbs
                    features['verb_count'] += 1
                elif tag.startswith('JJ'):  # Adjectives
                    features['adj_count'] += 1
                elif tag.startswith('RB'):  # Adverbs
                    features['adv_count'] += 1
            
            # Normalize by word count
            word_count = len(tokens) if tokens else 1
            features['noun_ratio'] = features['noun_count'] / word_count
            features['verb_ratio'] = features['verb_count'] / word_count
            features['adj_ratio'] = features['adj_count'] / word_count
            features['adv_ratio'] = features['adv_count'] / word_count
            
        except Exception as e:
            self.logger.error(f"Error extracting POS features: {e}")
        
        return features
    
    def extract_semantic_embeddings(self, text: str) -> np.ndarray:
        """
        Extract semantic embeddings using SBERT.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        if not text or self.sbert_model is None:
            # Return zero vector of appropriate size
            return np.zeros(384, dtype=np.float32)
        
        try:
            embedding = self.sbert_model.encode(text, convert_to_numpy=True)
            return embedding.astype(np.float32)
        except Exception as e:
            self.logger.error(f"Error extracting embeddings: {e}")
            return np.zeros(384, dtype=np.float32)
    
    def extract_all_features(self, text: str) -> np.ndarray:
        """
        Extract all text features and return as a single vector.
        
        Args:
            text: Input text
            
        Returns:
            Feature vector as numpy array
        """
        if not text:
            text = ""
        
        self.logger.info("Extracting all text features")
        
        all_features = []
        
        # Extract lexical features
        lexical = self.extract_lexical_features(text)
        all_features.extend(lexical.values())
        
        # Extract sentiment features
        sentiment = self.extract_sentiment(text)
        all_features.extend(sentiment.values())
        
        # Extract emotion lexicon features
        emotion_lex = self.extract_emotion_lexicon_features(text)
        all_features.extend(emotion_lex.values())
        
        # Extract POS features
        pos = self.extract_pos_features(text)
        all_features.extend(pos.values())
        
        # Extract semantic embeddings
        embeddings = self.extract_semantic_embeddings(text)
        all_features.extend(embeddings)
        
        feature_vector = np.array(all_features, dtype=np.float32)
        
        self.logger.info(f"Extracted {len(feature_vector)} text features")
        return feature_vector
    
    def get_feature_names(self) -> List[str]:
        """
        Get names of all extracted features.
        
        Returns:
            List of feature names
        """
        names = [
            'word_count', 'char_count', 'sentence_count',
            'avg_word_length', 'avg_sentence_length',
            'exclamation_count', 'question_count', 'comma_count',
            'uppercase_ratio',
            'sentiment_neg', 'sentiment_neu', 'sentiment_pos', 'sentiment_compound'
        ]
        
        # Emotion lexicon features
        for emotion in self.emotion_keywords.keys():
            names.append(f'emotion_keyword_{emotion}')
        names.append('total_emotion_keywords')
        
        # POS features
        names.extend([
            'noun_count', 'verb_count', 'adj_count', 'adv_count',
            'noun_ratio', 'verb_ratio', 'adj_ratio', 'adv_ratio'
        ])
        
        # Embeddings (numbered)
        embedding_dim = 384  # Default SBERT dimension
        names.extend([f'embedding_{i}' for i in range(embedding_dim)])
        
        return names
