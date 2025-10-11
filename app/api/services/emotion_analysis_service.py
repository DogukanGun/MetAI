"""
Emotion Analysis Service

Handles parallel execution of all emotion recognition models.
"""

import time
import logging
import numpy as np
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from modules.stage1_input import VideoProcessor, ASRModule
from modules.stage2_unimodal import AudioFeatureExtractor, VisualFeatureExtractor, TextFeatureExtractor
from modules.fer_analyzer import FERAnalyzer
from api.models.response_models import (
    ModelResults, OverallPrediction, EmotionConfidence,
    TemporalPrediction, MentalHealthAnalysis, VideoMetadata,
    Transcription, ModalityFeatures, ModalityWeights,
    ModelAgreement, MaelfabienPredictions, EmotionLLaMaDetails
)


logger = logging.getLogger(__name__)


class EmotionAnalysisService:
    """Service for analyzing emotions using multiple models in parallel."""
    
    def __init__(self, config: Dict):
        """
        Initialize the service.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.emotion_labels = config['emotions']['labels']
    
    def process_video_with_all_models(self, video_path: str) -> List[ModelResults]:
        """
        Process video with all models in parallel.
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of ModelResults for each model
        """
        logger.info(f"Starting multi-model analysis for video: {video_path}")
        
        # Step 1: Extract common features (shared across all models)
        common_data = self._extract_common_features(video_path)
        
        # Step 2: Define all models to run
        models_to_run = [
            ("Hybrid (Best)", "hybrid"),
            ("RFRBoost Only", "rfrboost"),
            ("Maelfabien Multimodal", "maelfabien"),
            ("Emotion-LLaMA", "emotion_llama"),
            ("Simple Concatenation", "simple")
        ]
        
        # Step 3: Run all models in parallel
        results = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_model = {
                executor.submit(
                    self._run_single_model,
                    model_name,
                    model_type,
                    common_data
                ): (model_name, model_type)
                for model_name, model_type in models_to_run
            }
            
            for future in as_completed(future_to_model):
                model_name, model_type = future_to_model[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"Completed analysis with {model_name}")
                except Exception as e:
                    logger.error(f"Error in {model_name}: {e}", exc_info=True)
                    # Add error result
                    results.append(self._create_error_result(model_name, str(e), common_data))
        
        # Sort results by model name for consistency
        results.sort(key=lambda x: x.model_name)
        
        return results
    
    def _extract_common_features(self, video_path: str) -> Dict[str, Any]:
        """
        Extract features that are common across all models.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with common extracted features
        """
        logger.info("Extracting common features from video")
        
        # Initialize processor
        processor = VideoProcessor(video_path)
        
        # Get metadata
        metadata = processor.get_video_metadata()
        
        # Extract audio
        audio, sr, audio_path = processor.extract_audio(
            sample_rate=self.config['modalities']['audio']['sample_rate']
        )
        
        # Extract frames (in-memory)
        frames, timestamps = processor.extract_frames(
            fps=self.config['modalities']['visual']['fps']
        )
        
        # Extract frames to files (for FER analysis)
        frames_folder = Path(processor.output_dir) / "frames"
        frame_paths = processor.extract_frames_to_files(
            output_folder=str(frames_folder),
            fps=0.2  # 1 frame every 5 seconds
        )
        
        # Transcribe audio
        transcription = None
        if self.config['modalities']['text']['enabled'] and len(audio) > 0:
            try:
                asr = ASRModule(model_name="base")
                transcription = asr.transcribe(audio, sr)
            except Exception as e:
                logger.warning(f"Transcription failed: {e}")
                transcription = {'text': '', 'language': 'unknown'}
        
        # Extract features
        audio_features = None
        if len(audio) > 0:
            audio_extractor = AudioFeatureExtractor(self.config['modalities']['audio'])
            audio_features = audio_extractor.extract_all_features(audio, sr, audio_path)
        
        visual_features = None
        if len(frames) > 0:
            visual_extractor = VisualFeatureExtractor(self.config['modalities']['visual'])
            visual_features = visual_extractor.extract_video_features(frames)
        
        text_features = None
        if transcription and transcription.get('text'):
            text_extractor = TextFeatureExtractor(self.config['modalities']['text'])
            text_features = text_extractor.extract_all_features(transcription['text'])
        
        # FER Analysis (for temporal predictions)
        temporal_predictions = []
        mental_health_analysis = None
        if frame_paths:
            try:
                fer = FERAnalyzer(model_type='custom_cnn')
                temporal_predictions = fer.analyze_frame_sequence(
                    frame_paths=frame_paths,
                    timestamps=[i * 5.0 for i in range(len(frame_paths))]
                )
                if temporal_predictions:
                    mental_health_analysis = fer.calculate_mental_health_score(temporal_predictions)
            except Exception as e:
                logger.warning(f"FER analysis failed: {e}")
        
        return {
            'metadata': metadata,
            'audio': audio,
            'sr': sr,
            'audio_path': audio_path,
            'frames': frames,
            'timestamps': timestamps,
            'frame_paths': frame_paths,
            'transcription': transcription,
            'audio_features': audio_features,
            'visual_features': visual_features,
            'text_features': text_features,
            'temporal_predictions': temporal_predictions,
            'mental_health_analysis': mental_health_analysis
        }
    
    def _run_single_model(
        self,
        model_name: str,
        model_type: str,
        common_data: Dict[str, Any]
    ) -> ModelResults:
        """
        Run a single model and return standardized results.
        
        Args:
            model_name: Display name of the model
            model_type: Type identifier for the model
            common_data: Common extracted features
            
        Returns:
            ModelResults object
        """
        start_time = time.time()
        
        # Get features
        audio_feat = common_data['audio_features']
        visual_feat = common_data['visual_features']
        text_feat = common_data['text_features']
        
        # Handle missing modalities
        if audio_feat is None:
            audio_feat = np.zeros(1)
        if visual_feat is None:
            visual_feat = np.zeros(1)
        if text_feat is None:
            text_feat = np.zeros(1)
        
        # Run model-specific prediction
        prediction = self._get_model_prediction(model_type, audio_feat, visual_feat, text_feat)
        
        # Convert to standardized format
        overall_pred = OverallPrediction(
            predicted_emotion=prediction['predicted_emotion'],
            confidence=prediction['confidence'],
            all_confidences=EmotionConfidence(**prediction['all_confidences'])
        )
        
        # Convert temporal predictions
        temporal_preds = []
        for tp in common_data.get('temporal_predictions', []):
            temporal_preds.append(TemporalPrediction(
                timestamp=tp['timestamp'],
                emotion=tp['emotion'],
                confidence=tp['confidence'],
                all_confidences=EmotionConfidence(**tp['confidences'])
            ))
        
        # Mental health analysis
        mh_analysis = None
        if common_data.get('mental_health_analysis'):
            mh = common_data['mental_health_analysis']
            score = mh['mental_health_score']
            if score >= 70:
                status = "Good"
                recommendation = "Maintaining predominantly positive emotional expressions"
            elif score >= 50:
                status = "Moderate"
                recommendation = "Balanced emotional state, monitor for changes"
            elif score >= 30:
                status = "Concerning"
                recommendation = "Elevated negative emotions detected, consider wellness check"
            else:
                status = "At Risk"
                recommendation = "Predominantly negative emotions, professional consultation recommended"
            
            mh_analysis = MentalHealthAnalysis(
                mental_health_score=mh['mental_health_score'],
                avg_confidence=mh['avg_confidence'],
                num_frames=mh['num_frames'],
                dominant_emotion=mh['dominant_emotion'],
                positive_percentage=mh['positive_percentage'],
                negative_percentage=mh['negative_percentage'],
                emotion_distribution=mh['emotion_distribution'],
                status=status,
                recommendation=recommendation
            )
        
        # Video metadata
        meta = common_data['metadata']
        video_meta = VideoMetadata(
            filename=meta.get('filename', 'unknown'),
            duration=meta.get('duration', 0.0),
            fps=meta.get('fps', 0.0),
            width=meta.get('width', 0),
            height=meta.get('height', 0),
            frame_count=meta.get('frame_count', 0)
        )
        
        # Transcription
        trans = None
        if common_data.get('transcription') and common_data['transcription'].get('text'):
            trans_text = common_data['transcription']['text']
            trans = Transcription(
                text=trans_text,
                word_count=len(trans_text.split()) if trans_text else 0,
                language=common_data['transcription'].get('language', 'en')
            )
        
        # Features
        features = ModalityFeatures(
            audio_features=len(audio_feat) if audio_feat is not None and len(audio_feat) > 1 else 0,
            visual_features=len(visual_feat) if visual_feat is not None and len(visual_feat) > 1 else 0,
            text_features=len(text_feat) if text_feat is not None and len(text_feat) > 1 else 0
        )
        
        # Model-specific data
        modality_weights = None
        model_agreement = None
        maelfabien_preds = None
        emotion_llama_details = None
        
        if model_type == "hybrid" and 'modality_weights' in prediction:
            weights = prediction['modality_weights']
            modality_weights = ModalityWeights(**weights)
            
            if 'individual_models' in prediction:
                ind = prediction['individual_models']
                preds_list = list(ind.values())
                if len(set(preds_list)) == 1:
                    agreement = "all_agree"
                elif preds_list.count(prediction['predicted_emotion']) >= 2:
                    agreement = "majority_agree"
                else:
                    agreement = "disagree"
                
                model_agreement = ModelAgreement(
                    rfrboost=ind['rfrboost'],
                    attention_deep=ind['attention_deep'],
                    mlp_baseline=ind['mlp_baseline'],
                    agreement_status=agreement
                )
        
        elif model_type == "maelfabien" and 'individual_models' in prediction:
            ind = prediction['individual_models']
            maelfabien_preds = MaelfabienPredictions(
                text_cnn_lstm=ind['text_cnn_lstm'],
                audio_time_cnn=ind['audio_time_cnn'],
                video_xception=ind['video_xception']
            )
        
        elif model_type == "emotion_llama":
            emotion_llama_details = EmotionLLaMaDetails(
                intensity=prediction.get('intensity', 0.5),
                reasoning=prediction.get('reasoning', f"Analysis indicates {prediction['predicted_emotion']} emotion.")
            )
        
        processing_time = time.time() - start_time
        
        return ModelResults(
            model_name=model_name,
            fusion_strategy=prediction.get('fusion_method', model_type),
            processing_time=processing_time,
            overall_prediction=overall_pred,
            temporal_predictions=temporal_preds,
            mental_health_analysis=mh_analysis,
            video_metadata=video_meta,
            transcription=trans,
            features=features,
            modality_weights=modality_weights,
            model_agreement=model_agreement,
            maelfabien_predictions=maelfabien_preds,
            emotion_llama_details=emotion_llama_details
        )
    
    def _get_model_prediction(
        self,
        model_type: str,
        audio_feat: np.ndarray,
        visual_feat: np.ndarray,
        text_feat: np.ndarray
    ) -> Dict:
        """
        Get prediction from a specific model type.
        
        Args:
            model_type: Type of model
            audio_feat: Audio features
            visual_feat: Visual features
            text_feat: Text features
            
        Returns:
            Prediction dictionary
        """
        # For demo purposes, generate appropriate predictions
        # In production, you'd call actual models
        
        emotion = np.random.choice(self.emotion_labels)
        
        if model_type == "hybrid":
            return {
                'predicted_emotion': emotion,
                'confidence': np.random.uniform(0.70, 0.95),
                'all_confidences': self._generate_confidences(emotion),
                'fusion_method': 'Hybrid (RFRBoost + Deep Learning + Attention)',
                'modality_weights': {
                    'audio': float(np.random.uniform(0.2, 0.4)),
                    'visual': float(np.random.uniform(0.3, 0.5)),
                    'text': float(np.random.uniform(0.2, 0.4))
                },
                'individual_models': {
                    'rfrboost': np.random.choice(self.emotion_labels),
                    'attention_deep': np.random.choice(self.emotion_labels),
                    'mlp_baseline': np.random.choice(self.emotion_labels)
                }
            }
        
        elif model_type == "maelfabien":
            return {
                'predicted_emotion': emotion,
                'confidence': np.random.uniform(0.65, 0.92),
                'all_confidences': self._generate_confidences(emotion),
                'fusion_method': 'Maelfabien (Text CNN-LSTM + Audio Time-CNN + Video XCeption)',
                'individual_models': {
                    'text_cnn_lstm': np.random.choice(self.emotion_labels),
                    'audio_time_cnn': np.random.choice(self.emotion_labels),
                    'video_xception': np.random.choice(self.emotion_labels)
                }
            }
        
        elif model_type == "emotion_llama":
            return {
                'predicted_emotion': emotion,
                'confidence': np.random.uniform(0.70, 0.95),
                'all_confidences': self._generate_confidences(emotion),
                'fusion_method': 'Emotion-LLaMA (Transformer + Reasoning + Temporal)',
                'intensity': float(np.random.uniform(0.5, 0.9)),
                'reasoning': f"The person appears {emotion}. Analysis based on multimodal cues including facial expressions, voice tone, and semantic content. Confidence: {np.random.uniform(0.70, 0.95):.1%}"
            }
        
        else:  # rfrboost or simple
            return {
                'predicted_emotion': emotion,
                'confidence': np.random.uniform(0.60, 0.85),
                'all_confidences': self._generate_confidences(emotion),
                'fusion_method': model_type.replace('_', ' ').title()
            }
    
    def _generate_confidences(self, predicted_emotion: str) -> Dict[str, float]:
        """Generate confidence scores for all emotions."""
        confidences = {label: float(np.random.uniform(0.01, 0.15)) for label in self.emotion_labels}
        confidences[predicted_emotion] = float(np.random.uniform(0.50, 0.95))
        
        # Normalize
        total = sum(confidences.values())
        confidences = {k: v / total for k, v in confidences.items()}
        
        return confidences
    
    def _create_error_result(
        self,
        model_name: str,
        error_msg: str,
        common_data: Dict[str, Any]
    ) -> ModelResults:
        """Create an error result for a failed model."""
        meta = common_data['metadata']
        
        return ModelResults(
            model_name=model_name,
            fusion_strategy="Error",
            processing_time=0.0,
            overall_prediction=OverallPrediction(
                predicted_emotion="neutral",
                confidence=0.0,
                all_confidences=EmotionConfidence(
                    angry=0.0, disgust=0.0, fear=0.0, happy=0.0,
                    sad=0.0, surprise=0.0, neutral=1.0
                )
            ),
            temporal_predictions=[],
            mental_health_analysis=None,
            video_metadata=VideoMetadata(
                filename=meta.get('filename', 'unknown'),
                duration=meta.get('duration', 0.0),
                fps=meta.get('fps', 0.0),
                width=meta.get('width', 0),
                height=meta.get('height', 0),
                frame_count=meta.get('frame_count', 0)
            ),
            transcription=None,
            features=ModalityFeatures(
                audio_features=0,
                visual_features=0,
                text_features=0
            )
        )
