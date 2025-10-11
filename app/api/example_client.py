"""
Example Client for Emotion Recognition API

Shows how to use the API from Python.
"""

import requests
import json
from pathlib import Path


class EmotionAPIClient:
    """Client for Emotion Recognition API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize client.
        
        Args:
            base_url: Base URL of the API
        """
        self.base_url = base_url
        self.api_base = f"{base_url}/api/v1/emotion"
    
    def health_check(self) -> dict:
        """Check API health."""
        response = requests.get(f"{self.api_base}/health")
        return response.json()
    
    def list_models(self) -> dict:
        """Get available models."""
        response = requests.get(f"{self.api_base}/models")
        return response.json()
    
    def analyze_video(self, video_path: str) -> dict:
        """
        Analyze emotions in a video.
        
        Args:
            video_path: Path to video file
            
        Returns:
            API response dictionary
        """
        with open(video_path, 'rb') as video_file:
            files = {'video': video_file}
            response = requests.post(
                f"{self.api_base}/analyze",
                files=files
            )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API Error: {response.status_code} - {response.text}")
    
    def print_results(self, results: dict):
        """Pretty print results."""
        print("\n" + "="*80)
        print(f"SUCCESS: {results['success']}")
        print(f"MESSAGE: {results['message']}")
        print(f"TOTAL PROCESSING TIME: {results['total_processing_time']:.2f}s")
        print("="*80)
        
        for i, model_result in enumerate(results['results'], 1):
            print(f"\n{'='*80}")
            print(f"MODEL {i}: {model_result['model_name']}")
            print(f"{'='*80}")
            
            # Overall prediction
            overall = model_result['overall_prediction']
            print(f"\nOVERALL PREDICTION:")
            print(f"  Emotion: {overall['predicted_emotion'].upper()}")
            print(f"  Confidence: {overall['confidence']:.1%}")
            
            # Mental health analysis
            if model_result.get('mental_health_analysis'):
                mh = model_result['mental_health_analysis']
                print(f"\nMENTAL HEALTH ANALYSIS:")
                print(f"  Score: {mh['mental_health_score']:.1f}/100")
                print(f"  Status: {mh['status']}")
                print(f"  Dominant Emotion: {mh['dominant_emotion'].title()}")
                print(f"  Positive: {mh['positive_percentage']:.1f}%")
                print(f"  Negative: {mh['negative_percentage']:.1f}%")
            
            # Transcription
            if model_result.get('transcription'):
                trans = model_result['transcription']
                print(f"\nTRANSCRIPTION:")
                print(f"  Words: {trans['word_count']}")
                print(f"  Text: {trans['text'][:100]}...")
            
            # Model-specific
            if model_result.get('modality_weights'):
                weights = model_result['modality_weights']
                print(f"\nMODALITY WEIGHTS:")
                print(f"  Audio: {weights['audio']:.1%}")
                print(f"  Visual: {weights['visual']:.1%}")
                print(f"  Text: {weights['text']:.1%}")
            
            if model_result.get('model_agreement'):
                agreement = model_result['model_agreement']
                print(f"\nMODEL AGREEMENT:")
                print(f"  Status: {agreement['agreement_status']}")
            
            print(f"\nPROCESSING TIME: {model_result['processing_time']:.2f}s")


def main():
    """Main example function."""
    # Initialize client
    client = EmotionAPIClient()
    
    # Health check
    print("Checking API health...")
    health = client.health_check()
    print(f"API Status: {health['status']}")
    
    # List models
    print("\nAvailable models:")
    models = client.list_models()
    for model in models['models']:
        print(f"  - {model['name']}: {model['description']}")
    
    # Analyze video (replace with actual video path)
    video_path = "path/to/your/video.mp4"
    
    if Path(video_path).exists():
        print(f"\nAnalyzing video: {video_path}")
        print("This may take 45-60 seconds...")
        
        results = client.analyze_video(video_path)
        client.print_results(results)
        
        # Save results to file
        output_file = "emotion_analysis_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")
    else:
        print(f"\nVideo not found: {video_path}")
        print("Please update the video_path variable with a valid video file.")


if __name__ == "__main__":
    main()
