"""Results Export Module

Exports emotion recognition results to various formats.
"""

import json
import csv
import os
from typing import Dict, List, Optional
import logging
import pandas as pd
from datetime import datetime


class ResultExporter:
    """Exports emotion recognition results to various formats."""
    
    def __init__(self, output_dir: str = "data/results"):
        """
        Initialize exporter.
        
        Args:
            output_dir: Directory to save exported files
        """
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def export_to_json(self, 
                      results: Dict,
                      filename: Optional[str] = None) -> str:
        """
        Export results to JSON format.
        
        Args:
            results: Results dictionary
            filename: Output filename (auto-generated if None)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"emotion_results_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = self._prepare_for_json(results)
        
        with open(filepath, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        self.logger.info(f"Results exported to JSON: {filepath}")
        return filepath
    
    def export_to_csv(self, 
                     timeline_data: List[Dict],
                     filename: Optional[str] = None) -> str:
        """
        Export timeline data to CSV format.
        
        Args:
            timeline_data: List of dictionaries with timestamp, emotion, confidence
            filename: Output filename (auto-generated if None)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"emotion_timeline_{timestamp}.csv"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Convert to DataFrame
        df = pd.DataFrame(timeline_data)
        df.to_csv(filepath, index=False)
        
        self.logger.info(f"Timeline exported to CSV: {filepath}")
        return filepath
    
    def export_metrics_to_csv(self, 
                             metrics: Dict,
                             filename: Optional[str] = None) -> str:
        """
        Export metrics to CSV format.
        
        Args:
            metrics: Metrics dictionary
            filename: Output filename (auto-generated if None)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"emotion_metrics_{timestamp}.csv"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Create a flattened version for CSV
        csv_data = []
        
        # Overall metrics
        csv_data.append({
            'metric': 'Accuracy',
            'value': metrics.get('accuracy', 0)
        })
        csv_data.append({
            'metric': 'Weighted Precision',
            'value': metrics.get('weighted_precision', 0)
        })
        csv_data.append({
            'metric': 'Weighted Recall',
            'value': metrics.get('weighted_recall', 0)
        })
        csv_data.append({
            'metric': 'Weighted F1',
            'value': metrics.get('weighted_f1', 0)
        })
        csv_data.append({
            'metric': 'Cohen\'s Kappa',
            'value': metrics.get('cohen_kappa', 0)
        })
        
        # Per-class metrics
        per_class = metrics.get('per_class', {})
        for emotion, emotion_metrics in per_class.items():
            for metric_name, value in emotion_metrics.items():
                csv_data.append({
                    'metric': f'{emotion}_{metric_name}',
                    'value': value
                })
        
        df = pd.DataFrame(csv_data)
        df.to_csv(filepath, index=False)
        
        self.logger.info(f"Metrics exported to CSV: {filepath}")
        return filepath
    
    def create_detailed_report(self, 
                              results: Dict,
                              metrics: Dict,
                              filename: Optional[str] = None) -> str:
        """
        Create a detailed text report.
        
        Args:
            results: Results dictionary
            metrics: Metrics dictionary
            filename: Output filename (auto-generated if None)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"emotion_report_{timestamp}.txt"
        
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write("="*70 + "\n")
            f.write("MULTIMODAL EMOTION RECOGNITION REPORT\n")
            f.write("="*70 + "\n\n")
            
            # Timestamp
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Video information
            if 'video_metadata' in results:
                f.write("VIDEO INFORMATION\n")
                f.write("-" * 70 + "\n")
                metadata = results['video_metadata']
                f.write(f"Filename: {metadata.get('filename', 'N/A')}\n")
                f.write(f"Duration: {metadata.get('duration', 0):.2f} seconds\n")
                f.write(f"Resolution: {metadata.get('width', 0)}x{metadata.get('height', 0)}\n")
                f.write(f"FPS: {metadata.get('fps', 0):.2f}\n\n")
            
            # Overall performance
            f.write("OVERALL PERFORMANCE\n")
            f.write("-" * 70 + "\n")
            f.write(f"Accuracy: {metrics.get('accuracy', 0):.4f}\n")
            f.write(f"Weighted Precision: {metrics.get('weighted_precision', 0):.4f}\n")
            f.write(f"Weighted Recall: {metrics.get('weighted_recall', 0):.4f}\n")
            f.write(f"Weighted F1-Score: {metrics.get('weighted_f1', 0):.4f}\n")
            f.write(f"Cohen's Kappa: {metrics.get('cohen_kappa', 0):.4f}\n")
            if metrics.get('roc_auc'):
                f.write(f"ROC-AUC: {metrics['roc_auc']:.4f}\n")
            f.write("\n")
            
            # Per-class metrics
            f.write("PER-CLASS METRICS\n")
            f.write("-" * 70 + "\n")
            per_class = metrics.get('per_class', {})
            if per_class:
                f.write(f"{'Emotion':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}\n")
                f.write("-" * 70 + "\n")
                for emotion, em_metrics in per_class.items():
                    f.write(f"{emotion:<15} "
                           f"{em_metrics.get('precision', 0):<12.4f} "
                           f"{em_metrics.get('recall', 0):<12.4f} "
                           f"{em_metrics.get('f1_score', 0):<12.4f} "
                           f"{em_metrics.get('support', 0):<10}\n")
            f.write("\n")
            
            # Confusion matrix
            f.write("CONFUSION MATRIX\n")
            f.write("-" * 70 + "\n")
            if 'confusion_matrix' in metrics:
                cm = metrics['confusion_matrix']
                emotion_labels = results.get('emotion_labels', [])
                
                # Header
                f.write(" " * 15)
                for label in emotion_labels:
                    f.write(f"{label[:8]:<10}")
                f.write("\n")
                
                # Rows
                for i, label in enumerate(emotion_labels):
                    f.write(f"{label[:15]:<15}")
                    for j in range(len(emotion_labels)):
                        f.write(f"{cm[i, j]:<10}")
                    f.write("\n")
            
            f.write("\n" + "="*70 + "\n")
        
        self.logger.info(f"Detailed report created: {filepath}")
        return filepath
    
    def _prepare_for_json(self, obj):
        """
        Recursively prepare object for JSON serialization.
        
        Args:
            obj: Object to prepare
            
        Returns:
            JSON-serializable object
        """
        import numpy as np
        
        if isinstance(obj, dict):
            return {key: self._prepare_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._prepare_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        else:
            return obj
