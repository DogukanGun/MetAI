"""Emotion Recognition Metrics Module

Computes various evaluation metrics for emotion classification.
"""

import numpy as np
from typing import Dict, List, Optional
import logging

try:
    from sklearn.metrics import (
        accuracy_score, 
        precision_recall_fscore_support,
        confusion_matrix,
        classification_report,
        cohen_kappa_score,
        roc_auc_score
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available, metrics will be limited")


class EmotionMetrics:
    """Computes evaluation metrics for emotion recognition."""
    
    def __init__(self, emotion_labels: List[str]):
        """
        Initialize metrics calculator.
        
        Args:
            emotion_labels: List of emotion class labels
        """
        self.emotion_labels = emotion_labels
        self.n_classes = len(emotion_labels)
        self.logger = logging.getLogger(__name__)
    
    def compute_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute classification accuracy.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            
        Returns:
            Accuracy score
        """
        if not SKLEARN_AVAILABLE:
            correct = np.sum(y_true == y_pred)
            return float(correct / len(y_true))
        
        return float(accuracy_score(y_true, y_pred))
    
    def compute_precision_recall_f1(self, 
                                    y_true: np.ndarray, 
                                    y_pred: np.ndarray,
                                    average: str = 'weighted') -> Dict[str, float]:
        """
        Compute precision, recall, and F1-score.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            average: Averaging strategy ('micro', 'macro', 'weighted')
            
        Returns:
            Dictionary with metrics
        """
        if not SKLEARN_AVAILABLE:
            self.logger.warning("scikit-learn not available")
            return {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}
        
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=average, zero_division=0
        )
        
        return {
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'support': int(np.sum(support)) if hasattr(support, '__len__') else int(support)
        }
    
    def compute_per_class_metrics(self, 
                                   y_true: np.ndarray, 
                                   y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Compute metrics for each emotion class.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary mapping emotion names to their metrics
        """
        if not SKLEARN_AVAILABLE:
            return {}
        
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        per_class = {}
        for i, emotion in enumerate(self.emotion_labels):
            per_class[emotion] = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1_score': float(f1[i]),
                'support': int(support[i])
            }
        
        return per_class
    
    def compute_confusion_matrix(self, 
                                 y_true: np.ndarray, 
                                 y_pred: np.ndarray) -> np.ndarray:
        """
        Compute confusion matrix.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            
        Returns:
            Confusion matrix
        """
        if not SKLEARN_AVAILABLE:
            # Simple confusion matrix implementation
            cm = np.zeros((self.n_classes, self.n_classes), dtype=int)
            for true, pred in zip(y_true, y_pred):
                cm[int(true), int(pred)] += 1
            return cm
        
        return confusion_matrix(y_true, y_pred)
    
    def compute_cohen_kappa(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute Cohen's Kappa score.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            
        Returns:
            Kappa score
        """
        if not SKLEARN_AVAILABLE:
            self.logger.warning("Cohen's Kappa requires scikit-learn")
            return 0.0
        
        return float(cohen_kappa_score(y_true, y_pred))
    
    def compute_roc_auc(self, 
                       y_true: np.ndarray, 
                       y_prob: np.ndarray,
                       multi_class: str = 'ovr') -> Optional[float]:
        """
        Compute ROC-AUC score.
        
        Args:
            y_true: Ground truth labels
            y_prob: Predicted probabilities [n_samples, n_classes]
            multi_class: Strategy for multiclass ('ovr' or 'ovo')
            
        Returns:
            ROC-AUC score or None if not available
        """
        if not SKLEARN_AVAILABLE:
            self.logger.warning("ROC-AUC requires scikit-learn")
            return None
        
        try:
            auc = roc_auc_score(
                y_true, 
                y_prob, 
                multi_class=multi_class,
                average='weighted'
            )
            return float(auc)
        except Exception as e:
            self.logger.error(f"Error computing ROC-AUC: {e}")
            return None
    
    def compute_all_metrics(self, 
                           y_true: np.ndarray, 
                           y_pred: np.ndarray,
                           y_prob: Optional[np.ndarray] = None) -> Dict:
        """
        Compute all available metrics.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (optional)
            
        Returns:
            Dictionary with all metrics
        """
        self.logger.info("Computing all metrics")
        
        metrics = {}
        
        # Overall metrics
        metrics['accuracy'] = self.compute_accuracy(y_true, y_pred)
        
        # Weighted metrics
        weighted = self.compute_precision_recall_f1(y_true, y_pred, average='weighted')
        metrics['weighted_precision'] = weighted['precision']
        metrics['weighted_recall'] = weighted['recall']
        metrics['weighted_f1'] = weighted['f1_score']
        
        # Macro metrics
        macro = self.compute_precision_recall_f1(y_true, y_pred, average='macro')
        metrics['macro_precision'] = macro['precision']
        metrics['macro_recall'] = macro['recall']
        metrics['macro_f1'] = macro['f1_score']
        
        # Per-class metrics
        metrics['per_class'] = self.compute_per_class_metrics(y_true, y_pred)
        
        # Confusion matrix
        metrics['confusion_matrix'] = self.compute_confusion_matrix(y_true, y_pred)
        
        # Cohen's Kappa
        metrics['cohen_kappa'] = self.compute_cohen_kappa(y_true, y_pred)
        
        # ROC-AUC if probabilities available
        if y_prob is not None:
            metrics['roc_auc'] = self.compute_roc_auc(y_true, y_prob)
        
        return metrics
    
    def get_classification_report(self, 
                                  y_true: np.ndarray, 
                                  y_pred: np.ndarray) -> str:
        """
        Get detailed classification report.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            
        Returns:
            Classification report as string
        """
        if not SKLEARN_AVAILABLE:
            return "Classification report requires scikit-learn"
        
        return classification_report(
            y_true, 
            y_pred, 
            target_names=self.emotion_labels,
            zero_division=0
        )
