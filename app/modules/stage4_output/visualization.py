"""Results Visualization Module

Creates various visualizations for emotion recognition results.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import logging

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logging.warning("plotly not available")


class ResultVisualizer:
    """Creates visualizations for emotion recognition results."""
    
    def __init__(self, emotion_labels: List[str], config: Optional[Dict] = None):
        """
        Initialize visualizer.
        
        Args:
            emotion_labels: List of emotion class labels
            config: Configuration dictionary (optional)
        """
        self.emotion_labels = emotion_labels
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Get emotion colors from config
        self.emotion_colors = self.config.get('emotions', {}).get('colors', {})
        if not self.emotion_colors:
            # Default colors
            self.emotion_colors = {
                'neutral': '#95a5a6',
                'happy': '#f1c40f',
                'sad': '#3498db',
                'angry': '#e74c3c',
                'fear': '#9b59b6',
                'disgust': '#16a085',
                'surprise': '#e67e22'
            }
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
    
    def plot_confusion_matrix(self, 
                             confusion_matrix: np.ndarray,
                             normalize: bool = False,
                             figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Plot confusion matrix as heatmap.
        
        Args:
            confusion_matrix: Confusion matrix array
            normalize: Whether to normalize values
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        cm = confusion_matrix.copy()
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'
        
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=self.emotion_labels,
            yticklabels=self.emotion_labels,
            ax=ax,
            cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
        )
        
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def plot_emotion_distribution(self, 
                                  emotion_counts: Dict[str, int],
                                  title: str = "Emotion Distribution") -> plt.Figure:
        """
        Plot bar chart of emotion distribution.
        
        Args:
            emotion_counts: Dictionary mapping emotions to counts
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        emotions = list(emotion_counts.keys())
        counts = list(emotion_counts.values())
        colors = [self.emotion_colors.get(e, '#95a5a6') for e in emotions]
        
        bars = ax.bar(emotions, counts, color=colors, alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontweight='bold')
        
        ax.set_xlabel('Emotion', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig
    
    def plot_emotion_timeline(self,
                             timestamps: List[float],
                             emotions: List[str],
                             confidences: Optional[List[float]] = None,
                             title: str = "Emotion Timeline") -> plt.Figure:
        """
        Plot emotion changes over time.
        
        Args:
            timestamps: List of timestamps
            emotions: List of emotion labels at each timestamp
            confidences: Optional confidence scores
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Map emotions to numeric values
        emotion_to_idx = {e: i for i, e in enumerate(self.emotion_labels)}
        emotion_values = [emotion_to_idx.get(e, 0) for e in emotions]
        
        # Create color list
        colors = [self.emotion_colors.get(e, '#95a5a6') for e in emotions]
        
        # Plot
        if confidences:
            scatter = ax.scatter(timestamps, emotion_values, 
                               c=colors, s=100, alpha=0.7,
                               edgecolors='black', linewidth=1)
            
            # Add confidence as size variation (optional)
            for i, (t, e_val, conf) in enumerate(zip(timestamps, emotion_values, confidences)):
                ax.scatter(t, e_val, s=conf*300, alpha=0.3, 
                          color=colors[i], edgecolors='none')
        else:
            ax.scatter(timestamps, emotion_values, 
                      c=colors, s=100, alpha=0.7,
                      edgecolors='black', linewidth=1)
        
        # Plot connecting lines
        ax.plot(timestamps, emotion_values, 'gray', alpha=0.3, linewidth=1)
        
        # Customize
        ax.set_yticks(range(len(self.emotion_labels)))
        ax.set_yticklabels(self.emotion_labels)
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Emotion', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_emotion_timeline_plotly(self,
                                    timestamps: List[float],
                                    emotions: List[str],
                                    confidences: Optional[List[float]] = None,
                                    title: str = "Emotion Timeline"):
        """
        Create interactive emotion timeline using Plotly.
        
        Args:
            timestamps: List of timestamps
            emotions: List of emotion labels
            confidences: Optional confidence scores
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        if not PLOTLY_AVAILABLE:
            self.logger.warning("Plotly not available")
            return None
        
        # Map emotions to numeric values
        emotion_to_idx = {e: i for i, e in enumerate(self.emotion_labels)}
        emotion_values = [emotion_to_idx.get(e, 0) for e in emotions]
        
        # Create figure
        fig = go.Figure()
        
        # Add scatter trace
        hover_text = [
            f"Time: {t:.2f}s<br>Emotion: {e}<br>Confidence: {c:.2%}" 
            if confidences else f"Time: {t:.2f}s<br>Emotion: {e}"
            for t, e, c in zip(timestamps, emotions, confidences or [0]*len(emotions))
        ]
        
        colors_list = [self.emotion_colors.get(e, '#95a5a6') for e in emotions]
        
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=emotion_values,
            mode='markers+lines',
            marker=dict(
                size=12 if not confidences else [c*20 for c in confidences],
                color=colors_list,
                line=dict(width=1, color='black')
            ),
            line=dict(color='gray', width=1),
            text=hover_text,
            hovertemplate='%{text}<extra></extra>'
        ))
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Time (seconds)",
            yaxis_title="Emotion",
            yaxis=dict(
                tickmode='array',
                tickvals=list(range(len(self.emotion_labels))),
                ticktext=self.emotion_labels
            ),
            hovermode='closest',
            template='plotly_white'
        )
        
        return fig
    
    def plot_per_class_metrics(self, 
                               per_class_metrics: Dict[str, Dict[str, float]],
                               metric_name: str = 'f1_score') -> plt.Figure:
        """
        Plot bar chart of per-class metrics.
        
        Args:
            per_class_metrics: Dictionary of per-class metrics
            metric_name: Name of metric to plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        emotions = list(per_class_metrics.keys())
        values = [per_class_metrics[e].get(metric_name, 0) for e in emotions]
        colors = [self.emotion_colors.get(e, '#95a5a6') for e in emotions]
        
        bars = ax.bar(emotions, values, color=colors, alpha=0.7, edgecolor='black')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontweight='bold')
        
        ax.set_xlabel('Emotion', fontsize=12)
        ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=12)
        ax.set_title(f'Per-Class {metric_name.replace("_", " ").title()}', 
                    fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig
    
    def create_summary_plot(self, metrics: Dict) -> plt.Figure:
        """
        Create comprehensive summary plot with multiple subplots.
        
        Args:
            metrics: Dictionary of all metrics
            
        Returns:
            Matplotlib figure with subplots
        """
        fig = plt.figure(figsize=(16, 10))
        
        # Confusion Matrix
        ax1 = plt.subplot(2, 2, 1)
        cm = metrics.get('confusion_matrix', np.zeros((len(self.emotion_labels), len(self.emotion_labels))))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.emotion_labels,
                   yticklabels=self.emotion_labels,
                   ax=ax1)
        ax1.set_title('Confusion Matrix')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('True')
        
        # Per-class F1 scores
        ax2 = plt.subplot(2, 2, 2)
        per_class = metrics.get('per_class', {})
        if per_class:
            emotions = list(per_class.keys())
            f1_scores = [per_class[e].get('f1_score', 0) for e in emotions]
            colors = [self.emotion_colors.get(e, '#95a5a6') for e in emotions]
            ax2.bar(emotions, f1_scores, color=colors, alpha=0.7, edgecolor='black')
            ax2.set_title('Per-Class F1 Score')
            ax2.set_ylabel('F1 Score')
            ax2.set_ylim(0, 1.0)
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        # Overall metrics bar chart
        ax3 = plt.subplot(2, 2, 3)
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        metric_values = [
            metrics.get('accuracy', 0),
            metrics.get('weighted_precision', 0),
            metrics.get('weighted_recall', 0),
            metrics.get('weighted_f1', 0)
        ]
        bars = ax3.bar(metric_names, metric_values, color='skyblue', alpha=0.7, edgecolor='black')
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom')
        ax3.set_title('Overall Metrics')
        ax3.set_ylim(0, 1.0)
        ax3.set_ylabel('Score')
        
        # Text summary
        ax4 = plt.subplot(2, 2, 4)
        ax4.axis('off')
        summary_text = f"""
        PERFORMANCE SUMMARY
        
        Accuracy: {metrics.get('accuracy', 0):.4f}
        
        Weighted Metrics:
        • Precision: {metrics.get('weighted_precision', 0):.4f}
        • Recall: {metrics.get('weighted_recall', 0):.4f}
        • F1-Score: {metrics.get('weighted_f1', 0):.4f}
        
        Macro Metrics:
        • Precision: {metrics.get('macro_precision', 0):.4f}
        • Recall: {metrics.get('macro_recall', 0):.4f}
        • F1-Score: {metrics.get('macro_f1', 0):.4f}
        
        Cohen's Kappa: {metrics.get('cohen_kappa', 0):.4f}
        """
        if metrics.get('roc_auc'):
            summary_text += f"\nROC-AUC: {metrics['roc_auc']:.4f}"
        
        ax4.text(0.1, 0.5, summary_text, fontsize=11, 
                verticalalignment='center', family='monospace')
        
        plt.tight_layout()
        return fig
