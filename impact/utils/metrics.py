"""
Evaluation metrics for IMPACT model.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from typing import Dict, List, Union

def compute_metrics(
    y_true: Union[List, np.ndarray],
    y_pred: Union[List, np.ndarray]
) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary of metrics
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Basic metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }
    
    # ROC AUC (only if predictions are probabilities)
    if len(np.unique(y_pred)) > 2:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_pred)
        except:
            metrics['auc'] = 0.5
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Additional metrics
    metrics.update({
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'ppv': tp / (tp + fp) if (tp + fp) > 0 else 0,  # Positive predictive value
        'npv': tn / (tn + fn) if (tn + fn) > 0 else 0   # Negative predictive value
    })
    
    return metrics

def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95
) -> Dict[str, Dict[str, float]]:
    """
    Compute bootstrap confidence intervals for metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level
        
    Returns:
        Dictionary of metrics with confidence intervals
    """
    n_samples = len(y_true)
    bootstrap_metrics = []
    
    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.randint(0, n_samples, n_samples)
        sample_true = y_true[indices]
        sample_pred = y_pred[indices]
        
        # Compute metrics for this sample
        metrics = compute_metrics(sample_true, sample_pred)
        bootstrap_metrics.append(metrics)
    
    # Compute confidence intervals
    alpha = (1 - confidence) / 2
    ci_metrics = {}
    
    for metric in bootstrap_metrics[0].keys():
        values = [m[metric] for m in bootstrap_metrics]
        ci_metrics[metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'lower': np.percentile(values, alpha * 100),
            'upper': np.percentile(values, (1 - alpha) * 100)
        }
    
    return ci_metrics
