"""
Evaluation Metrics
==================

Metrics for evaluating model performance.
"""

import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)
from scipy.stats import spearmanr, pearsonr
from typing import Dict, Optional


def compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    task: str = "regression",
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute evaluation metrics.
    
    Parameters
    ----------
    predictions : np.ndarray
        Model predictions
    targets : np.ndarray
        Ground truth targets
    task : str
        Task type: 'regression' or 'classification'
    threshold : float
        Threshold for binary classification
        
    Returns
    -------
    dict
        Dictionary of metrics
    """
    if task == "regression":
        return compute_regression_metrics(predictions, targets)
    elif task == "classification":
        return compute_classification_metrics(predictions, targets, threshold)
    else:
        raise ValueError(f"Unknown task type: {task}")


def compute_regression_metrics(
    predictions: np.ndarray,
    targets: np.ndarray
) -> Dict[str, float]:
    """
    Compute regression metrics.
    
    Parameters
    ----------
    predictions : np.ndarray
        Predicted values
    targets : np.ndarray
        True values
        
    Returns
    -------
    dict
        Dictionary of regression metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['mse'] = mean_squared_error(targets, predictions)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    metrics['mae'] = mean_absolute_error(targets, predictions)
    metrics['r2'] = r2_score(targets, predictions)
    
    # Correlation metrics
    pearson_corr, _ = pearsonr(predictions, targets)
    spearman_corr, _ = spearmanr(predictions, targets)
    
    metrics['pearson'] = pearson_corr
    metrics['spearman'] = spearman_corr
    
    return metrics


def compute_classification_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Parameters
    ----------
    predictions : np.ndarray
        Predicted probabilities or logits
    targets : np.ndarray
        True labels
    threshold : float
        Threshold for binary classification
        
    Returns
    -------
    dict
        Dictionary of classification metrics
    """
    metrics = {}
    
    # Handle binary vs multi-class
    if predictions.ndim == 1:
        # Binary classification
        pred_probs = predictions
        pred_labels = (predictions > threshold).astype(int)
        
        metrics['accuracy'] = accuracy_score(targets, pred_labels)
        metrics['precision'] = precision_score(targets, pred_labels, zero_division=0)
        metrics['recall'] = recall_score(targets, pred_labels, zero_division=0)
        metrics['f1'] = f1_score(targets, pred_labels, zero_division=0)
        
        try:
            metrics['auroc'] = roc_auc_score(targets, pred_probs)
            metrics['auprc'] = average_precision_score(targets, pred_probs)
        except ValueError:
            # Handle cases with single class in targets
            metrics['auroc'] = 0.0
            metrics['auprc'] = 0.0
    
    else:
        # Multi-class classification
        pred_labels = np.argmax(predictions, axis=1)
        
        metrics['accuracy'] = accuracy_score(targets, pred_labels)
        metrics['precision'] = precision_score(targets, pred_labels, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(targets, pred_labels, average='weighted', zero_division=0)
        metrics['f1'] = f1_score(targets, pred_labels, average='weighted', zero_division=0)
        
        try:
            metrics['auroc'] = roc_auc_score(targets, predictions, average='weighted', multi_class='ovr')
        except ValueError:
            metrics['auroc'] = 0.0
    
    return metrics


def compute_drug_response_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    drug_ids: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute drug response-specific metrics.
    
    Parameters
    ----------
    predictions : np.ndarray
        Predicted drug responses
    targets : np.ndarray
        True drug responses
    drug_ids : np.ndarray, optional
        Drug identifiers for per-drug analysis
        
    Returns
    -------
    dict
        Dictionary of drug response metrics
    """
    metrics = compute_regression_metrics(predictions, targets)
    
    # Add drug-specific metrics if drug IDs provided
    if drug_ids is not None:
        unique_drugs = np.unique(drug_ids)
        per_drug_correlations = []
        
        for drug in unique_drugs:
            mask = drug_ids == drug
            if mask.sum() > 1:  # Need at least 2 samples
                drug_preds = predictions[mask]
                drug_targets = targets[mask]
                corr, _ = spearmanr(drug_preds, drug_targets)
                if not np.isnan(corr):
                    per_drug_correlations.append(corr)
        
        if per_drug_correlations:
            metrics['mean_per_drug_spearman'] = np.mean(per_drug_correlations)
            metrics['median_per_drug_spearman'] = np.median(per_drug_correlations)
    
    return metrics


def compute_embedding_quality_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray
) -> Dict[str, float]:
    """
    Compute metrics for embedding quality.
    
    Parameters
    ----------
    embeddings : np.ndarray
        Learned embeddings
    labels : np.ndarray
        Labels for evaluation
        
    Returns
    -------
    dict
        Dictionary of embedding quality metrics
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    
    metrics = {}
    
    # Linear probing accuracy
    clf = LogisticRegression(max_iter=1000, random_state=42)
    scores = cross_val_score(clf, embeddings, labels, cv=5)
    
    metrics['linear_probe_acc_mean'] = scores.mean()
    metrics['linear_probe_acc_std'] = scores.std()
    
    # Silhouette score
    from sklearn.metrics import silhouette_score
    try:
        metrics['silhouette'] = silhouette_score(embeddings, labels)
    except:
        metrics['silhouette'] = 0.0
    
    return metrics


def compute_ranking_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    k: int = 10
) -> Dict[str, float]:
    """
    Compute ranking metrics.
    
    Parameters
    ----------
    predictions : np.ndarray
        Predicted scores
    targets : np.ndarray
        True scores or binary relevance
    k : int
        Top-k for ranking metrics
        
    Returns
    -------
    dict
        Dictionary of ranking metrics
    """
    metrics = {}
    
    # Sort by predictions
    sorted_indices = np.argsort(predictions)[::-1]
    sorted_targets = targets[sorted_indices]
    
    # Precision@K
    metrics[f'precision@{k}'] = sorted_targets[:k].mean()
    
    # Recall@K (for binary targets)
    if np.all(np.isin(targets, [0, 1])):
        relevant = targets.sum()
        if relevant > 0:
            metrics[f'recall@{k}'] = sorted_targets[:k].sum() / relevant
    
    # NDCG@K
    dcg_k = np.sum(sorted_targets[:k] / np.log2(np.arange(2, k + 2)))
    ideal_targets = np.sort(targets)[::-1]
    idcg_k = np.sum(ideal_targets[:k] / np.log2(np.arange(2, k + 2)))
    
    if idcg_k > 0:
        metrics[f'ndcg@{k}'] = dcg_k / idcg_k
    else:
        metrics[f'ndcg@{k}'] = 0.0
    
    return metrics
