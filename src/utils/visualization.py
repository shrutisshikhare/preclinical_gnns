"""
Visualization utilities.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
from typing import Optional, List


def plot_training_curves(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    save_path: Optional[str] = None
):
    """
    Plot training and validation loss curves.
    
    Parameters
    ----------
    train_losses : list
        Training losses per epoch
    val_losses : list, optional
        Validation losses per epoch
    save_path : str, optional
        Path to save figure
    """
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    
    if val_losses:
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_embeddings(
    embeddings: np.ndarray,
    labels: Optional[np.ndarray] = None,
    method: str = "umap",
    save_path: Optional[str] = None,
    title: str = "Embeddings Visualization"
):
    """
    Plot 2D visualization of embeddings.
    
    Parameters
    ----------
    embeddings : np.ndarray
        High-dimensional embeddings [n_samples, embedding_dim]
    labels : np.ndarray, optional
        Labels for coloring points
    method : str
        Dimensionality reduction method: 'umap', 'tsne', or 'pca'
    save_path : str, optional
        Path to save figure
    title : str
        Plot title
    """
    # Reduce to 2D
    if method == "umap":
        reducer = umap.UMAP(n_components=2, random_state=42)
        embeddings_2d = reducer.fit_transform(embeddings)
    elif method == "tsne":
        reducer = TSNE(n_components=2, random_state=42)
        embeddings_2d = reducer.fit_transform(embeddings)
    elif method == "pca":
        reducer = PCA(n_components=2, random_state=42)
        embeddings_2d = reducer.fit_transform(embeddings)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Plot
    plt.figure(figsize=(10, 8))
    
    if labels is not None:
        scatter = plt.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=labels,
            cmap='tab10',
            alpha=0.6,
            s=50
        )
        plt.colorbar(scatter, label='Label')
    else:
        plt.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            alpha=0.6,
            s=50
        )
    
    plt.xlabel(f'{method.upper()} 1', fontsize=12)
    plt.ylabel(f'{method.upper()} 2', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Predictions vs Targets"
):
    """
    Plot predictions vs targets for regression.
    
    Parameters
    ----------
    predictions : np.ndarray
        Model predictions
    targets : np.ndarray
        True targets
    save_path : str, optional
        Path to save figure
    title : str
        Plot title
    """
    plt.figure(figsize=(10, 8))
    
    # Scatter plot
    plt.scatter(targets, predictions, alpha=0.5, s=30)
    
    # Diagonal line (perfect prediction)
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    # Calculate metrics
    from scipy.stats import pearsonr, spearmanr
    pearson_r, _ = pearsonr(predictions, targets)
    spearman_r, _ = spearmanr(predictions, targets)
    
    # Add metrics text
    plt.text(
        0.05, 0.95,
        f'Pearson R: {pearson_r:.3f}\nSpearman R: {spearman_r:.3f}',
        transform=plt.gca().transAxes,
        fontsize=11,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    plt.xlabel('True Values', fontsize=12)
    plt.ylabel('Predictions', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_attention_weights(
    attention_weights: np.ndarray,
    node_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    title: str = "Attention Weights Heatmap"
):
    """
    Plot attention weights as a heatmap.
    
    Parameters
    ----------
    attention_weights : np.ndarray
        Attention weight matrix [n_nodes, n_nodes]
    node_names : list, optional
        Names of nodes
    save_path : str, optional
        Path to save figure
    title : str
        Plot title
    """
    plt.figure(figsize=(12, 10))
    
    sns.heatmap(
        attention_weights,
        xticklabels=node_names if node_names else False,
        yticklabels=node_names if node_names else False,
        cmap='viridis',
        cbar_kws={'label': 'Attention Weight'}
    )
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Target Nodes', fontsize=12)
    plt.ylabel('Source Nodes', fontsize=12)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_drug_response_distribution(
    drug_responses: pd.DataFrame,
    save_path: Optional[str] = None
):
    """
    Plot distribution of drug responses.
    
    Parameters
    ----------
    drug_responses : pd.DataFrame
        Drug response data [cell_lines x drugs]
    save_path : str, optional
        Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Overall distribution
    axes[0].hist(drug_responses.values.flatten(), bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Drug Response', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Overall Drug Response Distribution', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Per-drug variance
    drug_variances = drug_responses.var().sort_values(ascending=False)
    axes[1].bar(range(len(drug_variances)), drug_variances.values)
    axes[1].set_xlabel('Drug Index (sorted by variance)', fontsize=12)
    axes[1].set_ylabel('Variance', fontsize=12)
    axes[1].set_title('Per-Drug Response Variance', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
