"""
Model utilities.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count the number of parameters in a model.
    
    Parameters
    ----------
    model : nn.Module
        PyTorch model
        
    Returns
    -------
    dict
        Dictionary with total, trainable, and non-trainable parameters
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': non_trainable_params
    }


def save_embeddings(
    embeddings: np.ndarray,
    identifiers: list,
    save_path: str,
    metadata: Optional[Dict] = None
):
    """
    Save embeddings to file.
    
    Parameters
    ----------
    embeddings : np.ndarray
        Embeddings array [n_samples, embedding_dim]
    identifiers : list
        Sample identifiers (e.g., cell line names)
    save_path : str
        Path to save embeddings
    metadata : dict, optional
        Additional metadata to save
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as DataFrame for readability
    if save_path.suffix == '.csv':
        df = pd.DataFrame(
            embeddings,
            index=identifiers,
            columns=[f'dim_{i}' for i in range(embeddings.shape[1])]
        )
        df.to_csv(save_path)
        logger.info(f"Embeddings saved to {save_path}")
    
    # Save as numpy for efficiency
    elif save_path.suffix == '.npz':
        save_dict = {
            'embeddings': embeddings,
            'identifiers': np.array(identifiers)
        }
        if metadata:
            save_dict.update(metadata)
        np.savez(save_path, **save_dict)
        logger.info(f"Embeddings saved to {save_path}")
    
    else:
        raise ValueError(f"Unsupported file format: {save_path.suffix}")


def load_embeddings(load_path: str) -> tuple:
    """
    Load embeddings from file.
    
    Parameters
    ----------
    load_path : str
        Path to embeddings file
        
    Returns
    -------
    tuple
        (embeddings, identifiers, metadata)
    """
    load_path = Path(load_path)
    
    if load_path.suffix == '.csv':
        df = pd.read_csv(load_path, index_col=0)
        embeddings = df.values
        identifiers = df.index.tolist()
        metadata = None
        logger.info(f"Loaded embeddings from {load_path}")
    
    elif load_path.suffix == '.npz':
        data = np.load(load_path)
        embeddings = data['embeddings']
        identifiers = data['identifiers'].tolist()
        
        # Extract metadata
        metadata = {k: data[k] for k in data.files if k not in ['embeddings', 'identifiers']}
        logger.info(f"Loaded embeddings from {load_path}")
    
    else:
        raise ValueError(f"Unsupported file format: {load_path.suffix}")
    
    return embeddings, identifiers, metadata


def freeze_layers(model: nn.Module, layer_names: list):
    """
    Freeze specific layers in a model.
    
    Parameters
    ----------
    model : nn.Module
        PyTorch model
    layer_names : list
        Names of layers to freeze
    """
    for name, param in model.named_parameters():
        if any(layer_name in name for layer_name in layer_names):
            param.requires_grad = False
            logger.info(f"Froze layer: {name}")


def unfreeze_layers(model: nn.Module, layer_names: list):
    """
    Unfreeze specific layers in a model.
    
    Parameters
    ----------
    model : nn.Module
        PyTorch model
    layer_names : list
        Names of layers to unfreeze
    """
    for name, param in model.named_parameters():
        if any(layer_name in name for layer_name in layer_names):
            param.requires_grad = True
            logger.info(f"Unfroze layer: {name}")


def initialize_weights(model: nn.Module):
    """
    Initialize model weights using Xavier/Kaiming initialization.
    
    Parameters
    ----------
    model : nn.Module
        PyTorch model
    """
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        
        elif isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        
        elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


def get_device(gpu_id: Optional[int] = None) -> torch.device:
    """
    Get the appropriate device for PyTorch.
    
    Parameters
    ----------
    gpu_id : int, optional
        Specific GPU ID to use
        
    Returns
    -------
    torch.device
        PyTorch device
    """
    if torch.cuda.is_available():
        if gpu_id is not None:
            device = torch.device(f'cuda:{gpu_id}')
        else:
            device = torch.device('cuda')
        logger.info(f"Using device: {device} ({torch.cuda.get_device_name(device)})")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info("Using device: MPS (Apple Silicon)")
    else:
        device = torch.device('cpu')
        logger.info("Using device: CPU")
    
    return device


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Parameters
    ----------
    seed : int
        Random seed
    """
    import random
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    logger.info(f"Random seed set to {seed}")
