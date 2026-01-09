"""
Loss Functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def weighted_mse_loss_fixed(pred, target, weight=None):
    """Fixed version for regression with optional sample weights"""
    if weight is None:
        return F.mse_loss(pred, target)
    else:
        # For regression, apply uniform weighting or per-sample weighting
        return (weight * (pred - target.to(pred.dtype)).pow(2)).mean()

def regular_mse_loss(pred, target):
    """Simple MSE loss for regression"""
    return F.mse_loss(pred, target.to(pred.dtype))

def robust_loss(pred, target, beta=1.0):
    """Robust Huber loss for regression (less sensitive to outliers)"""
    diff = pred - target.to(pred.dtype)
    abs_diff = torch.abs(diff)
    quadratic = torch.minimum(abs_diff, torch.tensor(beta))
    linear = abs_diff - quadratic
    return (0.5 * quadratic.pow(2) + beta * linear).mean()

''' extra loss functions to fix in V2 
these functions are suitable for contrastive learning, 
temperature calibration, and more sophisticated weighted loss calculations
'''

# def get_loss_function(name: str, **kwargs):
#     """
#     Get loss function by name.
    
#     Parameters
#     ----------
#     name : str
#         Loss function name
#     **kwargs
#         Additional arguments for loss function
        
#     Returns
#     -------
#     callable
#         Loss function
#     """
#     loss_functions = {
#         'mse': nn.MSELoss(**kwargs),
#         'mae': nn.L1Loss(**kwargs),
#         'cross_entropy': nn.CrossEntropyLoss(**kwargs),
#         'bce': nn.BCEWithLogitsLoss(**kwargs),
#         'huber': nn.HuberLoss(**kwargs),
#         'contrastive': NTXentLoss(**kwargs),
#         'triplet': nn.TripletMarginLoss(**kwargs),
#     }
    
#     if name not in loss_functions:
#         raise ValueError(f"Unknown loss function: {name}")
    
#     return loss_functions[name]


# class NTXentLoss(nn.Module):
#     """Normalized Temperature-scaled Cross Entropy Loss (NT-Xent) for contrastive learning."""
    
#     def __init__(self, temperature: float = 0.07):
#         """
#         Initialize NT-Xent loss.
        
#         Parameters
#         ----------
#         temperature : float
#             Temperature parameter
#         """
#         super(NTXentLoss, self).__init__()
#         self.temperature = temperature
    
#     def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
#         """
#         Compute NT-Xent loss.
        
#         Parameters
#         ----------
#         z_i : torch.Tensor
#             Embeddings from first view [batch_size, embedding_dim]
#         z_j : torch.Tensor
#             Embeddings from second view [batch_size, embedding_dim]
            
#         Returns
#         -------
#         torch.Tensor
#             Loss value
#         """
#         batch_size = z_i.size(0)
        
#         # Normalize embeddings
#         z_i = F.normalize(z_i, dim=1)
#         z_j = F.normalize(z_j, dim=1)
        
#         # Concatenate
#         z = torch.cat([z_i, z_j], dim=0)
        
#         # Compute similarity matrix
#         sim_matrix = torch.mm(z, z.t()) / self.temperature
        
#         # Create positive pairs mask
#         mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        
#         # Remove diagonal (self-similarity)
#         sim_matrix = sim_matrix.masked_fill(mask, -float('inf'))
        
#         # Labels: for i-th sample, the positive is at position i+batch_size (and vice versa)
#         labels = torch.cat([
#             torch.arange(batch_size, 2 * batch_size),
#             torch.arange(batch_size)
#         ], dim=0).to(z.device)
        
#         # Compute cross-entropy loss
#         loss = F.cross_entropy(sim_matrix, labels)
        
#         return loss


class WeightedMSELoss(nn.Module):
    """Weighted MSE loss for handling imbalanced regression."""
    
    def __init__(self, weight: torch.Tensor = None):
        """
        Initialize weighted MSE loss.
        
        Parameters
        ----------
        weight : torch.Tensor, optional
            Sample weights
        """
        super(WeightedMSELoss, self).__init__()
        self.weight = weight
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute weighted MSE loss."""
        squared_error = (predictions - targets) ** 2
        
        if self.weight is not None:
            squared_error = squared_error * self.weight
        
        return squared_error.mean()


# class FocalLoss(nn.Module):
#     """Focal loss for handling class imbalance in classification."""
    
#     def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
#         """
#         Initialize focal loss.
        
#         Parameters
#         ----------
#         alpha : float
#             Weighting factor
#         gamma : float
#             Focusing parameter
#         """
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
    
#     def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
#         """Compute focal loss."""
#         ce_loss = F.cross_entropy(predictions, targets, reduction='none')
#         p_t = torch.exp(-ce_loss)
#         focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        
#         return focal_loss.mean()


# class RankingLoss(nn.Module):
#     """Ranking loss for drug response prediction."""
    
#     def __init__(self, margin: float = 1.0):
#         """
#         Initialize ranking loss.
        
#         Parameters
#         ----------
#         margin : float
#             Margin for ranking
#         """
#         super(RankingLoss, self).__init__()
#         self.margin = margin
    
#     def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
#         """Compute pairwise ranking loss."""
#         # Create pairs where target_i > target_j
#         n = predictions.size(0)
        
#         # Expand dimensions for pairwise comparison
#         pred_i = predictions.unsqueeze(1).expand(n, n)
#         pred_j = predictions.unsqueeze(0).expand(n, n)
#         target_i = targets.unsqueeze(1).expand(n, n)
#         target_j = targets.unsqueeze(0).expand(n, n)
        
#         # Mask for valid pairs (where target_i > target_j)
#         mask = (target_i > target_j).float()
        
#         # Ranking loss: max(0, margin - (pred_i - pred_j))
#         loss = torch.clamp(self.margin - (pred_i - pred_j), min=0)
#         loss = (loss * mask).sum() / (mask.sum() + 1e-8)
        
#         return loss


# class MultiTaskLoss(nn.Module):
#     """Multi-task loss with learnable task weights."""
    
#     def __init__(self, num_tasks: int, loss_fns: list):
#         """
#         Initialize multi-task loss.
        
#         Parameters
#         ----------
#         num_tasks : int
#             Number of tasks
#         loss_fns : list
#             List of loss functions for each task
#         """
#         super(MultiTaskLoss, self).__init__()
#         self.num_tasks = num_tasks
#         self.loss_fns = nn.ModuleList(loss_fns)
        
#         # Learnable task weights (uncertainty weighting)
#         self.log_vars = nn.Parameter(torch.zeros(num_tasks))
    
#     def forward(self, predictions: list, targets: list) -> torch.Tensor:
#         """
#         Compute multi-task loss.
        
#         Parameters
#         ----------
#         predictions : list
#             Predictions for each task
#         targets : list
#             Targets for each task
            
#         Returns
#         -------
#         torch.Tensor
#             Combined loss
#         """
#         total_loss = 0
        
#         for i in range(self.num_tasks):
#             task_loss = self.loss_fns[i](predictions[i], targets[i])
#             precision = torch.exp(-self.log_vars[i])
#             total_loss += precision * task_loss + self.log_vars[i]
        
#         return total_loss
