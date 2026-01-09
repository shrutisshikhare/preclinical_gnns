"""
Predictor Models
================

Models for downstream tasks:
- Drug response prediction
- Embedding extraction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class DrugResponsePredictor(nn.Module):
    """Model for predicting drug response from cell line embeddings."""
    
    def __init__(
        self,
        encoder: nn.Module,
        embedding_dim: int,
        hidden_dims: List[int] = [256, 128],
        num_drugs: Optional[int] = None,
        drug_embedding_dim: int = 64,
        dropout: float = 0.3,
        task: str = "regression"
    ):
        """
        Initialize drug response predictor.
        
        Parameters
        ----------
        encoder : nn.Module
            Pre-trained graph encoder (GNN or Transformer)
        embedding_dim : int
            Dimension of cell line embeddings
        hidden_dims : list of int
            Hidden layer dimensions for prediction head
        num_drugs : int, optional
            Number of drugs (for drug embedding)
        drug_embedding_dim : int
            Dimension of drug embeddings
        dropout : float
            Dropout probability
        task : str
            Task type: 'regression' or 'classification'
        """
        super(DrugResponsePredictor, self).__init__()
        
        self.encoder = encoder
        self.task = task
        self.use_drug_embedding = num_drugs is not None
        
        # Drug embedding
        if self.use_drug_embedding:
            self.drug_embedding = nn.Embedding(num_drugs, drug_embedding_dim)
            input_dim = embedding_dim + drug_embedding_dim
        else:
            input_dim = embedding_dim
        
        # Prediction head
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        output_dim = 1 if task == "regression" else 2
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.predictor = nn.Sequential(*layers)
    
    def forward(self, x, edge_index, drug_ids=None, edge_attr=None):
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Node features
        edge_index : torch.Tensor
            Edge indices
        drug_ids : torch.Tensor, optional
            Drug identifiers for embedding lookup
        edge_attr : torch.Tensor, optional
            Edge attributes
            
        Returns
        -------
        torch.Tensor
            Predictions
        """
        # Get embeddings from encoder
        embeddings = self.encoder(x, edge_index, edge_attr)
        
        # Combine with drug embeddings if available
        if self.use_drug_embedding and drug_ids is not None:
            drug_emb = self.drug_embedding(drug_ids)
            embeddings = torch.cat([embeddings, drug_emb], dim=-1)
        
        # Predict
        predictions = self.predictor(embeddings)
        
        if self.task == "regression":
            return predictions.squeeze(-1)
        else:
            return predictions
    
    def predict_drug_response(self, x, edge_index, drug_ids=None, edge_attr=None):
        """Predict drug response."""
        self.eval()
        with torch.no_grad():
            predictions = self.forward(x, edge_index, drug_ids, edge_attr)
        return predictions


class EmbeddingModel(nn.Module):
    """Model for extracting embeddings from graphs."""
    
    def __init__(
        self,
        encoder: nn.Module,
        projection_dim: Optional[int] = None,
        use_contrastive: bool = False,
        temperature: float = 0.07
    ):
        """
        Initialize embedding model.
        
        Parameters
        ----------
        encoder : nn.Module
            Graph encoder (GNN or Transformer)
        projection_dim : int, optional
            Dimension for projection head (for contrastive learning)
        use_contrastive : bool
            Whether to use contrastive learning
        temperature : float
            Temperature for contrastive loss
        """
        super(EmbeddingModel, self).__init__()
        
        self.encoder = encoder
        self.use_contrastive = use_contrastive
        self.temperature = temperature
        
        # Projection head for contrastive learning
        if use_contrastive and projection_dim is not None:
            # Get encoder output dimension
            encoder_out_dim = self._get_encoder_out_dim()
            
            self.projection_head = nn.Sequential(
                nn.Linear(encoder_out_dim, encoder_out_dim),
                nn.ReLU(),
                nn.Linear(encoder_out_dim, projection_dim)
            )
        else:
            self.projection_head = None
    
    def _get_encoder_out_dim(self):
        """Get output dimension of encoder."""
        # This is a simplified version - you may need to adjust based on your encoder
        if hasattr(self.encoder, 'out_channels'):
            return self.encoder.out_channels
        elif hasattr(self.encoder, 'output_proj'):
            return self.encoder.output_proj.out_features
        else:
            raise ValueError("Cannot determine encoder output dimension")
    
    def forward(self, x, edge_index, edge_attr=None):
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Node features
        edge_index : torch.Tensor
            Edge indices
        edge_attr : torch.Tensor, optional
            Edge attributes
            
        Returns
        -------
        dict
            Dictionary with 'embeddings' and optionally 'projections'
        """
        # Get embeddings from encoder
        embeddings = self.encoder(x, edge_index, edge_attr)
        
        output = {'embeddings': embeddings}
        
        # Apply projection head if using contrastive learning
        if self.use_contrastive and self.projection_head is not None:
            projections = self.projection_head(embeddings)
            projections = F.normalize(projections, dim=-1)
            output['projections'] = projections
        
        return output
    
    def get_embeddings(self, x, edge_index, edge_attr=None):
        """Extract embeddings."""
        self.eval()
        with torch.no_grad():
            output = self.forward(x, edge_index, edge_attr)
        return output['embeddings']
    
    def contrastive_loss(self, z1, z2):
        """
        Compute NT-Xent (normalized temperature-scaled cross entropy) loss.
        
        Parameters
        ----------
        z1 : torch.Tensor
            Projections from first augmentation
        z2 : torch.Tensor
            Projections from second augmentation
            
        Returns
        -------
        torch.Tensor
            Contrastive loss
        """
        batch_size = z1.size(0)
        
        # Concatenate projections
        z = torch.cat([z1, z2], dim=0)
        
        # Compute similarity matrix
        sim_matrix = torch.mm(z, z.t()) / self.temperature
        
        # Create labels (positive pairs are (i, i+batch_size) and (i+batch_size, i))
        labels = torch.cat([torch.arange(batch_size, 2 * batch_size),
                           torch.arange(batch_size)], dim=0)
        labels = labels.to(z.device)
        
        # Mask out self-similarity
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        sim_matrix = sim_matrix.masked_fill(mask, -float('inf'))
        
        # Compute loss
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss


class MultiTaskPredictor(nn.Module):
    """Multi-task learning model for multiple prediction tasks."""
    
    def __init__(
        self,
        encoder: nn.Module,
        embedding_dim: int,
        task_configs: List[dict],
        shared_layers: List[int] = [256, 128],
        dropout: float = 0.3
    ):
        """
        Initialize multi-task predictor.
        
        Parameters
        ----------
        encoder : nn.Module
            Shared encoder
        embedding_dim : int
            Embedding dimension
        task_configs : list of dict
            Configuration for each task with keys 'name', 'type', 'output_dim'
        shared_layers : list of int
            Shared hidden layer dimensions
        dropout : float
            Dropout probability
        """
        super(MultiTaskPredictor, self).__init__()
        
        self.encoder = encoder
        self.task_configs = task_configs
        
        # Shared layers
        layers = []
        prev_dim = embedding_dim
        for hidden_dim in shared_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.shared_net = nn.Sequential(*layers)
        
        # Task-specific heads
        self.task_heads = nn.ModuleDict()
        for task_config in task_configs:
            task_name = task_config['name']
            output_dim = task_config['output_dim']
            
            self.task_heads[task_name] = nn.Linear(prev_dim, output_dim)
    
    def forward(self, x, edge_index, edge_attr=None):
        """Forward pass for all tasks."""
        # Get embeddings
        embeddings = self.encoder(x, edge_index, edge_attr)
        
        # Shared representation
        shared_repr = self.shared_net(embeddings)
        
        # Task-specific predictions
        outputs = {}
        for task_config in self.task_configs:
            task_name = task_config['name']
            outputs[task_name] = self.task_heads[task_name](shared_repr)
        
        return outputs
