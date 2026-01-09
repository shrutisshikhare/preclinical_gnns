
"""
Heterogeneous Graph Predictors
===============================

Task-specific models for heterogeneous graphs:
1. Drug response prediction (link prediction)
2. Embedding generation with contrastive learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union

class HeteroEmbeddingModel(nn.Module):
    """
    Heterogeneous graph embedding model with optional contrastive learning.
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        projection_dim: int = 64,
        node_types: list = None,
        use_contrastive: bool = True,
        temperature: float = 0.07
    ):
        super().__init__()
        self.encoder = encoder
        self.use_contrastive = use_contrastive
        self.temperature = temperature
        self.node_types = node_types or ['cell_line', 'gene', 'drug']
        
        # Projection heads for contrastive learning
        if use_contrastive:
            self.projection_heads = nn.ModuleDict()
            embedding_dim = encoder.out_channels
            
            for node_type in self.node_types:
                self.projection_heads[node_type] = nn.Sequential(
                    nn.Linear(embedding_dim, embedding_dim),
                    nn.ReLU(),
                    nn.Linear(embedding_dim, projection_dim)
                )
    
    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        edge_attr_dict: Optional[Dict] = None
    ) -> Union[Dict[str, torch.Tensor], Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]:
        """Forward pass - returns embeddings or (embeddings, projections)."""
        # Get embeddings from encoder
        embeddings_dict = self.encoder(x_dict, edge_index_dict, edge_attr_dict)
        
        if self.use_contrastive:
            # Apply projection heads
            projections_dict = {}
            for node_type in self.node_types:
                if node_type in embeddings_dict:
                    projections_dict[node_type] = self.projection_heads[node_type](
                        embeddings_dict[node_type]
                    )
            return embeddings_dict, projections_dict
        else:
            return embeddings_dict
    
    def contrastive_loss(
        self,
        proj_dict: Dict[str, torch.Tensor],
        target_node_type: str = 'cell_line'
    ) -> torch.Tensor:
        """Compute InfoNCE contrastive loss."""
        if target_node_type not in proj_dict:
            raise ValueError(f"Node type '{target_node_type}' not in projections")
        
        z = proj_dict[target_node_type]
        z = F.normalize(z, dim=1)
        
        similarity_matrix = torch.matmul(z, z.t()) / self.temperature
        labels = torch.arange(z.shape[0], device=z.device)
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss


class HeteroDrugResponsePredictor(nn.Module):
    """
    Heterogeneous graph model for drug response prediction.
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        embedding_dim: int = 128,
        hidden_dims: list = [256, 128, 64],
        dropout: float = 0.3,
        use_edge_features: bool = False,
        activation: str = 'relu'
    ):
        super().__init__()
        self.encoder = encoder
        self.embedding_dim = embedding_dim
        self.use_edge_features = use_edge_features
        
        # MLP for drug response prediction
        layers = []
        input_dim = embedding_dim * 2  # cell_line + drug embeddings
        if use_edge_features:
            input_dim += 1  # Add edge feature dimension
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, 1))
        self.predictor = nn.Sequential(*layers)
    
    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        target_edge_type: Tuple[str, str, str] = ('cell_line', 'treated_with', 'drug'),
        edge_attr_dict: Optional[Dict] = None
    ) -> torch.Tensor:
        """Forward pass - predict drug response."""
        # Get node embeddings
        embeddings_dict = self.encoder(x_dict, edge_index_dict, edge_attr_dict)
        
        # Extract edge indices for target edge type
        src_type, _, dst_type = target_edge_type
        edge_index = edge_index_dict[target_edge_type]
        
        # Get embeddings for source and destination nodes
        src_emb = embeddings_dict[src_type][edge_index[0]]
        dst_emb = embeddings_dict[dst_type][edge_index[1]]
        
        # Concatenate embeddings
        edge_emb = torch.cat([src_emb, dst_emb], dim=1)
        
        # Add edge features if requested
        if self.use_edge_features and edge_attr_dict is not None:
            if target_edge_type in edge_attr_dict:
                edge_features = edge_attr_dict[target_edge_type]
                if edge_features.dim() == 1:
                    edge_features = edge_features.unsqueeze(1)
                edge_emb = torch.cat([edge_emb, edge_features], dim=1)
        
        # Predict
        predictions = self.predictor(edge_emb).squeeze(-1)
        return predictions
    
    def predict_new_edges(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        new_edges: torch.Tensor,
        target_edge_type: Tuple[str, str, str] = ('cell_line', 'treated_with', 'drug'),
        edge_attr_dict: Optional[Dict] = None
    ) -> torch.Tensor:
        """Predict for new edges not in the graph."""
        # Get embeddings
        embeddings_dict = self.encoder(x_dict, edge_index_dict, edge_attr_dict)
        
        src_type, _, dst_type = target_edge_type
        
        # Get embeddings for new edges
        src_emb = embeddings_dict[src_type][new_edges[0]]
        dst_emb = embeddings_dict[dst_type][new_edges[1]]
        
        # Concatenate and predict
        edge_emb = torch.cat([src_emb, dst_emb], dim=1)
        predictions = self.predictor(edge_emb).squeeze(-1)
        
        return predictions
