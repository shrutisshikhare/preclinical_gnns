"""
GNN Models
==========

Implementation of various Graph Neural Network architectures:
- GCN (Graph Convolutional Network)
- GAT (Graph Attention Network)
- GraphSAGE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool, global_max_pool
from typing import Optional, List


class GCNEncoder(nn.Module):
    """Graph Convolutional Network encoder."""
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int],
        out_channels: int,
        dropout: float = 0.5,
        batch_norm: bool = True
    ):
        """
        Initialize GCN encoder.
        
        Parameters
        ----------
        in_channels : int
            Input feature dimension
        hidden_channels : list of int
            Hidden layer dimensions
        out_channels : int
            Output embedding dimension
        dropout : float
            Dropout probability
        batch_norm : bool
            Whether to use batch normalization
        """
        super(GCNEncoder, self).__init__()
        
        self.dropout = dropout
        self.batch_norm = batch_norm
        
        # Build layers
        layers = []
        bns = []
        
        prev_channels = in_channels
        for hidden_dim in hidden_channels:
            layers.append(GCNConv(prev_channels, hidden_dim))
            if batch_norm:
                bns.append(nn.BatchNorm1d(hidden_dim))
            prev_channels = hidden_dim
        
        layers.append(GCNConv(prev_channels, out_channels))
        
        self.convs = nn.ModuleList(layers)
        if batch_norm:
            self.bns = nn.ModuleList(bns)
    
    def forward(self, x, edge_index, edge_weight=None):
        """Forward pass."""
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_weight)
            if self.batch_norm:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final layer without activation
        x = self.convs[-1](x, edge_index, edge_weight)
        
        return x
    
    def get_embeddings(self, x, edge_index, edge_weight=None):
        """Get node embeddings."""
        return self.forward(x, edge_index, edge_weight)


class GATEncoder(nn.Module):
    """Graph Attention Network encoder."""
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int],
        out_channels: int,
        heads: int = 4,
        dropout: float = 0.5,
        concat_heads: bool = True
    ):
        """
        Initialize GAT encoder.
        
        Parameters
        ----------
        in_channels : int
            Input feature dimension
        hidden_channels : list of int
            Hidden layer dimensions
        out_channels : int
            Output embedding dimension
        heads : int
            Number of attention heads
        dropout : float
            Dropout probability
        concat_heads : bool
            Whether to concatenate or average attention heads
        """
        super(GATEncoder, self).__init__()
        
        self.dropout = dropout
        self.heads = heads
        
        # Build layers
        layers = []
        
        prev_channels = in_channels
        for hidden_dim in hidden_channels:
            layers.append(GATConv(
                prev_channels, 
                hidden_dim, 
                heads=heads, 
                dropout=dropout,
                concat=concat_heads
            ))
            prev_channels = hidden_dim * heads if concat_heads else hidden_dim
        
        # Final layer (usually average heads)
        layers.append(GATConv(
            prev_channels,
            out_channels,
            heads=1,
            concat=False,
            dropout=dropout
        ))
        
        self.convs = nn.ModuleList(layers)
    
    def forward(self, x, edge_index):
        """Forward pass."""
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final layer
        x = self.convs[-1](x, edge_index)
        
        return x
    
    def get_embeddings(self, x, edge_index):
        """Get node embeddings."""
        return self.forward(x, edge_index)


class SAGEEncoder(nn.Module):
    """GraphSAGE encoder."""
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int],
        out_channels: int,
        dropout: float = 0.5,
        aggregator: str = "mean"
    ):
        """
        Initialize GraphSAGE encoder.
        
        Parameters
        ----------
        in_channels : int
            Input feature dimension
        hidden_channels : list of int
            Hidden layer dimensions
        out_channels : int
            Output embedding dimension
        dropout : float
            Dropout probability
        aggregator : str
            Aggregation method: 'mean', 'max', 'lstm'
        """
        super(SAGEEncoder, self).__init__()
        
        self.dropout = dropout
        
        # Build layers
        layers = []
        
        prev_channels = in_channels
        for hidden_dim in hidden_channels:
            layers.append(SAGEConv(prev_channels, hidden_dim, aggr=aggregator))
            prev_channels = hidden_dim
        
        layers.append(SAGEConv(prev_channels, out_channels, aggr=aggregator))
        
        self.convs = nn.ModuleList(layers)
    
    def forward(self, x, edge_index):
        """Forward pass."""
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final layer
        x = self.convs[-1](x, edge_index)
        
        return x
    
    def get_embeddings(self, x, edge_index):
        """Get node embeddings."""
        return self.forward(x, edge_index)


class GNNPooling(nn.Module):
    """Graph-level pooling for obtaining graph-level representations."""
    
    def __init__(self, pooling: str = "mean"):
        """
        Initialize pooling layer.
        
        Parameters
        ----------
        pooling : str
            Pooling method: 'mean', 'max', 'sum'
        """
        super(GNNPooling, self).__init__()
        self.pooling = pooling
    
    def forward(self, x, batch):
        """Apply pooling."""
        if self.pooling == "mean":
            return global_mean_pool(x, batch)
        elif self.pooling == "max":
            return global_max_pool(x, batch)
        elif self.pooling == "sum":
            return global_mean_pool(x, batch) * x.size(0)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")
