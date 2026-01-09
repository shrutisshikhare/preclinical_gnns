"""
Graph Transformer
=================

Implementation of Graph Transformer architecture for DepMap data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv
from typing import Optional, List
import math


class GraphTransformer(nn.Module):
    """Graph Transformer model."""
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_edge_attr: bool = False
    ):
        """
        Initialize Graph Transformer.
        
        Parameters
        ----------
        in_channels : int
            Input feature dimension
        hidden_channels : int
            Hidden dimension
        out_channels : int
            Output embedding dimension
        num_layers : int
            Number of transformer layers
        num_heads : int
            Number of attention heads
        dropout : float
            Dropout probability
        use_edge_attr : bool
            Whether to use edge attributes
        """
        super(GraphTransformer, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerConv(
                in_channels=hidden_channels,
                out_channels=hidden_channels // num_heads,
                heads=num_heads,
                dropout=dropout,
                edge_dim=1 if use_edge_attr else None,
                beta=True
            )
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_channels)
            for _ in range(num_layers)
        ])
        
        # Feed-forward networks
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels * 4, hidden_channels),
                nn.Dropout(dropout)
            )
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_channels, out_channels)
        
    def forward(self, x, edge_index, edge_attr=None):
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Node features [num_nodes, in_channels]
        edge_index : torch.Tensor
            Edge indices [2, num_edges]
        edge_attr : torch.Tensor, optional
            Edge attributes [num_edges, edge_dim]
            
        Returns
        -------
        torch.Tensor
            Node embeddings [num_nodes, out_channels]
        """
        # Input projection
        x = self.input_proj(x)
        x = F.gelu(x)
        
        # Apply transformer layers
        for i in range(self.num_layers):
            # Self-attention with residual connection
            residual = x
            x = self.transformer_layers[i](x, edge_index, edge_attr)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.layer_norms[i](x + residual)
            
            # Feed-forward with residual connection
            residual = x
            x = self.ffns[i](x)
            x = x + residual
        
        # Output projection
        x = self.output_proj(x)
        
        return x
    
    def get_embeddings(self, x, edge_index, edge_attr=None):
        """Get node embeddings."""
        return self.forward(x, edge_index, edge_attr)


class PositionalEncoding(nn.Module):
    """Positional encoding for graph nodes."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Initialize positional encoding.
        
        Parameters
        ----------
        d_model : int
            Dimension of embeddings
        max_len : int
            Maximum sequence length
        """
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """Add positional encoding to input."""
        return x + self.pe[:, :x.size(1), :]


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        """
        Initialize multi-head attention.
        
        Parameters
        ----------
        d_model : int
            Model dimension
        num_heads : int
            Number of attention heads
        dropout : float
            Dropout probability
        """
        super(MultiHeadAttention, self).__init__()
        
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask=None):
        """
        Forward pass.
        
        Parameters
        ----------
        query : torch.Tensor
            Query tensor
        key : torch.Tensor
            Key tensor
        value : torch.Tensor
            Value tensor
        mask : torch.Tensor, optional
            Attention mask
            
        Returns
        -------
        torch.Tensor
            Attention output
        """
        batch_size = query.size(0)
        
        # Linear projections and reshape for multi-head
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        context = torch.matmul(attn, V)
        
        # Reshape and apply output projection
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(context)
        
        return output
