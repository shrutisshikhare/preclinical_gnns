"""
Heterogeneous Graph Transformer
================================

Implementation of Graph Transformer architecture for heterogeneous graphs - 

these models work with HeteroData objects containing multiple node and edge types - for cell line, gene, drug graph systems

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, TransformerConv, Linear
from typing import Dict, Optional

class HGTModel(torch.nn.Module):
    """simple HGT-style model using linear transformations"""
    def __init__(self, input_dims, output_dim=64):
        super().__init__()
        self.encoders = torch.nn.ModuleDict()
        for node_type, input_dim in input_dims.items():
            self.encoders[node_type] = torch.nn.Sequential(
                torch.nn.Linear(input_dim, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, output_dim)
            )
    
    def forward(self, x_dict):
        embeddings = {}
        for node_type, x in x_dict.items():
            embeddings[node_type] = self.encoders[node_type](x)
        return embeddings


class HeteroGraphTransformer(nn.Module):
    """Heterogeneous Graph Transformer model"""
    
    def __init__(
        self,
        metadata: tuple,  # (node_types, edge_types)
        in_channels_dict: Dict[str, int],
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_edge_attr: bool = True
    ):
        """
        Initialise Heterogeneous Graph Transformer

        parameters
        -----
        metadata : tuple
            (node_types, edge_types) from HeteroData
        in_channels_dict : dict
            input feature dimension for each node type
        hidden_channels : int
            hidden dimension
        out_channels : int
            output embedding dimension
        num_layers : int
            no of transformer layers
        num_heads : int
            no of attention heads
        dropout : float
            dropout probability
        use_edge_attr : bool
            whether to use edge attributes
        """
        super(HeteroGraphTransformer, self).__init__()
        
        self.node_types, self.edge_types = metadata
        self.num_layers = num_layers
        self.dropout = dropout
        self.hidden_channels = hidden_channels
        
        # Input projection for each node type
        self.input_proj_dict = nn.ModuleDict()
        for node_type in self.node_types:
            self.input_proj_dict[node_type] = Linear(
                in_channels_dict[node_type],
                hidden_channels
            )
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList()
        for i in range(num_layers):
            conv = HeteroConv({
                edge_type: TransformerConv(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels // num_heads,
                    heads=num_heads,
                    dropout=dropout,
                    edge_dim=1 if use_edge_attr else None,
                    beta=True
                )
                for edge_type in self.edge_types
            }, aggr='sum')
            self.transformer_layers.append(conv)
        
        # Layer normalization for each node type
        self.layer_norms = nn.ModuleList()
        for i in range(num_layers):
            norm_dict = nn.ModuleDict()
            for node_type in self.node_types:
                norm_dict[node_type] = nn.LayerNorm(hidden_channels)
            self.layer_norms.append(norm_dict)
        
        # Feed-forward networks for each node type
        self.ffns = nn.ModuleList()
        for i in range(num_layers):
            ffn_dict = nn.ModuleDict()
            for node_type in self.node_types:
                ffn_dict[node_type] = nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_channels * 4, hidden_channels),
                    nn.Dropout(dropout)
                )
            self.ffns.append(ffn_dict)
        
        # Output projection for each node type
        self.output_proj_dict = nn.ModuleDict()
        for node_type in self.node_types:
            self.output_proj_dict[node_type] = Linear(
                hidden_channels,
                out_channels
            )
    
    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        """
        Forward pass
        
        parameters
        -----
        x_dict : dict
            node features for each node type
        edge_index_dict : dict
            edge indices for each edge type
        edge_attr_dict : dict, optional
            edge attributes for each edge type
            
        returns
        -----
        dict
            node embeddings for each node type
        """
        # Input projection
        x_dict = {
            node_type: self.input_proj_dict[node_type](x)
            for node_type, x in x_dict.items()
        }
        
        # Transformer layers with residual connections
        for i in range(self.num_layers):
            # Store residual
            residual_dict = {node_type: x.clone() for node_type, x in x_dict.items()}
            
            # Transformer convolution
            if edge_attr_dict is not None:
                # Pass edge attributes for each edge type
                x_dict_new = {}
                for edge_type in self.edge_types:
                    src, rel, dst = edge_type
                    edge_index = edge_index_dict[edge_type]
                    edge_attr = edge_attr_dict.get(edge_type, None)
                    
                    # Apply transformer for this edge type
                    conv = self.transformer_layers[i].convs[edge_type]
                    if src not in x_dict_new:
                        x_dict_new[src] = []
                    if dst not in x_dict_new:
                        x_dict_new[dst] = []
                    
                    out = conv(
                        (x_dict[src], x_dict[dst]) if src != dst else x_dict[src],
                        edge_index,
                        edge_attr
                    )
                    
                    if src == dst:
                        x_dict_new[src].append(out)
                    else:
                        # For bipartite edges, output goes to destination
                        x_dict_new[dst].append(out)
                
                # Aggregate messages for each node type
                x_dict = {
                    node_type: torch.stack(tensors).sum(dim=0) if tensors else x_dict[node_type]
                    for node_type, tensors in x_dict_new.items()
                }
            else:
                x_dict = self.transformer_layers[i](x_dict, edge_index_dict)
            
            # Add residual connection and apply layer norm
            x_dict = {
                node_type: self.layer_norms[i][node_type](x + residual_dict[node_type])
                for node_type, x in x_dict.items()
            }
            
            # Feed-forward network with residual
            residual_dict = {node_type: x.clone() for node_type, x in x_dict.items()}
            x_dict = {
                node_type: self.ffns[i][node_type](x) + residual_dict[node_type]
                for node_type, x in x_dict.items()
            }
        
        # Output projection
        x_dict = {
            node_type: self.output_proj_dict[node_type](x)
            for node_type, x in x_dict.items()
        }
        
        return x_dict
    
    def get_embeddings(self, x_dict, edge_index_dict, edge_attr_dict=None):
        """Get node embeddings."""
        return self.forward(x_dict, edge_index_dict, edge_attr_dict)
