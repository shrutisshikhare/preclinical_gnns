"""
Heterogeneous GNN Models
=========================

Implementation of heterogeneous Graph Neural Network architectures

these models work with HeteroData objects containing multiple node and edge types - for cell line, gene, drug graph systems
"""

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GCNConv, GATConv, SAGEConv, Linear, GATv2Conv
from typing import Dict, List, Optional

import torch_geometric
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.module_dict import ModuleDict
from torch_geometric.typing import Adj, EdgeType, NodeType
from torch_geometric.utils.hetero import check_add_self_loops

import warnings
from collections import defaultdict
from typing import Dict, List, Optional


def group(xs: List[Tensor], aggr: Optional[str]) -> Optional[Tensor]:
    
    if len(xs) == 0:
        return None
    elif aggr is None:
        return torch.stack(xs, dim=1)
    elif len(xs) == 1:
        return xs[0]
    elif aggr == "cat":
        return torch.cat(xs, dim=-1)
    else:
        out = torch.stack(xs, dim=0)
        out = getattr(torch, aggr)(out, dim=0)
        out = out[0] if isinstance(out, tuple) else out
        return out


class HeteroConv2(torch.nn.Module):
 
    def __init__(
        self,
        convs: Dict[EdgeType, MessagePassing],
        aggr: Optional[str] = "sum",
    ):
        super().__init__()

        for edge_type, module in convs.items():
            check_add_self_loops(module, [edge_type])

        src_node_types = set([key[0] for key in convs.keys()])
        dst_node_types = set([key[-1] for key in convs.keys()])
        if len(src_node_types - dst_node_types) > 0:
            warnings.warn(
                f"There exist node types ({src_node_types - dst_node_types}) "
                f"whose representations do not get updated during message "
                f"passing as they do not occur as destination type in any "
                f"edge type. This may lead to unexpected behavior.")

        self.convs = ModuleDict({'__'.join(k): v for k, v in convs.items()})
        self.aggr = aggr

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for conv in self.convs.values():
            conv.reset_parameters()

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[EdgeType, Adj],
        *args_dict,
        **kwargs_dict,
    ) -> Dict[NodeType, Tensor]:
        
        
        att_w_dict = {} # stores attention weights from any layers that use attention
        
        out_dict = defaultdict(list)
        for edge_type, edge_index in edge_index_dict.items():
            src, rel, dst = edge_type

            str_edge_type = '__'.join(edge_type)
            if str_edge_type not in self.convs:
                continue

            args = []
            for value_dict in args_dict:
                if edge_type in value_dict:
                    args.append(value_dict[edge_type])
                elif src == dst and src in value_dict:
                    args.append(value_dict[src])
                elif src in value_dict or dst in value_dict:
                    args.append(
                        (value_dict.get(src, None), value_dict.get(dst, None)))

            kwargs = {}
            for arg, value_dict in kwargs_dict.items():
                arg = arg[:-5]  # `{*}_dict`
                if edge_type in value_dict:
                    kwargs[arg] = value_dict[edge_type]
                elif src == dst and src in value_dict:
                    kwargs[arg] = value_dict[src]
                elif src in value_dict or dst in value_dict:
                    kwargs[arg] = (value_dict.get(src, None),
                                   value_dict.get(dst, None))

            conv = self.convs[str_edge_type]

            if src == dst:
                if isinstance(conv, torch_geometric.nn.GATv2Conv):
                    out, att_w = conv(x_dict[src], edge_index, return_attention_weights=True, *args, **kwargs)
                    att_w_dict[edge_type] = att_w
                else:
                    out = conv(x_dict[src], edge_index,  *args, **kwargs)
            else:
                if isinstance(conv, torch_geometric.nn.GATv2Conv):
                    out, att_w = conv((x_dict[src], x_dict[dst]), edge_index,
                                            return_attention_weights=True, *args, **kwargs)
                    att_w_dict[edge_type] = att_w
                else:
                    out = conv((x_dict[src], x_dict[dst]), edge_index, *args,
                               **kwargs)

            out_dict[dst].append(out)
        
        for key, value in out_dict.items():

            out_dict[key] = group(value, self.aggr)

        return out_dict, att_w_dict

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(num_relations={len(self.convs)})'

class HeteroGATWithAttention(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()

        for _ in range(num_layers):
            conv = HeteroConv2({
                
                ('cell_line', 'expresses', 'gene'): GATv2Conv((-1, -1), hidden_channels, edge_dim=1, dropout=0.4, add_self_loops=False),
                ('cell_line', 'has_mutation', 'gene'): GATv2Conv((-1, -1), hidden_channels, edge_dim=1, dropout=0.4, add_self_loops=False),
                ('cell_line', 'has_gene_effect', 'gene'): GATv2Conv((-1, -1), hidden_channels, edge_dim=1, dropout=0.4, add_self_loops=False),
                ('cell_line', 'treated_with', 'drug'): GATv2Conv((-1, -1), hidden_channels, edge_dim=1, dropout=0.2, add_self_loops=False),
                ('drug', 'targets', 'gene'): GATv2Conv((-1, -1), hidden_channels, edge_dim=1, dropout=0.2, add_self_loops=False),
                ('gene', 'interacts', 'gene'): GATv2Conv((-1, -1), hidden_channels, edge_dim=1, dropout=0.4, add_self_loops=False),
                ('gene', 'expressed_by', 'cell_line'): GATv2Conv((-1, -1), hidden_channels, edge_dim=1, dropout=0.4, add_self_loops=False),
                ('gene', 'mutated_in', 'cell_line'): GATv2Conv((-1, -1), hidden_channels, edge_dim=1, dropout=0.4, add_self_loops=False),
                ('gene', 'has_gene_effect_in', 'cell_line'): GATv2Conv((-1, -1), hidden_channels, edge_dim=1, dropout=0.4, add_self_loops=False),
                ('drug', 'treats', 'cell_line'): GATv2Conv((-1, -1), hidden_channels, edge_dim=1, dropout=0.2, add_self_loops=False),
                ('gene', 'targeted_by', 'drug'): GATv2Conv((-1, -1), hidden_channels, edge_dim=1, dropout=0.2, add_self_loops=False)
                
            }, aggr='sum')
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        for conv in self.convs:
            x_dict, att_w_dict = conv(x_dict, edge_index_dict, edge_attr_dict)    #x_dict = {key: x.relu() for key, x in x_dict.items()}
        return x_dict, att_w_dict

class DrugResponseEdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        z = torch.cat([z_dict['cell_line'][row], z_dict['drug'][col]], dim=-1)

        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)

class HeteroGNNDrugPredictor(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.z_dict = None
        self.att_w_dict = None
        self.encoder = HeteroGATWithAttention(hidden_channels, hidden_channels, num_layers=2)
        self.decoder = DrugResponseEdgeDecoder(hidden_channels)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict, edge_label_index):
        z_dict, att_w_dict = self.encoder(x_dict, edge_index_dict, edge_attr_dict)
        self.z_dict = z_dict
        self.att_w_dict = att_w_dict
        return self.decoder(z_dict, edge_label_index)

class HeteroGCNEncoder(nn.Module):
    """Heterogeneous Graph Convolutional Network encoder."""
    
    def __init__(
        self,
        metadata: tuple,  # (node_types, edge_types)
        in_channels_dict: Dict[str, int],
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        dropout: float = 0.5,
        aggr: str = 'sum'
    ):
        """
        Initialize Heterogeneous GCN encoder.
        
        Parameters
        ----------
        metadata : tuple
            (node_types, edge_types) from HeteroData
        in_channels_dict : dict
            Input feature dimension for each node type
        hidden_channels : int
            Hidden layer dimension
        out_channels : int
            Output embedding dimension
        num_layers : int
            Number of GCN layers
        dropout : float
            Dropout probability
        aggr : str
            Aggregation method: 'sum', 'mean', 'max'
        """
        super(HeteroGCNEncoder, self).__init__()
        
        self.node_types, self.edge_types = metadata
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input projection for each node type
        self.lin_dict = nn.ModuleDict()
        for node_type in self.node_types:
            self.lin_dict[node_type] = Linear(
                in_channels_dict[node_type], 
                hidden_channels
            )
        
        # Heterogeneous convolution layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            conv = HeteroConv({
                edge_type: GCNConv(
                    hidden_channels, 
                    hidden_channels if i < num_layers - 1 else out_channels
                )
                for edge_type in self.edge_types
            }, aggr=aggr)
            self.convs.append(conv)
        
        # Batch normalization for each node type
        self.norms = nn.ModuleList()
        for i in range(num_layers - 1):
            norm_dict = nn.ModuleDict()
            for node_type in self.node_types:
                norm_dict[node_type] = nn.BatchNorm1d(hidden_channels)
            self.norms.append(norm_dict)
    
    def forward(self, x_dict, edge_index_dict):
        """
        Forward pass.
        
        Parameters
        ----------
        x_dict : dict
            Node features for each node type
        edge_index_dict : dict
            Edge indices for each edge type
            
        Returns
        -------
        dict
            Node embeddings for each node type
        """
        # Input projection
        x_dict = {
            node_type: self.lin_dict[node_type](x)
            for node_type, x in x_dict.items()
        }
        
        # Graph convolutions
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            
            # Apply activation and normalization (except last layer)
            if i < self.num_layers - 1:
                x_dict = {
                    node_type: F.relu(self.norms[i][node_type](x))
                    for node_type, x in x_dict.items()
                }
                x_dict = {
                    node_type: F.dropout(x, p=self.dropout, training=self.training)
                    for node_type, x in x_dict.items()
                }
        
        return x_dict

class HeteroGATEncoder(nn.Module):
    """Heterogeneous Graph Attention Network encoder."""
    
    def __init__(
        self,
        metadata: tuple,
        in_channels_dict: Dict[str, int],
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.5,
        aggr: str = 'sum'
    ):
        """
        Initialize Heterogeneous GAT encoder.
        
        Parameters
        ----------
        metadata : tuple
            (node_types, edge_types) from HeteroData
        in_channels_dict : dict
            Input feature dimension for each node type
        hidden_channels : int
            Hidden layer dimension (per head)
        out_channels : int
            Output embedding dimension
        num_layers : int
            Number of GAT layers
        num_heads : int
            Number of attention heads
        dropout : float
            Dropout probability
        aggr : str
            Aggregation method
        """
        super(HeteroGATEncoder, self).__init__()
        
        self.node_types, self.edge_types = metadata
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_heads = num_heads
        
        # Input projection
        self.lin_dict = nn.ModuleDict()
        for node_type in self.node_types:
            self.lin_dict[node_type] = Linear(
                in_channels_dict[node_type],
                hidden_channels * num_heads
            )
        
        # Heterogeneous GAT layers
        self.convs = nn.ModuleList()
        for i in range(num_layers - 1):
            conv = HeteroConv({
                edge_type: GATConv(
                    hidden_channels * num_heads,
                    hidden_channels,
                    heads=num_heads,
                    dropout=dropout,
                    concat=True
                )
                for edge_type in self.edge_types
            }, aggr=aggr)
            self.convs.append(conv)
        
        # Final layer
        final_conv = HeteroConv({
            edge_type: GATConv(
                hidden_channels * num_heads,
                out_channels,
                heads=1,
                dropout=dropout,
                concat=False
            )
            for edge_type in self.edge_types
        }, aggr=aggr)
        self.convs.append(final_conv)
    
    def forward(self, x_dict, edge_index_dict):
        """Forward pass."""
        # Input projection
        x_dict = {
            node_type: F.relu(self.lin_dict[node_type](x))
            for node_type, x in x_dict.items()
        }
        
        # Graph convolutions
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            
            if i < self.num_layers - 1:
                x_dict = {
                    node_type: F.elu(x)
                    for node_type, x in x_dict.items()
                }
                x_dict = {
                    node_type: F.dropout(x, p=self.dropout, training=self.training)
                    for node_type, x in x_dict.items()
                }
        
        return x_dict

class HeteroSAGEEncoder(nn.Module):
    """Heterogeneous GraphSAGE encoder."""
    
    def __init__(
        self,
        metadata: tuple,
        in_channels_dict: Dict[str, int],
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        dropout: float = 0.5,
        aggr: str = 'mean'
    ):
        """
        Initialize Heterogeneous SAGE encoder.
        
        Parameters
        ----------
        metadata : tuple
            (node_types, edge_types) from HeteroData
        in_channels_dict : dict
            Input feature dimension for each node type
        hidden_channels : int
            Hidden layer dimension
        out_channels : int
            Output embedding dimension
        num_layers : int
            Number of SAGE layers
        dropout : float
            Dropout probability
        aggr : str
            Aggregation method
        """
        super(HeteroSAGEEncoder, self).__init__()
        
        self.node_types, self.edge_types = metadata
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input projection
        self.lin_dict = nn.ModuleDict()
        for node_type in self.node_types:
            self.lin_dict[node_type] = Linear(
                in_channels_dict[node_type],
                hidden_channels
            )
        
        # Heterogeneous SAGE layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            conv = HeteroConv({
                edge_type: SAGEConv(
                    hidden_channels,
                    hidden_channels if i < num_layers - 1 else out_channels,
                    aggr=aggr
                )
                for edge_type in self.edge_types
            }, aggr='sum')
            self.convs.append(conv)
        
        # Batch normalization
        self.norms = nn.ModuleList()
        for i in range(num_layers - 1):
            norm_dict = nn.ModuleDict()
            for node_type in self.node_types:
                norm_dict[node_type] = nn.BatchNorm1d(hidden_channels)
            self.norms.append(norm_dict)
    
    def forward(self, x_dict, edge_index_dict):
        """Forward pass."""
        # Input projection
        x_dict = {
            node_type: self.lin_dict[node_type](x)
            for node_type, x in x_dict.items()
        }
        
        # Graph convolutions
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            
            if i < self.num_layers - 1:
                x_dict = {
                    node_type: F.relu(self.norms[i][node_type](x))
                    for node_type, x in x_dict.items()
                }
                x_dict = {
                    node_type: F.dropout(x, p=self.dropout, training=self.training)
                    for node_type, x in x_dict.items()
                }
        
        return x_dict
