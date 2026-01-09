"""
Graph Builder
=============

Module for constructing graph structures from DepMap data for GNN models 

"""

import numpy as np
import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import Data, HeteroData
import logging
logger = logging.getLogger(__name__)

# %pip install --upgrade torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
import torch
from torch_geometric.data import HeteroData
import numpy as np

def build_depmap_heterogeneous_graph(
    model_features,
    gene_features,
    drug_features,
    cell_line_to_expression,
    cell_line_to_mutation,
    cell_line_to_gene_effect,
    cell_line_to_drug,
    drug_to_gene,
    gene_to_gene
):
    """
    Build a heterogeneous graph from DepMap data (optimized version).
    
    Parameters
    --------
    model_features : pd.DataFrame
        Cell line features with 'ModelID' column
    gene_features : pd.DataFrame
        Gene features with 'gene' column
    drug_features : pd.DataFrame
        Drug features with 'drugs' column
    cell_line_to_expression : pd.DataFrame
        Edges: ['ModelID', 'gene', 'expression']
    cell_line_to_mutation : pd.DataFrame
        Edges: ['ModelID', 'gene', 'mutation']
    cell_line_to_gene_effect : pd.DataFrame
        Edges: ['ModelID', 'gene', 'gene_effect']
    cell_line_to_drug : pd.DataFrame
        Edges: ['ModelID', 'DRUG_NAME', 'LN_IC50']
    drug_to_gene : pd.DataFrame
        Edges: ['DRUG_NAME', 'PUTATIVE_TARGET']
    gene_to_gene : pd.DataFrame
        Edges: ['source_name', 'target_name']
        
    Returns
    ------------
    HeteroData
        PyTorch Geometric HeteroData object
    """
    print("Building heterogeneous graph (optimized)...")
    data = HeteroData()
    
    # ==================== Node Features ====================
    
    # Cell line nodes
    cell_line_ids = model_features['ModelID'].values
    cell_line_to_idx = {cell_id: idx for idx, cell_id in enumerate(cell_line_ids)}
    
    # Extract numeric features for cell lines
    numeric_features = model_features.drop('ModelID', axis=1).values
    data['cell_line'].x = torch.FloatTensor(numeric_features)
    data['cell_line'].num_nodes = len(cell_line_ids)
    data['cell_line'].node_ids = cell_line_ids.tolist()
    
    print(f"Cell line nodes: {data['cell_line'].num_nodes}")
    print(f"Cell line features: {data['cell_line'].x.shape}")
    
    # Gene nodes
    gene_ids = gene_features['gene'].values
    gene_to_idx = {gene: idx for idx, gene in enumerate(gene_ids)}
    
    # Create simple identity features for genes (can be replaced with embeddings)
    data['gene'].x = torch.eye(len(gene_ids))
    data['gene'].num_nodes = len(gene_ids)
    data['gene'].node_ids = gene_ids.tolist()
    
    print(f"Gene nodes: {data['gene'].num_nodes}")
    
    # Drug nodes
    drug_ids = drug_features['drugs'].values
    drug_to_idx = {drug: idx for idx, drug in enumerate(drug_ids)}
    
    # Create simple identity features for drugs (can be replaced with embeddings)
    data['drug'].x = torch.eye(len(drug_ids))
    data['drug'].num_nodes = len(drug_ids)
    data['drug'].node_ids = drug_ids.tolist()
    
    print(f"Drug nodes: {data['drug'].num_nodes}")
    
    # ==================== Cell Line -> Gene Edges (VECTORIZED) ====================
    
    # Expression edges
    print("\nBuilding cell line -> gene (expression) edges...")
    expression_edges = cell_line_to_expression.dropna(subset=['expression'])
    
    # Vectorized mapping using pandas map
    expression_edges['cell_idx'] = expression_edges['ModelID'].map(cell_line_to_idx)
    expression_edges['gene_idx'] = expression_edges['gene'].map(gene_to_idx)
    
    # Filter valid edges
    valid_mask = expression_edges['cell_idx'].notna() & expression_edges['gene_idx'].notna()
    expression_edges = expression_edges[valid_mask]
    
    if len(expression_edges) > 0:
        cell_indices = expression_edges['cell_idx'].values.astype(np.int64)
        gene_indices = expression_edges['gene_idx'].values.astype(np.int64)
        expression_values = expression_edges['expression'].values.astype(np.float32)
        
        data['cell_line', 'expresses', 'gene'].edge_index = torch.LongTensor(np.stack([cell_indices, gene_indices]))
        data['cell_line', 'expresses', 'gene'].edge_attr = torch.FloatTensor(expression_values).unsqueeze(1)
        print(f"Expression edges: {len(cell_indices)}")
    
    # Mutation edges
    print("Building cell line -> gene (mutation) edges...")
    mutation_edges = cell_line_to_mutation[cell_line_to_mutation['mutation'] == 1].copy()
    
    mutation_edges['cell_idx'] = mutation_edges['ModelID'].map(cell_line_to_idx)
    mutation_edges['gene_idx'] = mutation_edges['gene'].map(gene_to_idx)
    
    valid_mask = mutation_edges['cell_idx'].notna() & mutation_edges['gene_idx'].notna()
    mutation_edges = mutation_edges[valid_mask]
    
    if len(mutation_edges) > 0:
        cell_indices = mutation_edges['cell_idx'].values.astype(np.int64)
        gene_indices = mutation_edges['gene_idx'].values.astype(np.int64)
        
        data['cell_line', 'has_mutation', 'gene'].edge_index = torch.LongTensor(np.stack([cell_indices, gene_indices]))
        print(f"Mutation edges: {len(cell_indices)}")
    
    # Gene effect edges
    print("Building cell line -> gene (gene effect) edges...")
    gene_effect_edges = cell_line_to_gene_effect.dropna(subset=['gene_effect']).copy()
    
    gene_effect_edges['cell_idx'] = gene_effect_edges['ModelID'].map(cell_line_to_idx)
    gene_effect_edges['gene_idx'] = gene_effect_edges['gene'].map(gene_to_idx)
    
    valid_mask = gene_effect_edges['cell_idx'].notna() & gene_effect_edges['gene_idx'].notna()
    gene_effect_edges = gene_effect_edges[valid_mask]
    
    if len(gene_effect_edges) > 0:
        cell_indices = gene_effect_edges['cell_idx'].values.astype(np.int64)
        gene_indices = gene_effect_edges['gene_idx'].values.astype(np.int64)
        gene_effect_values = gene_effect_edges['gene_effect'].values.astype(np.float32)
        
        data['cell_line', 'has_gene_effect', 'gene'].edge_index = torch.LongTensor(np.stack([cell_indices, gene_indices]))
        data['cell_line', 'has_gene_effect', 'gene'].edge_attr = torch.FloatTensor(gene_effect_values).unsqueeze(1)
        print(f"Gene effect edges: {len(cell_indices)}")
    
    # ==================== Cell Line -> Drug Edges ====================
    
    print("\nBuilding cell line -> drug edges...")
    cell_drug_edges = cell_line_to_drug.copy()
    
    cell_drug_edges['cell_idx'] = cell_drug_edges['ModelID'].map(cell_line_to_idx)
    cell_drug_edges['drug_idx'] = cell_drug_edges['DRUG_NAME'].map(drug_to_idx)
    

    valid_mask = cell_drug_edges['cell_idx'].notna() & cell_drug_edges['drug_idx'].notna()
    cell_drug_edges = cell_drug_edges[valid_mask]
    
    if len(cell_drug_edges) > 0:
        cell_indices = cell_drug_edges['cell_idx'].values.astype(np.int64)
        drug_indices = cell_drug_edges['drug_idx'].values.astype(np.int64)
        ic50_values = cell_drug_edges['LN_IC50'].fillna(0.0).values.astype(np.float32)
        
        data['cell_line', 'treated_with', 'drug'].edge_index = torch.LongTensor(np.stack([cell_indices, drug_indices]))
        data['cell_line', 'treated_with', 'drug'].edge_attr = torch.FloatTensor(ic50_values).unsqueeze(1)

        print(f"Cell line -> drug edges: {len(cell_indices)}")
    
    # ==================== Drug -> Gene Edges ====================
    
    print("\nBuilding drug -> gene (target) edges...")
    drug_gene_edges = drug_to_gene.copy()
    
    drug_gene_edges['drug_idx'] = drug_gene_edges['DRUG_NAME'].map(drug_to_idx)
    drug_gene_edges['gene_idx'] = drug_gene_edges['PUTATIVE_TARGET'].map(gene_to_idx)
    
    valid_mask = drug_gene_edges['drug_idx'].notna() & drug_gene_edges['gene_idx'].notna()
    drug_gene_edges = drug_gene_edges[valid_mask]
    
    if len(drug_gene_edges) > 0:
        drug_indices = drug_gene_edges['drug_idx'].values.astype(np.int64)
        gene_indices = drug_gene_edges['gene_idx'].values.astype(np.int64)
        # ic50_values = drug_gene_edges['LN_IC50'].fillna(0.0).values.astype(np.float32)
        
        data['drug', 'targets', 'gene'].edge_index = torch.LongTensor(np.stack([drug_indices, gene_indices]))
        # data['drug', 'targets', 'gene'].edge_attr = torch.FloatTensor(ic50_values).unsqueeze(1)
        print(f"Drug -> gene edges: {len(drug_indices)}")
    
    # ==================== Gene -> Gene Edges ====================
    
    print("\nBuilding gene -> gene (interaction) edges...")
    gene_gene_edges = gene_to_gene.copy()
    
    gene_gene_edges['source_idx'] = gene_gene_edges['source'].map(gene_to_idx)
    gene_gene_edges['target_idx'] = gene_gene_edges['target'].map(gene_to_idx)
    
    valid_mask = gene_gene_edges['source_idx'].notna() & gene_gene_edges['target_idx'].notna()
    gene_gene_edges = gene_gene_edges[valid_mask]
    
    if len(gene_gene_edges) > 0:
        source_indices = gene_gene_edges['source_idx'].values.astype(np.int64)
        target_indices = gene_gene_edges['target_idx'].values.astype(np.int64)
        
        data['gene', 'interacts', 'gene'].edge_index = torch.LongTensor(np.stack([source_indices, target_indices]))
        print(f"Gene -> gene edges: {len(source_indices)}")
    
    # ==================== Add Reverse Edges ====================
    
    print("\nAdding reverse edges...")
    
    # Gene -> Cell line (reverse of expression)
    if hasattr(data['cell_line', 'expresses', 'gene'], 'edge_index'):
        edge_index = data['cell_line', 'expresses', 'gene'].edge_index
        data['gene', 'expressed_by', 'cell_line'].edge_index = torch.stack([edge_index[1], edge_index[0]])
        if hasattr(data['cell_line', 'expresses', 'gene'], 'edge_attr'):
            data['gene', 'expressed_by', 'cell_line'].edge_attr = data['cell_line', 'expresses', 'gene'].edge_attr
    
    # Gene -> Cell line (reverse of mutation)
    if hasattr(data['cell_line', 'has_mutation', 'gene'], 'edge_index'):
        edge_index = data['cell_line', 'has_mutation', 'gene'].edge_index
        data['gene', 'mutated_in', 'cell_line'].edge_index = torch.stack([edge_index[1], edge_index[0]])

    # Gene -> Cell line (reverse of gene effect)
    if hasattr(data['cell_line', 'has_gene_effect', 'gene'], 'edge_index'):
        edge_index = data['cell_line', 'has_gene_effect', 'gene'].edge_index
        data['gene', 'has_gene_effect_in', 'cell_line'].edge_index = torch.stack([edge_index[1], edge_index[0]])
    
    # Drug -> Cell line (reverse of treated_with)
    if hasattr(data['cell_line', 'treated_with', 'drug'], 'edge_index'):
        edge_index = data['cell_line', 'treated_with', 'drug'].edge_index
        data['drug', 'treats', 'cell_line'].edge_index = torch.stack([edge_index[1], edge_index[0]])
    
    # Gene -> Drug (reverse of targets)
    if hasattr(data['drug', 'targets', 'gene'], 'edge_index'):
        edge_index = data['drug', 'targets', 'gene'].edge_index
        data['gene', 'targeted_by', 'drug'].edge_index = torch.stack([edge_index[1], edge_index[0]])
        if hasattr(data['drug', 'targets', 'gene'], 'edge_attr'):
            data['gene', 'targeted_by', 'drug'].edge_attr = data['drug', 'targets', 'gene'].edge_attr
    
    print("\n" + "="*60)
    print("Graph construction complete!")
    print(f"Node types: {list(data.node_types)}")
    print(f"Edge types: {list(data.edge_types)}")
    print("="*60)
    
    return data