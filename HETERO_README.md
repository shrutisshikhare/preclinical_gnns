# Heterogeneous Graph Neural Networks for DepMap

This directory contains implementations of heterogeneous GNN models for DepMap consortium data analysis.

## 🎯 Overview

The heterogeneous models support graphs with multiple node types (cell lines, genes, drugs) and multiple edge types (expression, mutation, drug response, etc.). This enables:

1. **Multi-relational learning**: Capture different types of relationships simultaneously
2. **Link prediction**: Predict drug response (LN_IC50) for cell line-drug pairs
3. **Multi-type embeddings**: Learn representations for different biological entities

## 📊 Graph Structure

### Node Types
- `cell_line`: Cancer cell lines with metadata features
- `gene`: Genes with identity or embedding features
- `drug`: Drugs with identity or embedding features

### Edge Types
- `(cell_line, expresses, gene)`: Gene expression levels
- `(cell_line, has_mutation, gene)`: Damaging mutations
- `(cell_line, has_gene_effect, gene)`: CRISPR gene effects
- `(cell_line, treated_with, drug)`: **Drug response (LN_IC50)** ⭐
- `(drug, targets, gene)`: Drug-gene target relationships
- `(gene, interacts, gene)`: Gene-gene interactions (from Hetionet)
- Reverse edges for bidirectional message passing

## 🏗️ Architecture

### Heterogeneous Encoders

#### 1. HeteroGraphTransformer (Recommended)
```python
from src.models import HeteroGraphTransformer

encoder = HeteroGraphTransformer(
    metadata=(node_types, edge_types),
    in_channels_dict={'cell_line': 128, 'gene': 19202, 'drug': 203},
    hidden_channels=256,
    out_channels=128,
    num_layers=3,
    num_heads=8,
    dropout=0.1,
    use_edge_attr=True
)
```

**Features:**
- Multi-head attention mechanism
- Edge attribute support
- Layer normalization and residual connections
- Best performance on drug response prediction

#### 2. HeteroGATEncoder
```python
from src.models import HeteroGATEncoder

encoder = HeteroGATEncoder(
    metadata=metadata,
    in_channels_dict=in_channels_dict,
    hidden_channels=64,  # per head
    out_channels=128,
    num_layers=3,
    num_heads=4,
    dropout=0.1
)
```

**Features:**
- Graph Attention Networks
- Learns attention weights for different edge types
- Good for interpretable attention patterns

#### 3. HeteroGCNEncoder
```python
from src.models import HeteroGCNEncoder

encoder = HeteroGCNEncoder(
    metadata=metadata,
    in_channels_dict=in_channels_dict,
    hidden_channels=256,
    out_channels=128,
    num_layers=3,
    dropout=0.1
)
```

**Features:**
- Simple and efficient
- Spectral graph convolutions
- Fast training

#### 4. HeteroSAGEEncoder
```python
from src.models import HeteroSAGEEncoder

encoder = HeteroSAGEEncoder(
    metadata=metadata,
    in_channels_dict=in_channels_dict,
    hidden_channels=256,
    out_channels=128,
    num_layers=3,
    dropout=0.1,
    aggr='mean'
)
```

**Features:**
- Inductive learning capability
- Sampling-based aggregation
- Scalable to large graphs

### Predictor Models

#### HeteroDrugResponsePredictor
Link prediction model for drug response (LN_IC50):

```python
from src.models import HeteroDrugResponsePredictor

model = HeteroDrugResponsePredictor(
    encoder=encoder,
    embedding_dim=128,
    hidden_dims=[256, 128, 64],
    dropout=0.3,
    use_edge_features=False,
    activation='relu'
)
```

**Prediction Process:**
1. Encode heterogeneous graph → get embeddings for all node types
2. Extract embeddings for source (cell_line) and destination (drug) nodes
3. Concatenate: `[cell_line_emb || drug_emb]`
4. Pass through MLP → predict LN_IC50

**Methods:**
- `forward()`: Predict response for existing edges
- `predict_new_edges()`: Predict response for new cell line-drug pairs

#### HeteroEmbeddingModel
Extract embeddings for downstream tasks:

```python
from src.models import HeteroEmbeddingModel

emb_model = HeteroEmbeddingModel(
    encoder=encoder,
    projection_dim=64,
    node_types=['cell_line', 'gene', 'drug'],
    use_contrastive=True,
    temperature=0.07
)
```

**Features:**
- Contrastive learning (optional)
- Separate projection heads per node type
- NT-Xent loss for self-supervised learning

## 🚀 Training

### Link Prediction Training

```python
from src.training import HeteroLinkPredictionTrainer

trainer = HeteroLinkPredictionTrainer(
    model=drug_response_model,
    train_data=train_hetero_graph,
    val_data=test_hetero_graph,
    optimizer=torch.optim.Adam(model.parameters(), lr=0.0001),
    device='cuda',
    target_edge_type=('cell_line', 'treated_with', 'drug'),
    loss_fn=torch.nn.MSELoss()
)

history = trainer.train(num_epochs=100, patience=15)
```

**Training Features:**
- Automatic early stopping
- Learning rate scheduling
- Checkpoint saving/loading
- Metrics: MSE, MAE

### Embedding Training

```python
from src.training import HeteroEmbeddingTrainer

emb_trainer = HeteroEmbeddingTrainer(
    model=embedding_model,
    train_data=train_hetero_graph,
    val_data=test_hetero_graph,
    optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
    device='cuda',
    target_node_type='cell_line'
)

history = emb_trainer.train(num_epochs=50, patience=10)
```

## 📓 Usage Example

See `notebooks/01_embedding_generation_hetero.ipynb` for complete workflow.

### Quick Start

```python
import torch
from src.models import HeteroGraphTransformer, HeteroDrugResponsePredictor
from src.training import HeteroLinkPredictionTrainer

# 1. Load heterogeneous graph
train_graph = torch.load('data/processed/depmap_hetero_graph_train.pt')
test_graph = torch.load('data/processed/depmap_hetero_graph_test.pt')

# 2. Prepare metadata
metadata = (train_graph.node_types, train_graph.edge_types)
in_channels_dict = {
    'cell_line': train_graph['cell_line'].x.shape[1],
    'gene': train_graph['gene'].x.shape[1],
    'drug': train_graph['drug'].x.shape[1]
}

# 3. Create model
encoder = HeteroGraphTransformer(
    metadata=metadata,
    in_channels_dict=in_channels_dict,
    hidden_channels=256,
    out_channels=128,
    num_layers=3,
    num_heads=8,
    dropout=0.1
)

model = HeteroDrugResponsePredictor(
    encoder=encoder,
    embedding_dim=128,
    hidden_dims=[256, 128, 64],
    dropout=0.3
)

# 4. Train
trainer = HeteroLinkPredictionTrainer(
    model=model,
    train_data=train_graph,
    val_data=test_graph,
    device='cuda'
)

history = trainer.train(num_epochs=100, patience=15)

# 5. Predict
predictions = model.predict_new_edges(
    train_graph.x_dict,
    train_graph.edge_index_dict,
    new_edge_index=torch.tensor([[0, 1], [10, 15]])
)
```

## 📈 Performance

### Drug Response Prediction Results

Typical performance on DepMap + GDSC data:
- **R² Score**: 0.65-0.75
- **MAE**: 0.8-1.2
- **RMSE**: 1.0-1.5

Best practices:
- Use HeteroGraphTransformer for best R²
- Include edge attributes for improved accuracy
- Train for 50-100 epochs with early stopping
- Use learning rate scheduling

## 🔧 File Structure

```
src/models/
├── hetero_gnn_models.py         # HeteroGCN, HeteroGAT, HeteroSAGE
├── hetero_graph_transformer.py  # HeteroGraphTransformer
├── hetero_predictors.py         # HeteroDrugResponsePredictor, HeteroEmbeddingModel
└── __init__.py

src/training/
├── hetero_trainer.py            # HeteroEmbeddingTrainer, HeteroLinkPredictionTrainer
└── __init__.py

notebooks/
├── 00_data_processing.ipynb             # Graph construction
└── 01_embedding_generation_hetero.ipynb # Training & evaluation
```

## 🎓 Key Concepts

### Heterogeneous Message Passing

```python
# For each edge type (src_type, relation, dst_type):
#   1. Project source and destination features
#   2. Apply relation-specific transformation
#   3. Aggregate messages at destination nodes
#   4. Combine messages from different edge types

# Example for (cell_line, treated_with, drug):
cell_emb = encoder(x_dict, edge_index_dict)['cell_line']  # [N_cells, 128]
drug_emb = encoder(x_dict, edge_index_dict)['drug']       # [N_drugs, 128]
```

### Edge Attributes

Edge attributes (e.g., LN_IC50, expression levels) can be incorporated:

```python
# Pass edge_attr_dict to encoder
embeddings = encoder(
    x_dict=graph.x_dict,
    edge_index_dict=graph.edge_index_dict,
    edge_attr_dict=graph.edge_attr_dict  # Optional
)
```

### Link Prediction

Predict existence/properties of edges:

```python
# Concatenate source and destination embeddings
src_emb = embeddings['cell_line'][edge_index[0]]  # [E, 128]
dst_emb = embeddings['drug'][edge_index[1]]        # [E, 128]
edge_emb = torch.cat([src_emb, dst_emb], dim=-1)  # [E, 256]

# Predict edge property
prediction = mlp(edge_emb)  # [E, 1]
```

## 🔬 Advanced Usage

### Custom Edge Weighting

```python
# Weight edges by biological significance
edge_weights = {
    ('cell_line', 'expresses', 'gene'): 1.0,
    ('cell_line', 'has_mutation', 'gene'): 2.0,  # Emphasize mutations
    ('drug', 'targets', 'gene'): 1.5
}
```

### Ablation Studies

Test contribution of different edge types:

```python
# Remove specific edge types
ablation_edge_types = [et for et in edge_types if 'mutation' not in et[1]]
```

### Transfer Learning

```python
# Pre-train on one dataset, fine-tune on another
# 1. Pre-train embeddings
emb_model.train()  # On large dataset

# 2. Transfer encoder
encoder_pretrained = emb_model.encoder
drug_model = HeteroDrugResponsePredictor(encoder=encoder_pretrained, ...)

# 3. Fine-tune
drug_model.train()  # On target dataset
```

## 🐛 Troubleshooting

### Memory Issues
- Reduce `hidden_channels` (256 → 128)
- Decrease `num_layers` (3 → 2)
- Use `HeteroGCNEncoder` instead of `HeteroGraphTransformer`
- Enable gradient checkpointing

### Poor Performance
- Increase model capacity (`hidden_channels`, `num_layers`)
- Try different encoders (GAT often works well)
- Add edge attributes (`use_edge_attr=True`)
- Normalize edge attributes
- Use learning rate scheduling

### NaN Loss
- Reduce learning rate (0.001 → 0.0001)
- Add gradient clipping (already in trainers)
- Check for inf/nan in input features
- Use batch normalization

## 📚 References

- PyTorch Geometric: https://pytorch-geometric.readthedocs.io/
- Heterogeneous Graph Attention Networks (HAN): https://arxiv.org/abs/1903.07293
- Graph Transformer Networks: https://arxiv.org/abs/1911.06455
- DepMap Consortium: https://depmap.org/

## 🤝 Contributing

To add new heterogeneous models:
1. Implement in `src/models/hetero_*.py`
2. Use `HeteroConv` for message passing
3. Handle `x_dict` and `edge_index_dict` inputs
4. Add to `__init__.py`
5. Create example notebook

## 📄 License

MIT License - see LICENSE file for details
