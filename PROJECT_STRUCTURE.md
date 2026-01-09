# GNN-DepMap Project Structure

## Overview

Complete project structure for applying Graph Neural Networks and Graph Transformers to DepMap preclinical data.

## Directory Structure

```
gnn-depmap-project/
│
├── src/                          # Source code
│   ├── __init__.py
│   ├── data/                     # Data processing
│   │   ├── __init__.py
│   │   ├── depmap_loader.py     # Load DepMap datasets
│   │   ├── graph_builder.py     # Construct graphs
│   │   ├── preprocessor.py      # Data preprocessing
│   │   └── depmap 25q2 public/  # DepMap Q2 2025 public datasets
│   │       ├── CRISPRGeneEffect.csv
│   │       ├── GDSC1_fitted_dose_response_27Oct23.xlsx
│   │       ├── GDSC2_fitted_dose_response_27Oct23.xlsx
│   │       ├── Model.csv
│   │       ├── OmicsExpressionProteinCodingGenesTPMLogp1.csv
│   │       ├── OmicsSomaticMutations.csv
│   │       ├── ScreenGeneEffect.csv
│   │       └── hetionet_public/  # Hetionet knowledge graph data
│   │           ├── hetionet-v1.0-edges.sif
│   │           ├── hetionet-v1.0-edges.sif.gz
│   │           ├── hetionet-v1.0-nodes.tsv
│   │           └── hetionet_g2g_edgelist.csv
│   │
│   ├── models/                   # Neural network models
│   │   ├── __init__.py
│   │   ├── gnn_models.py        # GCN, GAT, GraphSAGE
│   │   ├── graph_transformer.py # Graph Transformer
│   │   ├── predictors.py        # Prediction heads
│   │   ├── hetero_gnn_models.py # Heterogeneous GNN models
│   │   ├── hetero_graph_transformer.py # Heterogeneous Graph Transformer
│   │   └── hetero_predictors.py # Heterogeneous prediction heads
│   │
│   ├── training/                 # Training utilities
│   │   ├── __init__.py
│   │   ├── trainer.py           # Training loops
│   │   ├── hetero_trainer.py    # Heterogeneous graph training
│   │   ├── loss_functions.py    # Custom losses
│   │   └── metrics.py           # Evaluation metrics
│   │
│   └── utils/                    # Utilities
│       ├── __init__.py
│       ├── logger.py            # Logging setup
│       ├── visualization.py     # Plotting functions
│       └── model_utils.py       # Model helpers
│
├── notebooks/                    # Jupyter notebooks
│   ├── 00_data_processing.ipynb # Data preprocessing and exploration
│   └── 01_embedding_generation_hetero.ipynb # Heterogeneous embedding generation
│
├── results/                      # Experimental results
│   └── model_outputs/           # Trained model outputs
│       ├── eg1/                 # Example 1 outputs
│       │   ├── attention_weights.pt
│       │   ├── hetero_gnn_model.pt
│       │   └── node_embeddings.pt
│       └── eg2/                 # Example 2 outputs
│           ├── hgt_model.pt
│           └── node_embeddings.pt
│
├── README.md                     # Project README
├── HETERO_README.md             # Heterogeneous graph documentation
├── PROJECT_STRUCTURE.md         # This file - project structure
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
├── LICENSE                       # MIT License
└── CHANGELOG.md                  # Version history
```

## Key Components

### 1. Data Processing (`src/data/`)
- **DepMapDataLoader**: Load gene expression, mutations, drug response data
- **GraphBuilder**: Construct cell line graphs (KNN, threshold, heterogeneous)
- **DataPreprocessor**: Normalization, imputation, feature selection

### 2. Models (`src/models/`)
- **GNN Encoders**: GCN, GAT, GraphSAGE implementations
- **Graph Transformer**: Self-attention for graphs
- **Heterogeneous GNN Models**: Models for heterogeneous graphs with multiple node/edge types
- **Heterogeneous Graph Transformer**: Self-attention for heterogeneous graphs (HGT)
- **Predictors**: Drug response prediction, embedding extraction
- **Heterogeneous Predictors**: Prediction heads for heterogeneous graph outputs
- **Multi-task Learning**: Simultaneous prediction of multiple targets

### 3. Training (`src/training/`)
- **Trainers**: Embedding and prediction training loops
- **Heterogeneous Trainer**: Training loops for heterogeneous graphs
- **Loss Functions**: MSE, MAE, contrastive, focal, ranking losses
- **Metrics**: Regression/classification metrics, embedding quality

### 4. Utilities (`src/utils/`)
- **Logging**: Structured logging setup
- **Visualization**: Training curves, embeddings, predictions
- **Model Utils**: Save/load, parameter counting, device management

## Main Tasks

### Task 1: Data Processing and Exploration
Process and explore DepMap datasets including expression, mutations, and drug response data.

**Key Files:**
- `notebooks/00_data_processing.ipynb`
- `src/data/depmap_loader.py`
- `src/data/preprocessor.py`

**Data Sources:**
- DepMap Q2 2025 public release datasets
- Hetionet knowledge graph data
- GDSC drug response data

### Task 2: Heterogeneous Embedding Generation
Generate embeddings using heterogeneous graph neural networks incorporating multiple data types and knowledge graphs.

**Key Files:**
- `notebooks/01_embedding_generation_hetero.ipynb`
- `src/models/hetero_gnn_models.py`
- `src/models/hetero_graph_transformer.py`
- `src/training/hetero_trainer.py`

**Models:**
- Heterogeneous Graph Transformer (HGT)
- Heterogeneous Graph Attention Network
- Multi-modal graph encoders

**Output:**
- Cell line embeddings incorporating multiple data modalities
- Node embeddings for genes, drugs, and pathways
- Saved in `results/model_outputs/`

## Configuration

Model configurations and hyperparameters are currently defined within the notebooks and Python modules. The project uses:

- Notebook-based configuration for experimental runs
- Model-specific parameters defined in training scripts
- Data paths and preprocessing options set in data loaders

**Current Configuration Approach:**
- Hyperparameters defined in notebook cells
- Model architecture parameters in model classes
- Training settings in trainer modules

## Getting Started

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Data is included:**
   - DepMap Q2 2025 data is already in `src/data/depmap 25q2 public/`
   - Hetionet knowledge graph data in `src/data/depmap 25q2 public/hetionet_public/`

3. **Start with data processing:**
   ```bash
   jupyter notebook notebooks/00_data_processing.ipynb
   ```

4. **Generate embeddings:**
   ```bash
   jupyter notebook notebooks/01_embedding_generation_hetero.ipynb
   ```

5. **Check results:**
   - Model outputs saved in `results/model_outputs/`
   - Pre-trained models available in `eg1/` and `eg2/` directories

## Data Requirements

### Included Data (in `src/data/depmap 25q2 public/`)

1. **Expression Data**
   - `OmicsExpressionProteinCodingGenesTPMLogp1.csv` - Gene expression matrix
   - `depmap_expression_data.csv` - Processed expression data

2. **Mutation Data**
   - `OmicsSomaticMutations.csv` - Somatic mutations
   - `OmicsSomaticMutationsMatrixDamaging.csv` - Damaging mutations matrix
   - `depmap_mutation_data.csv` - Processed mutation data

3. **Drug Response Data**
   - `GDSC1_fitted_dose_response_27Oct23.xlsx` - GDSC1 drug responses
   - `GDSC2_fitted_dose_response_27Oct23.xlsx` - GDSC2 drug responses
   - `depmap_gdsc_drug_response.csv` - Processed drug response data

4. **CRISPR Screening Data**
   - `CRISPRGeneDependency.csv` - Gene dependency scores
   - `CRISPRGeneEffect.csv` - CRISPR gene effects
   - `ScreenGeneEffect.csv` - Screen gene effects
   - `depmap_gene_effect_data.csv` - Processed gene effect data

5. **Copy Number Data**
   - `OmicsCNGeneWGS.csv` - Copy number variations

6. **Knowledge Graph Data** (in `hetionet_public/`)
   - `hetionet-v1.0-nodes.tsv` - Hetionet nodes
   - `hetionet-v1.0-edges.sif` - Hetionet edges
   - `hetionet_g2g_edgelist.csv` - Gene-gene relationships

7. **Metadata**
   - `Model.csv` - Cell line model information
   - `depmap_clinical_data.csv` - Clinical annotations
   - `model_ids_train_test_split.csv` - Train/test splits

### Data Sources

- [DepMap Portal](https://depmap.org/portal/download/)
- [CCLE](https://sites.broadinstitute.org/ccle/)
- [GDSC](https://www.cancerrxgene.org/)

## Model Architectures

### Homogeneous Graph Models

#### GCN (Graph Convolutional Network)
- Classic message-passing GNN
- Fast and efficient
- Good baseline

#### GAT (Graph Attention Network)
- Attention-based aggregation
- Learns edge importance
- Better for heterogeneous data

#### GraphSAGE
- Sampling-based approach
- Scalable to large graphs
- Inductive learning

#### Graph Transformer
- Self-attention for graphs
- Captures long-range dependencies
- State-of-the-art performance

### Heterogeneous Graph Models

#### Heterogeneous Graph Transformer (HGT)
- Self-attention for heterogeneous graphs
- Different attention mechanisms for different node/edge types
- Incorporates knowledge graphs and multi-modal data

#### Heterogeneous Graph Attention Network
- Attention-based aggregation for heterogeneous graphs
- Type-specific attention weights
- Multi-relational message passing

## Evaluation

### Embedding Quality
- Linear probe accuracy
- Silhouette score
- UMAP/t-SNE visualization

### Drug Response Prediction
- MSE, MAE, RMSE
- Pearson/Spearman correlation
- R² score
- Per-drug metrics

## Outputs

### Embeddings
- Format: PyTorch (.pt) tensors
- Location: `results/model_outputs/`
- Types: Node embeddings, attention weights
- Includes: Cell line embeddings, gene embeddings, drug embeddings

### Models
- Format: PyTorch (.pt) model files
- Location: `results/model_outputs/eg1/`, `results/model_outputs/eg2/`
- Available models:
  - `hetero_gnn_model.pt` - Heterogeneous GNN model
  - `hgt_model.pt` - Heterogeneous Graph Transformer
  - `attention_weights.pt` - Learned attention patterns
  - `node_embeddings.pt` - Generated node embeddings

### Notebooks Output
- Interactive analysis and visualization
- Data processing results
- Model training progress
- Embedding quality assessment

## License

MIT License - See LICENSE file for details
