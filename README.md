# Heterogeneous GNNs for Preclinical Drug Response

Heterogeneous graph neural network models for predicting drug response using DepMap preclinical data.

This project is a public re-upload from a Drug Discovery DS/AI Hackathon from 2025 Q1.
## Overview

This project implements heterogeneous graph neural networks to model multi-relational biological data from the DepMap consortium.  
The goal is to predict drug response (LN_IC50) for cell line–drug pairs while learning meaningful embeddings for cell lines, genes, and drugs.

The approach leverages heterogeneous message passing to integrate gene expression, mutations, gene effects, drug targets, and gene–gene interactions in a single graph.

## Data

- Dataset: DepMap (+ GDSC drug response)
- Nodes: cancer cell lines, genes, drugs
- Edges: expression, mutation, gene effect, drug response, drug–target, gene–gene interactions
- Target variable: LN_IC50 (cell line–drug edges)

## Graph Structure

- Graph type: heterogeneous
- Node types: `cell_line`, `gene`, `drug`
- Edge types:
  - cell_line → gene (expression, mutation, gene effect)
  - cell_line → drug (treated_with; LN_IC50)
  - drug → gene (targets)
  - gene → gene (interacts)
- Prediction task: link regression on (cell_line, treated_with, drug)

## Model

- Architectures: HeteroGraphTransformer, HeteroGAT, HeteroGCN, HeteroSAGE
- Core model: HeteroGraphTransformer with multi-head attention
- Frameworks: PyTorch, PyTorch Geometric
- Output: node embeddings + drug response prediction

## Training

- Task: link prediction (regression)
- Loss function: MSE
- Metrics: R², MAE, RMSE
- Optimizer: Adam
- Hardware: GPU (CUDA)

## Usage
Follow notebooks/01_embedding_generation_hetero.ipynb

## File Structure
src/
├── models/        # heterogeneous GNN encoders and predictors
├── training/      # training loops and trainers
notebooks/
├── 00_data_processing.ipynb
├── 01_embedding_generation_hetero.ipynb
README.md
