"""
Heterogeneous Graph Trainer
============================

Training loops for heterogeneous graph models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from typing import Dict, Optional, Callable, Tuple
import logging
from pathlib import Path
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)



class HeteroEmbeddingTrainer:
    """Trainer for heterogeneous graph embedding models."""
    
    def __init__(
        self,
        model: nn.Module,
        train_data,  # HeteroData object
        val_data = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir: str = "./models/checkpoints",
        target_node_type: str = "cell_line"
    ):
        """
        Initialize heterogeneous embedding trainer.
        
        Parameters
        ----------
        model : nn.Module
            Heterogeneous embedding model
        train_data : HeteroData
            Training graph
        val_data : HeteroData, optional
            Validation graph
        optimizer : Optimizer, optional
            Optimizer (default: Adam)
        scheduler : LRScheduler, optional
            Learning rate scheduler
        device : str
            Device for training
        checkpoint_dir : str
            Directory for saving checkpoints
        target_node_type : str
            Primary node type for evaluation
        """
        self.model = model.to(device)
        self.train_data = train_data.to(device)
        self.val_data = val_data.to(device) if val_data is not None else None
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.target_node_type = target_node_type
        
        # Optimizer
        if optimizer is None:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        else:
            self.optimizer = optimizer
        
        self.scheduler = scheduler
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.history = {
            'train_loss': [],
            'val_loss': []
        }
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        
        # Forward pass
        result = self.model(
            self.train_data.x_dict,
            self.train_data.edge_index_dict,
            self.train_data.edge_attr_dict if hasattr(self.train_data, 'edge_attr_dict') else None
        )
        
        # Compute loss
        if isinstance(result, tuple):
            # Contrastive learning
            embeddings_dict, proj_dict = result
            loss = self.model.contrastive_loss(proj_dict, self.target_node_type)
        else:
            # Reconstruction loss (optional)
            embeddings_dict = result
            # For now, use a simple reconstruction loss
            loss = self._compute_reconstruction_loss(embeddings_dict)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def _compute_reconstruction_loss(self, embeddings_dict: Dict) -> torch.Tensor:
        """
        Compute reconstruction loss using node features.
        
        This is a simple approach - reconstruct node features from embeddings.
        """
        loss = 0.0
        count = 0
        
        for node_type, emb in embeddings_dict.items():
            if node_type in self.train_data.x_dict:
                # Simple MSE between original features and reconstructed features
                # We'd need a decoder here, so for now just compute self-similarity
                # This encourages smooth embeddings
                similarity = torch.matmul(emb, emb.T)
                target = torch.eye(emb.shape[0], device=emb.device)
                loss += F.mse_loss(similarity, target)
                count += 1
        
        return loss / max(count, 1)
    
    def validate(self) -> Dict[str, float]:
        """Validate model."""
        if self.val_data is None:
            return {}
        
        self.model.eval()
        with torch.no_grad():
            result = self.model(
                self.val_data.x_dict,
                self.val_data.edge_index_dict,
                self.val_data.edge_attr_dict if hasattr(self.val_data, 'edge_attr_dict') else None
            )
            
            if isinstance(result, tuple):
                embeddings_dict, proj_dict = result
                val_loss = self.model.contrastive_loss(proj_dict, self.target_node_type)
            else:
                embeddings_dict = result
                val_loss = self._compute_reconstruction_loss(embeddings_dict)
        
        return {'val_loss': val_loss.item()}
    
    def train(self, num_epochs: int, patience: int = 10):
        """
        Train model.
        
        Parameters
        ----------
        num_epochs : int
            Number of epochs
        patience : int
            Early stopping patience
        """
        
        patience_counter = 0
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch()
            self.history['train_loss'].append(train_loss)
            
            # Validate
            if self.val_data is not None:
                val_metrics = self.validate()
                val_loss = val_metrics.get('val_loss', float('inf'))
                self.history['val_loss'].append(val_loss)
                
                # Early stopping
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    patience_counter = 0
                    self.save_checkpoint('best_model.pt')
                else:
                    patience_counter += 1
                
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs} - "
                    f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )
                
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}")
            
            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()
        
        logger.info("Training completed")
        return self.history
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint_path = self.checkpoint_dir / filename
        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history
        }, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        checkpoint_path = self.checkpoint_dir / filename
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
        logger.info(f"Checkpoint loaded from {checkpoint_path}")


class HeteroLinkPredictionTrainer:
    """Trainer for heterogeneous graph link prediction (drug response)."""
    
    def __init__(
        self,
        model: nn.Module,
        train_data,  # HeteroData
        val_data = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir: str = "./models/checkpoints",
        target_edge_type: Tuple[str, str, str] = ('cell_line', 'treated_with', 'drug'),
        loss_fn: Optional[Callable] = None
    ):
        """
        Initialize heterogeneous link prediction trainer.
        
        Parameters
        ----------
        model : nn.Module
            Heterogeneous drug response predictor
        train_data : HeteroData
            Training graph
        val_data : HeteroData, optional
            Validation graph
        optimizer : Optimizer, optional
            Optimizer
        scheduler : LRScheduler, optional
            Learning rate scheduler
        device : str
            Device for training
        checkpoint_dir : str
            Directory for checkpoints
        target_edge_type : tuple
            Edge type for link prediction
        loss_fn : callable, optional
            Loss function (default: MSE for regression)
        """
        self.model = model.to(device)
        self.train_data = train_data.to(device)
        self.val_data = val_data.to(device) if val_data is not None else None
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.target_edge_type = target_edge_type
        
        # Loss function
        if loss_fn is None:
            self.loss_fn = nn.MSELoss()
        else:
            self.loss_fn = loss_fn
        
        # Optimizer
        if optimizer is None:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        else:
            self.optimizer = optimizer
        
        self.scheduler = scheduler
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': []
        }
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        # Forward pass
        predictions = self.model(
            self.train_data.x_dict,
            self.train_data.edge_index_dict,
            target_edge_type=self.target_edge_type,
            edge_attr_dict=self.train_data.edge_attr_dict if hasattr(self.train_data, 'edge_attr_dict') else None
        )
        
        # Get ground truth
        edge_attr = self.train_data[self.target_edge_type].edge_attr
        if edge_attr.dim() > 1:
            targets = edge_attr.squeeze(-1)
        else:
            targets = edge_attr
        
        # Compute loss
        loss = self.loss_fn(predictions, targets)
        mae = F.l1_loss(predictions, targets)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return {
            'loss': loss.item(),
            'mae': mae.item()
        }
    
    def validate(self) -> Dict[str, float]:
        """Validate model."""
        if self.val_data is None:
            return {}
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(
                self.val_data.x_dict,
                self.val_data.edge_index_dict,
                target_edge_type=self.target_edge_type,
                edge_attr_dict=self.val_data.edge_attr_dict if hasattr(self.val_data, 'edge_attr_dict') else None
            )
            
            edge_attr = self.val_data[self.target_edge_type].edge_attr
            if edge_attr.dim() > 1:
                targets = edge_attr.squeeze(-1)
            else:
                targets = edge_attr
            
            val_loss = self.loss_fn(predictions, targets)
            val_mae = F.l1_loss(predictions, targets)
        
        return {
            'val_loss': val_loss.item(),
            'val_mae': val_mae.item()
        }
    
    def train(self, num_epochs: int, patience: int = 15):
        """Train model with early stopping."""
        logger.info(f"Starting link prediction training for {num_epochs} epochs")
        
        patience_counter = 0
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_mae'].append(train_metrics['mae'])
            
            # Validate
            if self.val_data is not None:
                val_metrics = self.validate()
                self.history['val_loss'].append(val_metrics['val_loss'])
                self.history['val_mae'].append(val_metrics['val_mae'])
                
                # Early stopping
                if val_metrics['val_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['val_loss']
                    patience_counter = 0
                    self.save_checkpoint('best_model.pt')
                else:
                    patience_counter += 1
                
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs} - "
                    f"Train Loss: {train_metrics['loss']:.4f}, "
                    f"Train MAE: {train_metrics['mae']:.4f}, "
                    f"Val Loss: {val_metrics['val_loss']:.4f}, "
                    f"Val MAE: {val_metrics['val_mae']:.4f}"
                )
                
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs} - "
                    f"Train Loss: {train_metrics['loss']:.4f}, "
                    f"Train MAE: {train_metrics['mae']:.4f}"
                )
            
            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()
        
        logger.info("Training completed")
        return self.history
    
    def save_checkpoint(self, filename: str):
        """Save checkpoint."""
        checkpoint_path = self.checkpoint_dir / filename
        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history
        }, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, filename: str):
        """Load checkpoint."""
        checkpoint_path = self.checkpoint_dir / filename
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
