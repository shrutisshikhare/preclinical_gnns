"""
Trainer Classes
===============

Training loops for different tasks.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from typing import Dict, Optional, Callable
import logging
from pathlib import Path
import numpy as np
from tqdm import tqdm
import wandb

logger = logging.getLogger(__name__)


class Trainer:
    """Base trainer class."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir: str = "./models/checkpoints",
        use_wandb: bool = False
    ):
        """
        Initialize trainer.
        
        Parameters
        ----------
        model : nn.Module
            Model to train
        train_loader : DataLoader
            Training data loader
        val_loader : DataLoader, optional
            Validation data loader
        optimizer : Optimizer, optional
            Optimizer (default: Adam)
        scheduler : LRScheduler, optional
            Learning rate scheduler
        device : str
            Device for training
        checkpoint_dir : str
            Directory for saving checkpoints
        use_wandb : bool
            Whether to log to Weights & Biases
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.use_wandb = use_wandb
        
        # Optimizer
        if optimizer is None:
            self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        else:
            self.optimizer = optimizer
        
        self.scheduler = scheduler
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        raise NotImplementedError("Subclasses must implement train_epoch")
    
    def validate(self) -> Dict[str, float]:
        """Validate model."""
        raise NotImplementedError("Subclasses must implement validate")
    
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
        logger.info(f"Starting training for {num_epochs} epochs")
        
        patience_counter = 0
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}")
            
            # Validate
            if self.val_loader is not None:
                val_metrics = self.validate()
                val_loss = val_metrics['loss']
                self.val_losses.append(val_loss)
                
                logger.info(f"Epoch {epoch+1}/{num_epochs} - Val Loss: {val_loss:.4f}")
                
                # Log to wandb
                if self.use_wandb:
                    wandb.log({
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'epoch': epoch,
                        **{f'val_{k}': v for k, v in val_metrics.items() if k != 'loss'}
                    })
                
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint('best_model.pt')
                    patience_counter = 0
                    logger.info(f"New best model saved with val loss: {val_loss:.4f}")
                else:
                    patience_counter += 1
                
                # Early stopping
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                if self.use_wandb:
                    wandb.log({'train_loss': train_loss, 'epoch': epoch})
            
            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()
        
        logger.info("Training completed")
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, self.checkpoint_dir / filename)
        logger.info(f"Checkpoint saved: {filename}")
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        checkpoint = torch.load(self.checkpoint_dir / filename, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.best_val_loss = checkpoint['best_val_loss']
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"Checkpoint loaded: {filename}")


class EmbeddingTrainer(Trainer):
    """Trainer for embedding models."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        loss_fn: Optional[Callable] = None,
        **kwargs
    ):
        """Initialize embedding trainer."""
        super().__init__(model, train_loader, val_loader, **kwargs)
        
        # Default to contrastive loss if model supports it
        if loss_fn is None and hasattr(model, 'contrastive_loss'):
            self.loss_fn = model.contrastive_loss
        elif loss_fn is None:
            self.loss_fn = nn.MSELoss()
        else:
            self.loss_fn = loss_fn
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}"):
            batch = batch.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(batch.x, batch.edge_index, 
                              batch.edge_attr if hasattr(batch, 'edge_attr') else None)
            
            # Compute loss
            if isinstance(output, dict) and 'projections' in output:
                # Contrastive learning (requires augmented views)
                # This is simplified - you may need data augmentation
                loss = self.loss_fn(output['projections'], output['projections'])
            else:
                # Reconstruction or other loss
                loss = self.loss_fn(output, batch.x)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    def validate(self) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)
                
                output = self.model(batch.x, batch.edge_index,
                                  batch.edge_attr if hasattr(batch, 'edge_attr') else None)
                
                if isinstance(output, dict) and 'projections' in output:
                    loss = self.loss_fn(output['projections'], output['projections'])
                else:
                    loss = self.loss_fn(output, batch.x)
                
                total_loss += loss.item()
        
        return {'loss': total_loss / len(self.val_loader)}


class PredictionTrainer(Trainer):
    """Trainer for prediction models."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        task: str = "regression",
        metrics_fn: Optional[Callable] = None,
        **kwargs
    ):
        """
        Initialize prediction trainer.
        
        Parameters
        ----------
        task : str
            Task type: 'regression' or 'classification'
        metrics_fn : callable, optional
            Function to compute additional metrics
        """
        super().__init__(model, train_loader, val_loader, **kwargs)
        
        self.task = task
        self.metrics_fn = metrics_fn
        
        # Loss function
        if task == "regression":
            self.loss_fn = nn.MSELoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss()
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}"):
            batch = batch.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(
                batch.x, 
                batch.edge_index,
                batch.drug_id if hasattr(batch, 'drug_id') else None,
                batch.edge_attr if hasattr(batch, 'edge_attr') else None
            )
            
            # Compute loss
            loss = self.loss_fn(predictions, batch.y)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    def validate(self) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)
                
                predictions = self.model(
                    batch.x,
                    batch.edge_index,
                    batch.drug_id if hasattr(batch, 'drug_id') else None,
                    batch.edge_attr if hasattr(batch, 'edge_attr') else None
                )
                
                loss = self.loss_fn(predictions, batch.y)
                total_loss += loss.item()
                
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(batch.y.cpu().numpy())
        
        # Compute metrics
        metrics = {'loss': total_loss / len(self.val_loader)}
        
        if self.metrics_fn is not None:
            all_predictions = np.concatenate(all_predictions)
            all_targets = np.concatenate(all_targets)
            additional_metrics = self.metrics_fn(all_predictions, all_targets)
            metrics.update(additional_metrics)
        
        return metrics
