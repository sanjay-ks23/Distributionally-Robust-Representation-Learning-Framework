"""
Base trainer class for the DRRL Framework.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, Callable
from pathlib import Path
from abc import ABC, abstractmethod

from utils.helpers import AverageMeter, EarlyStopping, save_checkpoint
from utils.logging_utils import Logger


class BaseTrainer(ABC):
    """
    Abstract base class for all trainers.
    
    Provides common training loop structure, logging, and checkpointing.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        device: torch.device = None,
        logger: Optional[Logger] = None,
        save_dir: str = './outputs',
        epochs: int = 50,
        log_interval: int = 10,
        early_stopping_patience: int = 10,
        gradient_clip: Optional[float] = 1.0
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or torch.device('cpu')
        self.logger = logger
        self.save_dir = Path(save_dir)
        self.epochs = epochs
        self.log_interval = log_interval
        self.gradient_clip = gradient_clip
        
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.model.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_acc = 0.0
        
        self.early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            mode='max',
            save_path=self.save_dir / 'best_model.pt'
        )
        
        # Metrics tracking
        self.train_metrics = {}
        self.val_metrics = {}
    
    def train(self) -> Dict[str, float]:
        """Run complete training loop."""
        self._log("Starting training...")
        
        for epoch in range(self.epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch()
            self.train_metrics = train_metrics
            
            # Validate
            val_metrics = self.validate()
            self.val_metrics = val_metrics
            
            # Log metrics
            self._log_epoch_metrics(train_metrics, val_metrics)
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if hasattr(self.scheduler, 'step'):
                    self.scheduler.step()
            
            # Early stopping
            val_acc = val_metrics.get('accuracy', 0.0)
            if self.early_stopping(val_acc, self.model, epoch):
                self._log(f"Early stopping at epoch {epoch}")
                break
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self._save_checkpoint(is_best=True)
        
        self._log(f"Training complete. Best val accuracy: {self.best_val_acc:.4f}")
        return self.val_metrics
    
    @abstractmethod
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch. Must be implemented by subclasses."""
        pass
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate model on validation set."""
        self.model.eval()
        
        loss_meter = AverageMeter('loss')
        correct = 0
        total = 0
        
        for batch in self.val_loader:
            inputs, targets, groups = self._unpack_batch(batch)
            
            logits, _ = self.model(inputs)
            loss = self.criterion(logits, targets)
            
            loss_meter.update(loss.item(), inputs.size(0))
            
            preds = logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
        
        self.model.train()
        
        return {
            'loss': loss_meter.avg,
            'accuracy': correct / total
        }
    
    def _unpack_batch(self, batch):
        """Unpack batch and move to device."""
        inputs, targets, groups = batch
        return (
            inputs.to(self.device),
            targets.to(self.device),
            groups.to(self.device)
        )
    
    def _log(self, message: str):
        """Log message."""
        if self.logger:
            self.logger.info(message)
        else:
            print(message)
    
    def _log_epoch_metrics(self, train_metrics: Dict, val_metrics: Dict):
        """Log epoch metrics."""
        msg = f"Epoch {self.current_epoch}: "
        msg += f"train_loss={train_metrics.get('loss', 0):.4f} "
        msg += f"train_acc={train_metrics.get('accuracy', 0):.4f} "
        msg += f"val_loss={val_metrics.get('loss', 0):.4f} "
        msg += f"val_acc={val_metrics.get('accuracy', 0):.4f}"
        self._log(msg)
        
        if self.logger:
            self.logger.log_metrics(train_metrics, self.current_epoch, prefix='train')
            self.logger.log_metrics(val_metrics, self.current_epoch, prefix='val')
    
    def _save_checkpoint(self, is_best: bool = False):
        """Save training checkpoint."""
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=self.current_epoch,
            metrics=self.val_metrics,
            save_path=self.save_dir / f'checkpoint_epoch_{self.current_epoch}.pt',
            scheduler=self.scheduler,
            is_best=is_best
        )
