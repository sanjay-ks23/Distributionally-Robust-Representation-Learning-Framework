"""
DRO (Distributionally Robust Optimization) trainer.
"""

import torch
import numpy as np
from typing import Dict, Optional, Any
from torch.utils.data import DataLoader

from train.base_trainer import BaseTrainer
from robust.dro import GroupDRO
from utils.helpers import AverageMeter
from utils.logging_utils import Logger


class DROTrainer(BaseTrainer):
    """
    Training with Group Distributionally Robust Optimization.
    
    Optimizes worst-group performance by dynamically reweighting groups.
    """
    
    def __init__(
        self,
        model,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        n_groups: int,
        dro_step_size: float = 0.01,
        scheduler: Optional[Any] = None,
        device: torch.device = None,
        logger: Optional[Logger] = None,
        save_dir: str = './outputs',
        epochs: int = 50,
        **kwargs
    ):
        super().__init__(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            logger=logger,
            save_dir=save_dir,
            epochs=epochs,
            **kwargs
        )
        
        self.n_groups = n_groups
        self.group_dro = GroupDRO(
            n_groups=n_groups,
            step_size=dro_step_size
        ).to(self.device)
        
        # Track per-group metrics
        self.group_accuracies = np.zeros(n_groups)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train one epoch with GroupDRO."""
        self.model.train()
        
        loss_meter = AverageMeter('loss')
        correct = 0
        total = 0
        
        # Per-group tracking
        group_correct = np.zeros(self.n_groups)
        group_total = np.zeros(self.n_groups)
        
        for batch_idx, batch in enumerate(self.train_loader):
            inputs, targets, groups = self._unpack_batch(batch)
            
            self.optimizer.zero_grad()
            
            logits, _ = self.model(inputs)
            
            # Compute DRO loss
            loss = self.group_dro(logits, targets, groups)
            
            loss.backward()
            
            if self.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip
                )
            
            self.optimizer.step()
            
            # Track metrics
            loss_meter.update(loss.item(), inputs.size(0))
            preds = logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            
            # Per-group accuracy
            for g in range(self.n_groups):
                mask = (groups == g)
                if mask.sum() > 0:
                    group_correct[g] += (preds[mask] == targets[mask]).sum().item()
                    group_total[g] += mask.sum().item()
            
            self.global_step += 1
            
            if batch_idx % self.log_interval == 0 and self.logger:
                self.logger.log_scalar('train/batch_loss', loss.item(), self.global_step)
                
                # Log group weights
                weights = self.group_dro.get_group_weights()
                for g, w in enumerate(weights):
                    self.logger.log_scalar(f'train/group_{g}_weight', w, self.global_step)
        
        # Compute per-group accuracies
        self.group_accuracies = group_correct / (group_total + 1e-8)
        
        return {
            'loss': loss_meter.avg,
            'accuracy': correct / total,
            'worst_group_accuracy': self.group_accuracies.min()
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate with per-group metrics."""
        self.model.eval()
        
        loss_meter = AverageMeter('loss')
        correct = 0
        total = 0
        
        group_correct = np.zeros(self.n_groups)
        group_total = np.zeros(self.n_groups)
        
        for batch in self.val_loader:
            inputs, targets, groups = self._unpack_batch(batch)
            
            logits, _ = self.model(inputs)
            loss = self.criterion(logits, targets)
            
            loss_meter.update(loss.item(), inputs.size(0))
            
            preds = logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            
            for g in range(self.n_groups):
                mask = (groups == g)
                if mask.sum() > 0:
                    group_correct[g] += (preds[mask] == targets[mask]).sum().item()
                    group_total[g] += mask.sum().item()
        
        self.model.train()
        
        group_accuracies = group_correct / (group_total + 1e-8)
        
        metrics = {
            'loss': loss_meter.avg,
            'accuracy': correct / total,
            'worst_group_accuracy': group_accuracies.min()
        }
        
        for g, acc in enumerate(group_accuracies):
            metrics[f'group_{g}_accuracy'] = acc
        
        return metrics
