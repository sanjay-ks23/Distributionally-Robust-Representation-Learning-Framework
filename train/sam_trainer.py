"""
SAM (Sharpness-Aware Minimization) trainer.
"""

import torch
from typing import Dict, Optional, Any
from torch.utils.data import DataLoader

from train.base_trainer import BaseTrainer
from robust.sam import SAM
from utils.helpers import AverageMeter
from utils.logging_utils import Logger


class SAMTrainer(BaseTrainer):
    """
    Training with Sharpness-Aware Minimization.
    
    Uses two-step gradient updates to find flat minima.
    """
    
    def __init__(
        self,
        model,
        train_loader: DataLoader,
        val_loader: DataLoader,
        base_optimizer: torch.optim.Optimizer,
        rho: float = 0.05,
        adaptive: bool = False,
        scheduler: Optional[Any] = None,
        device: torch.device = None,
        logger: Optional[Logger] = None,
        save_dir: str = './outputs',
        epochs: int = 50,
        **kwargs
    ):
        # Wrap base optimizer with SAM
        sam_optimizer = SAM(
            model.parameters(),
            base_optimizer,
            rho=rho,
            adaptive=adaptive
        )
        
        super().__init__(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=sam_optimizer,
            scheduler=scheduler,
            device=device,
            logger=logger,
            save_dir=save_dir,
            epochs=epochs,
            **kwargs
        )
        
        self.rho = rho
        self.adaptive = adaptive
    
    def train_epoch(self) -> Dict[str, float]:
        """Train one epoch with SAM."""
        self.model.train()
        
        loss_meter = AverageMeter('loss')
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            inputs, targets, groups = self._unpack_batch(batch)
            
            # First forward-backward pass
            logits, _ = self.model(inputs)
            loss = self.criterion(logits, targets)
            loss.backward()
            self.optimizer.first_step(zero_grad=True)
            
            # Second forward-backward pass
            logits_2, _ = self.model(inputs)
            loss_2 = self.criterion(logits_2, targets)
            loss_2.backward()
            self.optimizer.second_step(zero_grad=True)
            
            # Track metrics (use first loss for logging)
            loss_meter.update(loss.item(), inputs.size(0))
            preds = logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            
            self.global_step += 1
            
            if batch_idx % self.log_interval == 0 and self.logger:
                self.logger.log_scalar('train/batch_loss', loss.item(), self.global_step)
        
        return {
            'loss': loss_meter.avg,
            'accuracy': correct / total
        }
