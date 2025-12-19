"""
ERM (Empirical Risk Minimization) trainer.
"""

import torch
from typing import Dict

from train.base_trainer import BaseTrainer
from utils.helpers import AverageMeter


class ERMTrainer(BaseTrainer):
    """
    Standard ERM training with cross-entropy loss.
    
    This is the baseline trainer that minimizes average loss.
    """
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch using standard ERM."""
        self.model.train()
        
        loss_meter = AverageMeter('loss')
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            inputs, targets, groups = self._unpack_batch(batch)
            
            self.optimizer.zero_grad()
            
            logits, _ = self.model(inputs)
            loss = self.criterion(logits, targets)
            
            loss.backward()
            
            # Gradient clipping
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
            
            self.global_step += 1
            
            # Logging
            if batch_idx % self.log_interval == 0:
                if self.logger:
                    self.logger.log_scalar('train/batch_loss', loss.item(), self.global_step)
        
        return {
            'loss': loss_meter.avg,
            'accuracy': correct / total
        }
