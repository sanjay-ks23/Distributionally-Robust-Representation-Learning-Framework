"""
Evaluator class for the DRRL Framework.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple, List

from eval.metrics import MetricsTracker


class Evaluator:
    """
    Comprehensive model evaluator.
    
    Computes metrics, extracts embeddings, and generates predictions.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        n_groups: int = 4,
        n_classes: int = 2
    ):
        self.model = model
        self.device = device
        self.n_groups = n_groups
        self.n_classes = n_classes
    
    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        return_embeddings: bool = False
    ) -> Dict:
        """
        Evaluate model on a dataset.
        
        Args:
            dataloader: DataLoader for evaluation
            return_embeddings: Whether to return embeddings
            
        Returns:
            Dictionary with metrics and optionally embeddings
        """
        self.model.eval()
        
        tracker = MetricsTracker(self.n_groups, self.n_classes)
        
        all_embeddings = []
        all_targets = []
        all_groups = []
        
        for batch in dataloader:
            inputs, targets, groups = batch
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            groups = groups.to(self.device)
            
            logits, embeddings = self.model(inputs, return_embeddings=True)
            predictions = logits.argmax(dim=1)
            
            tracker.update(predictions, targets, groups)
            
            if return_embeddings and embeddings is not None:
                all_embeddings.append(embeddings.cpu())
                all_targets.append(targets.cpu())
                all_groups.append(groups.cpu())
        
        results = tracker.compute()
        
        if return_embeddings:
            results['embeddings'] = torch.cat(all_embeddings, dim=0).numpy()
            results['targets'] = torch.cat(all_targets, dim=0).numpy()
            results['groups'] = torch.cat(all_groups, dim=0).numpy()
        
        return results
    
    @torch.no_grad()
    def get_predictions(
        self,
        dataloader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get all predictions for a dataset.
        
        Returns:
            Tuple of (predictions, probabilities, targets, groups)
        """
        self.model.eval()
        
        all_preds = []
        all_probs = []
        all_targets = []
        all_groups = []
        
        for batch in dataloader:
            inputs, targets, groups = batch
            inputs = inputs.to(self.device)
            
            logits, _ = self.model(inputs)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_targets.append(targets.numpy())
            all_groups.append(groups.numpy())
        
        return (
            np.concatenate(all_preds),
            np.concatenate(all_probs),
            np.concatenate(all_targets),
            np.concatenate(all_groups)
        )
    
    @torch.no_grad()
    def extract_embeddings(
        self,
        dataloader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract embeddings from the model.
        
        Returns:
            Tuple of (embeddings, targets, groups)
        """
        self.model.eval()
        
        embeddings = []
        targets = []
        groups = []
        
        for batch in dataloader:
            inputs, t, g = batch
            inputs = inputs.to(self.device)
            
            emb = self.model.get_embeddings(inputs)
            
            embeddings.append(emb.cpu().numpy())
            targets.append(t.numpy())
            groups.append(g.numpy())
        
        return (
            np.concatenate(embeddings),
            np.concatenate(targets),
            np.concatenate(groups)
        )


def compare_models(
    models: Dict[str, nn.Module],
    dataloader: DataLoader,
    device: torch.device,
    n_groups: int = 4,
    n_classes: int = 2
) -> Dict[str, Dict]:
    """
    Compare multiple models on the same dataset.
    
    Args:
        models: Dictionary mapping model names to models
        dataloader: DataLoader for evaluation
        device: Compute device
        
    Returns:
        Dictionary mapping model names to their metrics
    """
    results = {}
    
    for name, model in models.items():
        evaluator = Evaluator(model, device, n_groups, n_classes)
        results[name] = evaluator.evaluate(dataloader)
    
    return results
