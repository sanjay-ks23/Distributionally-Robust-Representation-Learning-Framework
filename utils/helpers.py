"""
Helper utilities for the DRRL Framework.

Provides common utility functions for device management, checkpointing,
timing, and other general-purpose operations.
"""

import os
import time
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, Union
from contextlib import contextmanager
from functools import wraps

import torch
import torch.nn as nn


def get_device(device: str = "auto") -> torch.device:
    """
    Get the appropriate device for computation.
    
    Args:
        device: Device specification. Options:
            - "auto": Automatically select best available device
            - "cuda": Use CUDA GPU
            - "cuda:N": Use specific CUDA device N
            - "mps": Use Apple Metal Performance Shaders
            - "cpu": Use CPU
    
    Returns:
        torch.device object for the selected device.
    
    Example:
        >>> device = get_device("auto")
        >>> model = model.to(device)
    """
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    return torch.device(device)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Count the number of parameters in a model.
    
    Args:
        model: PyTorch model.
        trainable_only: If True, count only trainable parameters.
    
    Returns:
        Number of parameters.
    
    Example:
        >>> model = nn.Linear(100, 10)
        >>> count_parameters(model)
        1010
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def format_parameters(n_params: int) -> str:
    """
    Format parameter count in human-readable form.
    
    Args:
        n_params: Number of parameters.
    
    Returns:
        Formatted string (e.g., "1.5M", "500K").
    """
    if n_params >= 1e9:
        return f"{n_params / 1e9:.1f}B"
    elif n_params >= 1e6:
        return f"{n_params / 1e6:.1f}M"
    elif n_params >= 1e3:
        return f"{n_params / 1e3:.1f}K"
    return str(n_params)


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path.
    
    Returns:
        Path object for the directory.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    save_path: Union[str, Path],
    scheduler: Optional[Any] = None,
    is_best: bool = False,
    extra_state: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save a training checkpoint.
    
    Args:
        model: Model to save.
        optimizer: Optimizer state to save.
        epoch: Current epoch number.
        metrics: Dictionary of metrics (e.g., loss, accuracy).
        save_path: Path to save the checkpoint.
        scheduler: Optional learning rate scheduler.
        is_best: If True, also save as 'best_model.pt'.
        extra_state: Optional additional state to save.
    
    Example:
        >>> save_checkpoint(
        ...     model, optimizer, epoch=10,
        ...     metrics={'val_acc': 0.95},
        ...     save_path='./checkpoints/model_10.pt',
        ...     is_best=True
        ... )
    """
    save_path = Path(save_path)
    ensure_dir(save_path.parent)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    if extra_state is not None:
        checkpoint.update(extra_state)
    
    torch.save(checkpoint, save_path)
    
    if is_best:
        best_path = save_path.parent / 'best_model.pt'
        shutil.copy(str(save_path), str(best_path))


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None,
    strict: bool = True
) -> Dict[str, Any]:
    """
    Load a training checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file.
        model: Model to load state into.
        optimizer: Optional optimizer to load state into.
        scheduler: Optional scheduler to load state into.
        device: Device to load the checkpoint to.
        strict: If True, require exact key matching for model state.
    
    Returns:
        Dictionary containing checkpoint info (epoch, metrics, etc.).
    
    Example:
        >>> info = load_checkpoint(
        ...     'checkpoints/best_model.pt',
        ...     model, optimizer
        ... )
        >>> print(f"Loaded from epoch {info['epoch']}")
    """
    checkpoint_path = Path(checkpoint_path)
    
    if device is None:
        device = get_device()
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return {
        'epoch': checkpoint.get('epoch', 0),
        'metrics': checkpoint.get('metrics', {})
    }


class Timer:
    """
    Context manager and utility for timing code execution.
    
    Example:
        >>> timer = Timer()
        >>> with timer:
        ...     # Code to time
        ...     pass
        >>> print(f"Elapsed: {timer.elapsed:.2f}s")
        
        >>> # Or use as decorator
        >>> @Timer.decorate
        ... def my_function():
        ...     pass
    """
    
    def __init__(self, name: Optional[str] = None, verbose: bool = False):
        """
        Initialize timer.
        
        Args:
            name: Optional name for the timer.
            verbose: If True, print elapsed time on exit.
        """
        self.name = name
        self.verbose = verbose
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.elapsed: float = 0.0
    
    def __enter__(self):
        """Start the timer."""
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the timer and calculate elapsed time."""
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time
        
        if self.verbose:
            name_str = f"[{self.name}] " if self.name else ""
            print(f"{name_str}Elapsed time: {self.elapsed:.4f}s")
        
        return False
    
    def start(self) -> 'Timer':
        """Manually start the timer."""
        self.start_time = time.perf_counter()
        return self
    
    def stop(self) -> float:
        """Manually stop the timer and return elapsed time."""
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time
        return self.elapsed
    
    def reset(self) -> 'Timer':
        """Reset the timer."""
        self.start_time = None
        self.end_time = None
        self.elapsed = 0.0
        return self
    
    @staticmethod
    def decorate(func):
        """Decorator to time function execution."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            timer = Timer(name=func.__name__, verbose=True)
            with timer:
                result = func(*args, **kwargs)
            return result
        return wrapper


class AverageMeter:
    """
    Computes and stores the average and current value.
    
    Useful for tracking metrics like loss or accuracy over batches.
    
    Example:
        >>> meter = AverageMeter('loss')
        >>> for batch in batches:
        ...     loss = compute_loss(batch)
        ...     meter.update(loss, n=batch_size)
        >>> print(f"Average loss: {meter.avg:.4f}")
    """
    
    def __init__(self, name: str = ''):
        """
        Initialize the meter.
        
        Args:
            name: Name of the metric being tracked.
        """
        self.name = name
        self.reset()
    
    def reset(self) -> None:
        """Reset all statistics."""
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0
    
    def update(self, val: float, n: int = 1) -> None:
        """
        Update the meter with a new value.
        
        Args:
            val: Value to add.
            n: Number of samples (for weighted average).
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0
    
    def __str__(self) -> str:
        """String representation."""
        return f"{self.name}: {self.val:.4f} (avg: {self.avg:.4f})"


class EarlyStopping:
    """
    Early stopping to stop training when validation metric stops improving.
    
    Example:
        >>> early_stop = EarlyStopping(patience=10, mode='max')
        >>> for epoch in range(100):
        ...     val_acc = validate(model)
        ...     if early_stop(val_acc, model):
        ...         print("Early stopping triggered!")
        ...         break
    """
    
    def __init__(
        self,
        patience: int = 10,
        mode: str = 'min',
        min_delta: float = 0.0,
        save_path: Optional[Union[str, Path]] = None
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait for improvement.
            mode: 'min' or 'max' - whether to minimize or maximize the metric.
            min_delta: Minimum change to qualify as an improvement.
            save_path: Path to save best model checkpoint.
        """
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.save_path = Path(save_path) if save_path else None
        
        self.counter = 0
        self.best_score: Optional[float] = None
        self.best_epoch: int = 0
        self.should_stop = False
        
        if mode == 'min':
            self.is_better = lambda x, best: x < best - min_delta
        else:
            self.is_better = lambda x, best: x > best + min_delta
    
    def __call__(self, score: float, model: Optional[nn.Module] = None, epoch: int = 0) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current validation metric.
            model: Optional model to save if this is the best score.
            epoch: Current epoch number.
        
        Returns:
            True if training should stop.
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            if model is not None and self.save_path is not None:
                self._save_model(model)
        elif self.is_better(score, self.best_score):
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            if model is not None and self.save_path is not None:
                self._save_model(model)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop
    
    def _save_model(self, model: nn.Module) -> None:
        """Save the model checkpoint."""
        ensure_dir(self.save_path.parent)
        torch.save(model.state_dict(), self.save_path)
    
    def reset(self) -> None:
        """Reset the early stopping state."""
        self.counter = 0
        self.best_score = None
        self.best_epoch = 0
        self.should_stop = False


@contextmanager
def torch_eval_mode(model: nn.Module):
    """
    Context manager to temporarily set model to evaluation mode.
    
    Args:
        model: PyTorch model.
    
    Example:
        >>> with torch_eval_mode(model):
        ...     outputs = model(inputs)
    """
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        if was_training:
            model.train()
