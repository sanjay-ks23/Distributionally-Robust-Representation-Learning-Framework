"""
Seed utilities for reproducibility in the DRRL Framework.

Provides functions to set seeds for all random number generators
to ensure reproducible experiments.
"""

import os
import random
from typing import Optional

import numpy as np
import torch


# Global seed storage
_GLOBAL_SEED: Optional[int] = None


def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set random seeds for reproducibility across all libraries.
    
    This function sets seeds for:
    - Python's random module
    - NumPy's random number generator
    - PyTorch (CPU and CUDA)
    - CUDA operations (if deterministic=True)
    
    Args:
        seed: The seed value to use.
        deterministic: If True, enable PyTorch deterministic algorithms.
                      Note: This may reduce performance.
    
    Example:
        >>> set_seed(42)
        >>> # All random operations are now reproducible
    """
    global _GLOBAL_SEED
    _GLOBAL_SEED = seed
    
    # Python random
    random.seed(seed)
    
    # Environment variable for some libraries
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    
    # PyTorch CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    # Deterministic operations
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Enable deterministic algorithms (PyTorch >= 1.8)
        if hasattr(torch, 'use_deterministic_algorithms'):
            try:
                torch.use_deterministic_algorithms(True)
            except RuntimeError:
                # Some operations don't have deterministic implementations
                # Fall back to allowing non-deterministic ops
                torch.use_deterministic_algorithms(False)
    else:
        # Allow cuDNN to auto-tune for better performance
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def get_seed() -> Optional[int]:
    """
    Get the currently set global seed.
    
    Returns:
        The global seed if set, None otherwise.
    """
    return _GLOBAL_SEED


def seed_worker(worker_id: int) -> None:
    """
    Worker initialization function for DataLoader reproducibility.
    
    Use this as the `worker_init_fn` argument to DataLoader to ensure
    each worker has a unique but reproducible seed.
    
    Args:
        worker_id: The worker ID assigned by DataLoader.
    
    Example:
        >>> from torch.utils.data import DataLoader
        >>> loader = DataLoader(
        ...     dataset,
        ...     num_workers=4,
        ...     worker_init_fn=seed_worker
        ... )
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_generator(seed: Optional[int] = None) -> torch.Generator:
    """
    Create a PyTorch random generator with specified seed.
    
    Useful for creating reproducible random operations within
    specific code blocks without affecting global state.
    
    Args:
        seed: Seed for the generator. Uses global seed if None.
    
    Returns:
        A seeded PyTorch Generator.
    
    Example:
        >>> g = get_generator(42)
        >>> torch.randperm(10, generator=g)
        tensor([7, 8, 0, 2, 4, 3, 1, 6, 5, 9])
    """
    if seed is None:
        seed = _GLOBAL_SEED if _GLOBAL_SEED is not None else 42
    
    g = torch.Generator()
    g.manual_seed(seed)
    return g


class SeedManager:
    """
    Context manager for temporarily changing the random seed.
    
    Useful when you need a specific random state for a block of code
    without affecting the rest of the program.
    
    Example:
        >>> set_seed(42)
        >>> with SeedManager(123):
        ...     # Operations with seed 123
        ...     pass
        >>> # Back to seed 42 state
    """
    
    def __init__(self, seed: int):
        """
        Initialize the seed manager.
        
        Args:
            seed: Temporary seed to use within the context.
        """
        self.seed = seed
        self._saved_states = {}
    
    def __enter__(self):
        """Save current states and set new seed."""
        # Save current states
        self._saved_states['random'] = random.getstate()
        self._saved_states['numpy'] = np.random.get_state()
        self._saved_states['torch'] = torch.get_rng_state()
        
        if torch.cuda.is_available():
            self._saved_states['cuda'] = torch.cuda.get_rng_state_all()
        
        # Set new seed
        set_seed(self.seed, deterministic=False)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore previous states."""
        random.setstate(self._saved_states['random'])
        np.random.set_state(self._saved_states['numpy'])
        torch.set_rng_state(self._saved_states['torch'])
        
        if torch.cuda.is_available() and 'cuda' in self._saved_states:
            torch.cuda.set_rng_state_all(self._saved_states['cuda'])
        
        return False
