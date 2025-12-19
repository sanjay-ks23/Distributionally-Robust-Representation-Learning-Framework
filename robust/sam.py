"""
Sharpness-Aware Minimization (SAM) optimizer.

SAM seeks parameters that lie in neighborhoods with uniformly low loss,
leading to flatter minima and better generalization.

Reference: Foret et al. "Sharpness-Aware Minimization for Efficiently
           Improving Generalization" (ICLR 2021)
"""

import torch
from torch.optim import Optimizer
from typing import Callable, Optional


class SAM(Optimizer):
    """
    Sharpness-Aware Minimization optimizer.
    
    Wraps a base optimizer and performs two-step gradient updates:
    1. Perturb weights in the direction of maximum loss increase
    2. Compute gradients at perturbed point and update original weights
    
    Example:
        >>> base_optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        >>> optimizer = SAM(model.parameters(), base_optimizer, rho=0.05)
        >>> 
        >>> # Training step
        >>> loss = criterion(model(inputs), targets)
        >>> loss.backward()
        >>> optimizer.first_step(zero_grad=True)
        >>> 
        >>> criterion(model(inputs), targets).backward()
        >>> optimizer.second_step(zero_grad=True)
    """
    
    def __init__(
        self,
        params,
        base_optimizer: Optimizer,
        rho: float = 0.05,
        adaptive: bool = False
    ):
        """
        Initialize SAM optimizer.
        
        Args:
            params: Model parameters to optimize.
            base_optimizer: Base optimizer (e.g., SGD, Adam).
            rho: Perturbation radius.
            adaptive: Use adaptive SAM (ASAM) variant.
        """
        assert rho >= 0.0, f"Invalid rho: {rho}"
        
        defaults = dict(rho=rho, adaptive=adaptive)
        super(SAM, self).__init__(params, defaults)
        
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        
        for group in self.param_groups:
            group['rho'] = rho
            group['adaptive'] = adaptive
    
    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        """
        Compute perturbation and move to perturbed point.
        
        Args:
            zero_grad: Whether to zero gradients after step.
        """
        grad_norm = self._grad_norm()
        
        for group in self.param_groups:
            scale = group['rho'] / (grad_norm + 1e-12)
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Store original parameters
                self.state[p]['old_p'] = p.data.clone()
                
                # Compute perturbation
                if group['adaptive']:
                    # ASAM: scale by parameter magnitudes
                    e_w = (torch.pow(p, 2) * p.grad * scale).to(p)
                else:
                    e_w = p.grad * scale
                
                # Move to perturbed point
                p.add_(e_w)
        
        if zero_grad:
            self.zero_grad()
    
    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        """
        Restore original parameters and apply gradient update.
        
        Args:
            zero_grad: Whether to zero gradients after step.
        """
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Restore original parameters
                p.data = self.state[p]['old_p']
        
        # Apply base optimizer update
        self.base_optimizer.step()
        
        if zero_grad:
            self.zero_grad()
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> torch.Tensor:
        """
        Perform full SAM step (both first and second steps).
        
        Args:
            closure: Closure that reevaluates the model and returns loss.
            
        Returns:
            Loss value from the second forward pass.
        """
        assert closure is not None, "SAM requires closure for step()"
        
        # First step
        self.first_step(zero_grad=True)
        
        # Second forward pass at perturbed point
        with torch.enable_grad():
            loss = closure()
        
        # Second step
        self.second_step(zero_grad=True)
        
        return loss
    
    def _grad_norm(self) -> torch.Tensor:
        """Compute the norm of all gradients."""
        # Use shared device of the first parameter
        shared_device = self.param_groups[0]['params'][0].device
        
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group['adaptive'] else 1.0) * p.grad)
                .norm(p=2).to(shared_device)
                for group in self.param_groups
                for p in group['params']
                if p.grad is not None
            ]),
            p=2
        )
        return norm
    
    def load_state_dict(self, state_dict: dict) -> None:
        """Load optimizer state."""
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


def create_sam_optimizer(
    model_params,
    base_optimizer_class,
    rho: float = 0.05,
    adaptive: bool = False,
    **optimizer_kwargs
) -> SAM:
    """
    Create a SAM optimizer with specified base optimizer.
    
    Args:
        model_params: Model parameters.
        base_optimizer_class: Base optimizer class (e.g., torch.optim.SGD).
        rho: Perturbation radius.
        adaptive: Use adaptive SAM.
        **optimizer_kwargs: Arguments for base optimizer.
        
    Returns:
        SAM optimizer instance.
    """
    base_optimizer = base_optimizer_class(model_params, **optimizer_kwargs)
    return SAM(model_params, base_optimizer, rho=rho, adaptive=adaptive)
