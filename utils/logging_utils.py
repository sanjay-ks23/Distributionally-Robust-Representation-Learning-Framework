"""
Logging utilities for the DRRL Framework.

Provides a unified Logger class supporting Weights & Biases and TensorBoard
for experiment tracking, metric logging, and visualization uploads.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union, List
from datetime import datetime
from abc import ABC, abstractmethod

import numpy as np


# Global logger instance
_LOGGER_INSTANCE: Optional['Logger'] = None


class LoggerBackend(ABC):
    """Abstract base class for logging backends."""
    
    @abstractmethod
    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a scalar value."""
        pass
    
    @abstractmethod
    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int) -> None:
        """Log multiple scalar values."""
        pass
    
    @abstractmethod
    def log_image(self, tag: str, image: np.ndarray, step: int) -> None:
        """Log an image."""
        pass
    
    @abstractmethod
    def log_figure(self, tag: str, figure: Any, step: int) -> None:
        """Log a matplotlib figure."""
        pass
    
    @abstractmethod
    def log_histogram(self, tag: str, values: np.ndarray, step: int) -> None:
        """Log a histogram of values."""
        pass
    
    @abstractmethod
    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters."""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close the logger and flush any pending writes."""
        pass


class TensorBoardBackend(LoggerBackend):
    """TensorBoard logging backend."""
    
    def __init__(self, log_dir: str, experiment_name: str):
        """
        Initialize TensorBoard backend.
        
        Args:
            log_dir: Directory for TensorBoard logs.
            experiment_name: Name of the experiment.
        """
        from torch.utils.tensorboard import SummaryWriter
        
        self.log_path = Path(log_dir) / experiment_name
        self.log_path.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_path))
    
    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a scalar value."""
        self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int) -> None:
        """Log multiple scalar values."""
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def log_image(self, tag: str, image: np.ndarray, step: int) -> None:
        """Log an image (expects HWC or CHW format)."""
        if image.ndim == 3 and image.shape[0] in [1, 3, 4]:
            # Already in CHW format
            self.writer.add_image(tag, image, step)
        else:
            # Convert HWC to CHW
            self.writer.add_image(tag, image, step, dataformats='HWC')
    
    def log_figure(self, tag: str, figure: Any, step: int) -> None:
        """Log a matplotlib figure."""
        self.writer.add_figure(tag, figure, step)
    
    def log_histogram(self, tag: str, values: np.ndarray, step: int) -> None:
        """Log a histogram of values."""
        self.writer.add_histogram(tag, values, step)
    
    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters."""
        # Flatten nested dicts for TensorBoard
        flat_params = self._flatten_dict(params)
        self.writer.add_hparams(flat_params, {})
    
    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '/') -> Dict:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def close(self) -> None:
        """Close the TensorBoard writer."""
        self.writer.close()


class WandBBackend(LoggerBackend):
    """Weights & Biases logging backend."""
    
    def __init__(
        self,
        project_name: str,
        experiment_name: str,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ):
        """
        Initialize Weights & Biases backend.
        
        Args:
            project_name: W&B project name.
            experiment_name: Name of this run.
            config: Configuration dictionary to log.
            tags: Optional tags for the run.
        """
        import wandb
        
        self.wandb = wandb
        self.run = wandb.init(
            project=project_name,
            name=experiment_name,
            config=config,
            tags=tags,
            reinit=True
        )
    
    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a scalar value."""
        self.wandb.log({tag: value}, step=step)
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int) -> None:
        """Log multiple scalar values."""
        prefixed = {f"{main_tag}/{k}": v for k, v in tag_scalar_dict.items()}
        self.wandb.log(prefixed, step=step)
    
    def log_image(self, tag: str, image: np.ndarray, step: int) -> None:
        """Log an image."""
        self.wandb.log({tag: self.wandb.Image(image)}, step=step)
    
    def log_figure(self, tag: str, figure: Any, step: int) -> None:
        """Log a matplotlib figure."""
        self.wandb.log({tag: self.wandb.Image(figure)}, step=step)
    
    def log_histogram(self, tag: str, values: np.ndarray, step: int) -> None:
        """Log a histogram of values."""
        self.wandb.log({tag: self.wandb.Histogram(values)}, step=step)
    
    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters (already done in init)."""
        self.wandb.config.update(params)
    
    def close(self) -> None:
        """Finish the W&B run."""
        self.wandb.finish()


class ConsoleBackend(LoggerBackend):
    """Console/file logging backend."""
    
    def __init__(self, log_dir: str, experiment_name: str, level: int = logging.INFO):
        """
        Initialize console/file logging backend.
        
        Args:
            log_dir: Directory for log files.
            experiment_name: Name of the experiment.
            level: Logging level.
        """
        self.log_path = Path(log_dir) / experiment_name
        self.log_path.mkdir(parents=True, exist_ok=True)
        
        # Setup logger
        self.logger = logging.getLogger(f"drrl.{experiment_name}")
        self.logger.setLevel(level)
        self.logger.handlers = []
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_format = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
        
        # File handler
        file_handler = logging.FileHandler(self.log_path / 'training.log')
        file_handler.setLevel(level)
        file_format = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        self.logger.addHandler(file_handler)
    
    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a scalar value."""
        self.logger.info(f"Step {step} | {tag}: {value:.6f}")
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int) -> None:
        """Log multiple scalar values."""
        values_str = " | ".join([f"{k}: {v:.6f}" for k, v in tag_scalar_dict.items()])
        self.logger.info(f"Step {step} | {main_tag} | {values_str}")
    
    def log_image(self, tag: str, image: np.ndarray, step: int) -> None:
        """Log an image (save to disk)."""
        from PIL import Image
        img = Image.fromarray(image.astype(np.uint8))
        img.save(self.log_path / f"{tag.replace('/', '_')}_{step}.png")
    
    def log_figure(self, tag: str, figure: Any, step: int) -> None:
        """Log a matplotlib figure (save to disk)."""
        figure.savefig(self.log_path / f"{tag.replace('/', '_')}_{step}.png", dpi=150, bbox_inches='tight')
    
    def log_histogram(self, tag: str, values: np.ndarray, step: int) -> None:
        """Log histogram statistics."""
        self.logger.info(
            f"Step {step} | {tag} | mean: {values.mean():.4f}, "
            f"std: {values.std():.4f}, min: {values.min():.4f}, max: {values.max():.4f}"
        )
    
    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters to file."""
        import json
        with open(self.log_path / 'hyperparams.json', 'w') as f:
            json.dump(params, f, indent=2, default=str)
        self.logger.info(f"Hyperparameters saved to {self.log_path / 'hyperparams.json'}")
    
    def info(self, message: str) -> None:
        """Log an info message."""
        self.logger.info(message)
    
    def warning(self, message: str) -> None:
        """Log a warning message."""
        self.logger.warning(message)
    
    def error(self, message: str) -> None:
        """Log an error message."""
        self.logger.error(message)
    
    def close(self) -> None:
        """Close all handlers."""
        for handler in self.logger.handlers:
            handler.close()


class Logger:
    """
    Unified logger for the DRRL Framework.
    
    Supports multiple backends (TensorBoard, W&B, Console) and provides
    a consistent API for logging metrics, images, and figures.
    
    Example:
        >>> logger = Logger(
        ...     log_dir="./logs",
        ...     experiment_name="my_experiment",
        ...     use_tensorboard=True,
        ...     use_wandb=False
        ... )
        >>> logger.log_scalar("train/loss", 0.5, step=100)
        >>> logger.log_figure("embeddings", fig, step=100)
    """
    
    def __init__(
        self,
        log_dir: str = "./logs",
        experiment_name: Optional[str] = None,
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        project_name: str = "drrl-framework",
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ):
        """
        Initialize the unified logger.
        
        Args:
            log_dir: Directory for logs.
            experiment_name: Name of the experiment. Auto-generated if None.
            use_tensorboard: Whether to use TensorBoard backend.
            use_wandb: Whether to use W&B backend.
            project_name: W&B project name (if using W&B).
            config: Configuration to log.
            tags: Tags for W&B run.
        """
        # Generate experiment name if not provided
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"drrl_{timestamp}"
        
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.backends: List[LoggerBackend] = []
        
        # Always add console backend
        self.console = ConsoleBackend(log_dir, experiment_name)
        self.backends.append(self.console)
        
        # Add TensorBoard backend
        if use_tensorboard:
            tb_backend = TensorBoardBackend(log_dir, experiment_name)
            self.backends.append(tb_backend)
        
        # Add W&B backend
        if use_wandb:
            wandb_backend = WandBBackend(project_name, experiment_name, config, tags)
            self.backends.append(wandb_backend)
        
        # Log hyperparameters
        if config is not None:
            self.log_hyperparams(config)
        
        self.info(f"Logger initialized: {experiment_name}")
        self.info(f"Log directory: {self.log_dir / experiment_name}")
    
    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a scalar value to all backends."""
        for backend in self.backends:
            backend.log_scalar(tag, value, step)
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int) -> None:
        """Log multiple scalar values to all backends."""
        for backend in self.backends:
            backend.log_scalars(main_tag, tag_scalar_dict, step)
    
    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = "") -> None:
        """
        Log multiple metrics.
        
        Args:
            metrics: Dictionary of metric names and values.
            step: Current step/epoch.
            prefix: Optional prefix for metric names.
        """
        for name, value in metrics.items():
            tag = f"{prefix}/{name}" if prefix else name
            self.log_scalar(tag, value, step)
    
    def log_image(self, tag: str, image: np.ndarray, step: int) -> None:
        """Log an image to all backends."""
        for backend in self.backends:
            backend.log_image(tag, image, step)
    
    def log_figure(self, tag: str, figure: Any, step: int, close: bool = True) -> None:
        """
        Log a matplotlib figure to all backends.
        
        Args:
            tag: Tag for the figure.
            figure: Matplotlib figure object.
            step: Current step/epoch.
            close: Whether to close the figure after logging.
        """
        import matplotlib.pyplot as plt
        
        for backend in self.backends:
            backend.log_figure(tag, figure, step)
        
        if close:
            plt.close(figure)
    
    def log_histogram(self, tag: str, values: np.ndarray, step: int) -> None:
        """Log a histogram of values to all backends."""
        for backend in self.backends:
            backend.log_histogram(tag, values, step)
    
    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters to all backends."""
        for backend in self.backends:
            backend.log_hyperparams(params)
    
    def info(self, message: str) -> None:
        """Log an info message."""
        self.console.info(message)
    
    def warning(self, message: str) -> None:
        """Log a warning message."""
        self.console.warning(message)
    
    def error(self, message: str) -> None:
        """Log an error message."""
        self.console.error(message)
    
    def close(self) -> None:
        """Close all logging backends."""
        for backend in self.backends:
            backend.close()


def get_logger() -> Optional[Logger]:
    """Get the global logger instance."""
    return _LOGGER_INSTANCE


def set_global_logger(logger: Logger) -> None:
    """Set the global logger instance."""
    global _LOGGER_INSTANCE
    _LOGGER_INSTANCE = logger
