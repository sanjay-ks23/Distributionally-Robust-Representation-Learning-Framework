"""
Configuration utilities for the DRRL Framework.

Provides dataclasses for type-safe configuration and utilities for
loading, validating, and merging configurations.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
import yaml
import json
from enum import Enum


class TrainingMethod(Enum):
    """Enumeration of supported training methods."""
    ERM = "erm"
    SAM = "sam"
    DRO = "dro"


class DatasetType(Enum):
    """Enumeration of supported dataset types."""
    WATERBIRDS = "waterbirds"
    CIFAR10C = "cifar10c"
    SYNTHETIC = "synthetic"


class EncoderType(Enum):
    """Enumeration of supported encoder architectures."""
    RESNET18 = "resnet18"
    RESNET50 = "resnet50"
    SIMPLE_CNN = "simple_cnn"


@dataclass
class DataConfig:
    """Configuration for dataset and data loading."""
    
    dataset: str = "synthetic"
    data_dir: str = "./data"
    batch_size: int = 64
    num_workers: int = 4
    pin_memory: bool = True
    
    # Distribution shift parameters
    spurious_correlation: float = 0.9
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    
    # Augmentation
    use_augmentation: bool = True
    image_size: int = 224
    
    # Group information
    n_groups: int = 4
    group_balanced: bool = False
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        assert 0 <= self.spurious_correlation <= 1.0
        assert self.train_ratio + self.val_ratio < 1.0
        assert self.batch_size > 0
        assert self.n_groups > 0


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    
    encoder: str = "resnet18"
    pretrained: bool = True
    freeze_encoder: bool = False
    
    # Classifier configuration
    classifier_type: str = "linear"
    hidden_dims: List[int] = field(default_factory=lambda: [256])
    dropout: float = 0.5
    
    # Number of classes
    num_classes: int = 2
    
    # Embedding dimension (set automatically based on encoder)
    embedding_dim: Optional[int] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        assert 0 <= self.dropout < 1.0
        assert self.num_classes > 1


@dataclass
class TrainConfig:
    """Configuration for training procedure."""
    
    method: str = "erm"
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    momentum: float = 0.9
    
    # Optimizer
    optimizer: str = "sgd"
    
    # Learning rate scheduling
    scheduler: str = "cosine"
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 10
    
    # Gradient clipping
    gradient_clip: Optional[float] = 1.0
    
    # SAM specific
    sam_rho: float = 0.05
    sam_adaptive: bool = False
    
    # DRO specific
    dro_step_size: float = 0.01
    dro_robust_step_size: float = 0.01
    dro_gamma: float = 0.1
    dro_normalize: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        assert self.epochs > 0
        assert self.learning_rate > 0
        assert self.patience > 0


@dataclass  
class LoggingConfig:
    """Configuration for logging and experiment tracking."""
    
    use_wandb: bool = False
    use_tensorboard: bool = True
    
    project_name: str = "drrl-framework"
    experiment_name: Optional[str] = None
    
    log_dir: str = "./logs"
    save_dir: str = "./outputs"
    plot_dir: str = "./plots"
    
    log_interval: int = 10
    save_interval: int = 5
    
    # Visualization settings
    save_embeddings: bool = True
    embedding_interval: int = 10
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        for dir_path in [self.log_dir, self.save_dir, self.plot_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)


@dataclass
class EvalConfig:
    """Configuration for evaluation."""
    
    eval_batch_size: int = 128
    compute_confusion: bool = True
    compute_embeddings: bool = True
    
    # OOD evaluation
    ood_datasets: List[str] = field(default_factory=list)
    
    # Metrics
    metrics: List[str] = field(default_factory=lambda: [
        "accuracy", "worst_group_accuracy", "per_group_accuracy"
    ])


@dataclass
class DRRLConfig:
    """
    Master configuration class for the DRRL Framework.
    
    Combines all sub-configurations into a single, unified config object.
    Supports loading from YAML files and command-line overrides.
    """
    
    # Sub-configurations
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    
    # Global settings
    seed: int = 42
    device: str = "auto"
    mixed_precision: bool = False
    deterministic: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        from dataclasses import asdict
        return asdict(self)
    
    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DRRLConfig':
        """Create configuration from dictionary."""
        data_config = DataConfig(**config_dict.get('data', {}))
        model_config = ModelConfig(**config_dict.get('model', {}))
        train_config = TrainConfig(**config_dict.get('train', {}))
        logging_config = LoggingConfig(**config_dict.get('logging', {}))
        eval_config = EvalConfig(**config_dict.get('eval', {}))
        
        return cls(
            data=data_config,
            model=model_config,
            train=train_config,
            logging=logging_config,
            eval=eval_config,
            seed=config_dict.get('seed', 42),
            device=config_dict.get('device', 'auto'),
            mixed_precision=config_dict.get('mixed_precision', False),
            deterministic=config_dict.get('deterministic', True)
        )


def load_config(path: Union[str, Path]) -> DRRLConfig:
    """
    Load configuration from a YAML or JSON file.
    
    Args:
        path: Path to the configuration file.
        
    Returns:
        DRRLConfig object with loaded settings.
        
    Raises:
        FileNotFoundError: If the configuration file doesn't exist.
        ValueError: If the file format is not supported.
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    
    suffix = path.suffix.lower()
    
    with open(path, 'r') as f:
        if suffix in ['.yaml', '.yml']:
            config_dict = yaml.safe_load(f)
        elif suffix == '.json':
            config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {suffix}")
    
    return DRRLConfig.from_dict(config_dict)


def validate_config(config: DRRLConfig) -> bool:
    """
    Validate the configuration for consistency and correctness.
    
    Args:
        config: DRRLConfig object to validate.
        
    Returns:
        True if configuration is valid.
        
    Raises:
        ValueError: If configuration is invalid.
    """
    # Validate training method
    valid_methods = [m.value for m in TrainingMethod]
    if config.train.method not in valid_methods:
        raise ValueError(f"Invalid training method: {config.train.method}. "
                        f"Must be one of {valid_methods}")
    
    # Validate dataset
    valid_datasets = [d.value for d in DatasetType]
    if config.data.dataset not in valid_datasets:
        raise ValueError(f"Invalid dataset: {config.data.dataset}. "
                        f"Must be one of {valid_datasets}")
    
    # Validate encoder
    valid_encoders = [e.value for e in EncoderType]
    if config.model.encoder not in valid_encoders:
        raise ValueError(f"Invalid encoder: {config.model.encoder}. "
                        f"Must be one of {valid_encoders}")
    
    # DRO requires group information
    if config.train.method == "dro" and config.data.n_groups < 2:
        raise ValueError("DRO training requires at least 2 groups")
    
    return True


def merge_configs(base: DRRLConfig, override: Dict[str, Any]) -> DRRLConfig:
    """
    Merge override settings into a base configuration.
    
    Args:
        base: Base configuration object.
        override: Dictionary of override values.
        
    Returns:
        New DRRLConfig with merged settings.
    """
    base_dict = base.to_dict()
    
    def deep_merge(d1: Dict, d2: Dict) -> Dict:
        """Recursively merge two dictionaries."""
        result = d1.copy()
        for key, value in d2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    merged_dict = deep_merge(base_dict, override)
    return DRRLConfig.from_dict(merged_dict)
