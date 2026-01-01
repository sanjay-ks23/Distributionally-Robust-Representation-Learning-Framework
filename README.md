# DRRL Framework

**Distributionally Robust Representation Learning Framework**

A modular, production-ready PyTorch framework for learning robust representations that generalize under distribution shift. Implements state-of-the-art robust training methods including ERM, SAM, and GroupDRO.

---

## Overview

Modern machine learning models often rely on **spurious correlations** present in training data, leading to poor performance on underrepresented groups or out-of-distribution data. This framework provides tools to:

- **Learn robust representations** that focus on core features rather than spurious correlations
- **Improve worst-group performance** using distributionally robust optimization
- **Visualize and analyze** model behavior, embedding geometry, and robustness metrics

### Supported Training Methods

| Method | Description | Key Benefit |
|--------|-------------|-------------|
| **ERM** | Empirical Risk Minimization | Baseline, minimizes average loss |
| **SAM** | Sharpness-Aware Minimization | Finds flat minima for better generalization |
| **DRO** | Group Distributionally Robust Optimization | Optimizes worst-group performance |

---

## Project Structure

```
drrl-framework/
├── configs/                 # YAML configuration files
│   ├── default.yaml
│   ├── erm.yaml
│   ├── sam.yaml
│   └── dro.yaml
├── data/                    # Dataset loaders and transforms
│   ├── base.py              # Base dataset classes
│   ├── synthetic.py         # Synthetic dataset with spurious correlations
│   ├── waterbirds.py        # Waterbirds benchmark
│   ├── cifar10c.py          # CIFAR-10-C corrupted dataset
│   ├── transforms.py        # Data augmentations
│   └── groups.py            # Group-aware sampling
├── models/                  # Model architectures
│   ├── encoders.py          # Feature extractors (ResNet, SimpleCNN)
│   ├── classifiers.py       # Classification heads
│   └── drrl_model.py        # Main model class
├── robust/                  # Robust training methods
│   ├── sam.py               # Sharpness-Aware Minimization
│   ├── dro.py               # Group DRO implementation
│   └── losses.py            # Robust loss functions
├── train/                   # Training pipeline
│   ├── base_trainer.py      # Abstract trainer class
│   ├── erm_trainer.py       # ERM training
│   ├── sam_trainer.py       # SAM training
│   ├── dro_trainer.py       # DRO training
│   └── schedulers.py        # LR schedulers
├── eval/                    # Evaluation utilities
│   ├── metrics.py           # Accuracy, per-group metrics
│   ├── evaluator.py         # Evaluation engine
│   └── confusion.py         # Confusion matrix utils
├── viz/                     # Visualization
│   ├── training_curves.py   # Loss/accuracy plots
│   ├── embeddings.py        # UMAP/t-SNE visualizations
│   ├── group_performance.py # Per-group analysis
│   └── distribution_shift.py# OOD performance plots
├── scripts/                 # Executable scripts
│   ├── train.py             # Main training script
│   ├── evaluate.py          # Evaluation script
│   └── run_experiments.py   # Batch experiments
├── tests/                   # Unit tests
├── utils/                   # Utilities
│   ├── config.py            # Configuration management
│   ├── logging_utils.py     # TensorBoard/W&B logging
│   └── seed.py              # Reproducibility
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/example/drrl-framework.git
cd drrl-framework

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Train a Model

```bash
# Train with ERM (baseline)
python scripts/train.py --method erm --dataset synthetic --epochs 50

# Train with SAM
python scripts/train.py --method sam --dataset synthetic --sam_rho 0.05

# Train with DRO
python scripts/train.py --method dro --dataset synthetic --dro_step_size 0.01

# Use a config file
python scripts/train.py --config configs/dro.yaml
```

### Evaluate a Model

```bash
python scripts/evaluate.py \
    --checkpoint outputs/best_model.pt \
    --dataset synthetic \
    --output_dir plots/
```

### Run All Experiments

```bash
# Compare all methods
python scripts/run_experiments.py --all --epochs 50
```

---

## Visualizations

The framework generates comprehensive visualizations for analysis:

### Training Curves

Track loss and accuracy across training:

```python
from viz import plot_training_curves

plot_training_curves(
    history={'train_loss': [...], 'val_loss': [...], 'train_accuracy': [...], 'val_accuracy': [...]},
    save_path='plots/training_curves'
)
```

### Embedding Visualizations

Visualize learned representations with UMAP/t-SNE:

```python
from viz import plot_embeddings_by_group_and_class

plot_embeddings_by_group_and_class(
    embeddings,           # (N, D) embedding array
    targets,              # Class labels
    groups,               # Group labels (spurious features)
    class_names=['Landbird', 'Waterbird'],
    group_names=['Majority', 'Minority', ...],
    save_path='plots/embeddings'
)
```

**Interpretation**: The left plot shows embeddings colored by true class labels. The right plot shows the same embeddings colored by spurious features. A robust model should separate classes regardless of spurious features.

### Group Performance Analysis

Compare per-group and worst-group accuracy:



**Interpretation**: ERM achieves highest average accuracy but lowest worst-group accuracy. DRO sacrifices some average performance for much better worst-group performance.

### Confusion Matrices

Analyze per-group error patterns:

```python
from viz import plot_per_group_confusion

plot_per_group_confusion(
    group_cms={0: cm0, 1: cm1, 2: cm2, 3: cm3},
    group_names=['Majority A', 'Minority A', 'Minority B', 'Majority B'],
    save_path='plots/group_confusion'
)
```

---

## Configuration

Use YAML configuration files for reproducible experiments:

```yaml
# configs/dro.yaml
seed: 42
device: auto

data:
  dataset: synthetic
  batch_size: 64
  spurious_correlation: 0.9
  n_groups: 4

model:
  encoder: simple_cnn
  num_classes: 2

train:
  method: dro
  epochs: 50
  learning_rate: 0.001
  dro_step_size: 0.01

logging:
  use_tensorboard: true
  save_dir: ./outputs/dro
```

### Key Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `train.method` | Training method (erm, sam, dro) | erm |
| `train.sam_rho` | SAM perturbation radius | 0.05 |
| `train.dro_step_size` | DRO weight update rate | 0.01 |
| `data.spurious_correlation` | Spurious correlation strength | 0.9 |
| `model.encoder` | Encoder architecture | simple_cnn |

---

## Expected Results

### Synthetic Dataset

| Method | Average Acc | Worst-Group Acc | Robustness Gap |
|--------|-------------|-----------------|----------------|
| ERM    | 92.5%       | 65.0%           | 27.5%          |
| SAM    | 91.0%       | 72.0%           | 19.0%          |
| DRO    | 88.0%       | 82.0%           | 6.0%           |

**Key Insight**: DRO significantly reduces the robustness gap (difference between average and worst-group accuracy) at a small cost to average accuracy.

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=. --cov-report=html

# Run specific test file
pytest tests/test_models.py -v
```

---

## Docker

```bash
# Build image
docker build -t drrl-framework .

# Run training
docker run --gpus all drrl-framework python scripts/train.py --method dro

# Run tests
docker run drrl-framework pytest tests/ -v
```

---

## References

- **SAM**: Foret et al. "Sharpness-Aware Minimization for Efficiently Improving Generalization" (ICLR 2021)
- **GroupDRO**: Sagawa et al. "Distributionally Robust Neural Networks for Group Shifts" (ICLR 2020)
- **Waterbirds**: Sagawa et al. "Distributionally Robust Neural Networks" (ICLR 2020)

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---
