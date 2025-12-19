#!/usr/bin/env python
"""
Main training script for the DRRL Framework.

Examples:
    # Train with ERM
    python scripts/train.py --method erm --dataset synthetic
    
    # Train with SAM
    python scripts/train.py --method sam --config configs/sam.yaml
    
    # Train with DRO
    python scripts/train.py --method dro --config configs/dro.yaml
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from utils.config import load_config, DRRLConfig
from utils.seed import set_seed
from utils.helpers import get_device
from utils.logging_utils import Logger
from data import create_synthetic_splits, get_train_transform, get_eval_transform
from data import create_dataloaders
from models import build_model
from train import get_trainer, get_optimizer, get_scheduler


def parse_args():
    parser = argparse.ArgumentParser(description='DRRL Training')
    
    # Config
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file')
    
    # Method
    parser.add_argument('--method', type=str, default='erm',
                       choices=['erm', 'sam', 'dro'],
                       help='Training method')
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='synthetic',
                       help='Dataset name')
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Data directory')
    
    # Model
    parser.add_argument('--encoder', type=str, default='simple_cnn',
                       help='Encoder architecture')
    parser.add_argument('--pretrained', action='store_true',
                       help='Use pretrained weights')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    
    # SAM specific
    parser.add_argument('--sam_rho', type=float, default=0.05,
                       help='SAM perturbation radius')
    
    # DRO specific
    parser.add_argument('--dro_step_size', type=float, default=0.01,
                       help='DRO group weight step size')
    
    # Environment
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (auto, cuda, cpu)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Output
    parser.add_argument('--save_dir', type=str, default='./outputs',
                       help='Save directory')
    parser.add_argument('--exp_name', type=str, default=None,
                       help='Experiment name')
    
    # Logging
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases')
    parser.add_argument('--use_tensorboard', action='store_true', default=True,
                       help='Use TensorBoard')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load config if provided
    if args.config:
        config = load_config(args.config)
    else:
        config = DRRLConfig()
    
    # Override with CLI args
    config.seed = args.seed
    config.device = args.device
    config.train.method = args.method
    config.train.epochs = args.epochs
    config.train.learning_rate = args.lr
    config.data.batch_size = args.batch_size
    
    # Set seed
    set_seed(config.seed)
    
    # Device
    device = get_device(config.device)
    print(f"Using device: {device}")
    
    # Experiment name
    exp_name = args.exp_name or f"{args.method}_{args.dataset}_{args.seed}"
    
    # Logger
    logger = Logger(
        log_dir='./logs',
        experiment_name=exp_name,
        use_tensorboard=args.use_tensorboard,
        use_wandb=args.use_wandb,
        config=config.to_dict()
    )
    
    # Create dataset
    print("Loading dataset...")
    if args.dataset == 'synthetic':
        train_dataset, val_dataset, test_dataset = create_synthetic_splits(
            n_train=8000,
            n_val=1000,
            n_test=1000,
            train_spurious_corr=0.9,
            test_spurious_corr=0.5,
            seed=config.seed
        )
        n_groups = 4
        n_classes = 2
        image_size = 64
    else:
        raise ValueError(f"Dataset {args.dataset} not supported in CLI")
    
    # DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset, val_dataset, test_dataset,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers
    )
    
    # Model
    print("Building model...")
    model = build_model({
        'encoder': args.encoder,
        'num_classes': n_classes,
        'pretrained': args.pretrained
    })
    model.to(device)
    
    # Optimizer
    optimizer = get_optimizer(
        'sgd',
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay
    )
    
    # Scheduler
    scheduler = get_scheduler(
        'cosine',
        optimizer,
        total_epochs=config.train.epochs,
        warmup_epochs=config.train.warmup_epochs
    )
    
    # Build trainer
    print(f"Training with {args.method.upper()}...")
    
    trainer_kwargs = {
        'model': model,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'device': device,
        'logger': logger,
        'save_dir': args.save_dir,
        'epochs': config.train.epochs
    }
    
    if args.method == 'sam':
        from train.sam_trainer import SAMTrainer
        trainer = SAMTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            base_optimizer=optimizer,
            rho=args.sam_rho,
            scheduler=scheduler,
            device=device,
            logger=logger,
            save_dir=args.save_dir,
            epochs=config.train.epochs
        )
    elif args.method == 'dro':
        from train.dro_trainer import DROTrainer
        trainer = DROTrainer(
            n_groups=n_groups,
            dro_step_size=args.dro_step_size,
            **trainer_kwargs
        )
    else:
        from train.erm_trainer import ERMTrainer
        trainer = ERMTrainer(**trainer_kwargs)
    
    # Train
    final_metrics = trainer.train()
    
    # Print results
    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)
    print(f"Final Validation Accuracy: {final_metrics.get('accuracy', 0):.4f}")
    print(f"Worst-Group Accuracy: {final_metrics.get('worst_group_accuracy', 0):.4f}")
    
    logger.close()


if __name__ == '__main__':
    main()
