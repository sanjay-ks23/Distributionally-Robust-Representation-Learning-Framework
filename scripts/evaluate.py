#!/usr/bin/env python
"""
Evaluation script for the DRRL Framework.

Examples:
    python scripts/evaluate.py --checkpoint outputs/best_model.pt --dataset synthetic
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from utils.seed import set_seed
from utils.helpers import get_device, load_checkpoint
from data import create_synthetic_splits, create_dataloaders
from models import build_model
from eval import Evaluator, compute_confusion_matrix
from viz import (
    plot_confusion_matrix,
    plot_group_accuracies,
    plot_embeddings_by_group_and_class
)


def parse_args():
    parser = argparse.ArgumentParser(description='DRRL Evaluation')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='synthetic')
    parser.add_argument('--encoder', type=str, default='simple_cnn')
    parser.add_argument('--output_dir', type=str, default='./plots')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--seed', type=int, default=42)
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    set_seed(args.seed)
    device = get_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print("Loading dataset...")
    if args.dataset == 'synthetic':
        _, _, test_dataset = create_synthetic_splits(
            test_spurious_corr=0.5,
            seed=args.seed
        )
        n_groups = 4
        n_classes = 2
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")
    
    _, _, test_loader = create_dataloaders(
        test_dataset, test_dataset, test_dataset,
        batch_size=128
    )
    
    # Load model
    print("Loading model...")
    model = build_model({
        'encoder': args.encoder,
        'num_classes': n_classes
    })
    
    load_checkpoint(args.checkpoint, model, device=device)
    model.to(device)
    model.eval()
    
    # Evaluate
    print("Evaluating...")
    evaluator = Evaluator(model, device, n_groups, n_classes)
    results = evaluator.evaluate(test_loader, return_embeddings=True)
    
    print("\n" + "="*50)
    print("Evaluation Results")
    print("="*50)
    print(f"Overall Accuracy: {results['accuracy']:.4f}")
    print(f"Worst-Group Accuracy: {results['worst_group_accuracy']:.4f}")
    print(f"Robustness Gap: {results['robustness_gap']:.4f}")
    
    for g in range(n_groups):
        key = f'group_{g}_accuracy'
        if key in results:
            print(f"  Group {g} Accuracy: {results[key]:.4f}")
    
    # Generate plots
    print("\nGenerating visualizations...")
    
    # Confusion matrix
    preds, _, targets, groups = evaluator.get_predictions(test_loader)
    cm = compute_confusion_matrix(preds, targets, n_classes)
    plot_confusion_matrix(
        cm,
        class_names=['Class 0', 'Class 1'],
        save_path=str(output_dir / 'confusion_matrix')
    )
    
    # Group accuracies
    group_accs = {g: results[f'group_{g}_accuracy'] for g in range(n_groups)}
    plot_group_accuracies(
        group_accs,
        save_path=str(output_dir / 'group_accuracies')
    )
    
    # Embeddings
    if 'embeddings' in results:
        plot_embeddings_by_group_and_class(
            results['embeddings'],
            results['targets'],
            results['groups'],
            save_path=str(output_dir / 'embeddings')
        )
    
    print(f"\nPlots saved to {output_dir}")


if __name__ == '__main__':
    main()
