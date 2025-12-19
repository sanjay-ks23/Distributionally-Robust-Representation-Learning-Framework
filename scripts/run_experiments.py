#!/usr/bin/env python
"""
Run all experiments for the DRRL Framework.

Trains ERM, SAM, and DRO models and generates comparison plots.

Examples:
    python scripts/run_experiments.py --all
    python scripts/run_experiments.py --methods erm sam
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(description='Run DRRL Experiments')
    
    parser.add_argument('--all', action='store_true',
                       help='Run all methods')
    parser.add_argument('--methods', type=str, nargs='+',
                       default=['erm', 'sam', 'dro'],
                       help='Methods to run')
    parser.add_argument('--dataset', type=str, default='synthetic')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42],
                       help='Random seeds to run')
    parser.add_argument('--output_dir', type=str, default='./experiments')
    
    return parser.parse_args()


def run_training(method: str, seed: int, args) -> Path:
    """Run a single training experiment."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{method}_{args.dataset}_seed{seed}"
    save_dir = Path(args.output_dir) / exp_name
    
    cmd = [
        sys.executable, 'scripts/train.py',
        '--method', method,
        '--dataset', args.dataset,
        '--epochs', str(args.epochs),
        '--seed', str(seed),
        '--save_dir', str(save_dir),
        '--exp_name', exp_name,
        '--use_tensorboard'
    ]
    
    print(f"\n{'='*60}")
    print(f"Running: {method.upper()} (seed={seed})")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    
    return save_dir


def main():
    args = parse_args()
    
    methods = args.methods if not args.all else ['erm', 'sam', 'dro']
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    for seed in args.seeds:
        for method in methods:
            save_dir = run_training(method, seed, args)
            results[(method, seed)] = save_dir
    
    print("\n" + "="*60)
    print("All experiments completed!")
    print("="*60)
    
    for (method, seed), save_dir in results.items():
        print(f"  {method.upper()} (seed={seed}): {save_dir}")
    
    # Generate comparison plots
    print("\nGenerating comparison plots...")
    
    plot_cmd = [
        sys.executable, 'scripts/generate_plots.py',
        '--logs', *[str(d) for d in results.values()],
        '--output', str(output_dir / 'plots'),
        '--compare'
    ]
    
    subprocess.run(plot_cmd, cwd=Path(__file__).parent.parent)
    
    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
