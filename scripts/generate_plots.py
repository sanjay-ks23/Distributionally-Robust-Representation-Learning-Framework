#!/usr/bin/env python
"""
Generate plots from training logs.

Examples:
    python scripts/generate_plots.py --logs logs/experiment_1 --output plots/
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from viz import (
    plot_training_curves,
    plot_method_comparison,
    plot_worst_group_comparison,
    plot_worst_vs_average,
    plot_robustness_gap
)


def parse_args():
    parser = argparse.ArgumentParser(description='Generate DRRL Plots')
    
    parser.add_argument('--logs', type=str, nargs='+',
                       help='Log directories to plot')
    parser.add_argument('--output', type=str, default='./plots',
                       help='Output directory')
    parser.add_argument('--compare', action='store_true',
                       help='Compare multiple experiments')
    
    return parser.parse_args()


def load_training_history(log_dir: Path) -> dict:
    """Load training history from log directory."""
    history_path = log_dir / 'history.json'
    
    if history_path.exists():
        with open(history_path) as f:
            return json.load(f)
    
    # Try to reconstruct from TensorBoard logs
    return {}


def main():
    args = parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not args.logs:
        print("No log directories specified. Use --logs to specify.")
        return
    
    if args.compare and len(args.logs) > 1:
        # Compare multiple experiments
        histories = {}
        for log_path in args.logs:
            log_dir = Path(log_path)
            name = log_dir.name
            histories[name] = load_training_history(log_dir)
        
        if histories:
            plot_method_comparison(
                histories,
                metric='val_accuracy',
                save_path=str(output_dir / 'method_comparison')
            )
            
            plot_worst_group_comparison(
                histories,
                save_path=str(output_dir / 'worst_group_comparison')
            )
            
            print(f"Comparison plots saved to {output_dir}")
    else:
        # Single experiment
        log_dir = Path(args.logs[0])
        history = load_training_history(log_dir)
        
        if history:
            plot_training_curves(
                history,
                save_path=str(output_dir / 'training_curves')
            )
            print(f"Training curves saved to {output_dir}")
        else:
            print(f"No training history found in {log_dir}")


if __name__ == '__main__':
    main()
