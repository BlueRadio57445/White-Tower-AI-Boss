"""
Main entry point for training the reinforcement learning agent.

This script trains a PPO agent in a 2D RPG environment using
squared probability distribution (hardware-friendly, no dead neurons).

Usage:
    python main.py                    # Full training with defaults
    python main.py --epochs 1000      # Quick training
    python main.py --no-render        # Training without visualization
    python main.py --export model.json  # Custom export path
"""

import argparse
import numpy as np

from training.trainer import Trainer, TrainingConfig
from ai.export import WeightExporter


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train a PPO agent in the 2D RPG environment'
    )

    parser.add_argument(
        '--epochs', type=int, default=12000,
        help='Number of training epochs (default: 12000)'
    )
    parser.add_argument(
        '--steps', type=int, default=50,
        help='Steps per epoch (default: 50)'
    )
    parser.add_argument(
        '--no-render', action='store_true',
        help='Disable rendering of final epochs'
    )
    parser.add_argument(
        '--export', type=str, default='weights.json',
        help='Path to export weights (default: weights.json)'
    )
    parser.add_argument(
        '--no-export', action='store_true',
        help='Disable weight export'
    )
    parser.add_argument(
        '--plot', type=str, default=None,
        help='Save training plot to file'
    )
    parser.add_argument(
        '--no-plot', action='store_true',
        help='Disable training plot display'
    )
    parser.add_argument(
        '--ascii', action='store_true',
        help='Use ASCII rendering instead of Pygame'
    )

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Configure training
    config = TrainingConfig(
        epochs=args.epochs,
        steps_per_epoch=args.steps,
        render_last_n=0 if args.no_render else 10,
        export_weights=not args.no_export,
        export_path=args.export,
        use_pygame=not args.ascii
    )

    # Create trainer and run
    trainer = Trainer(config)
    history = trainer.train(render=not args.no_render)

    # Print final statistics
    stats = trainer.get_final_stats()
    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)
    print(f"Total epochs: {stats['total_epochs']}")
    print(f"Final 100-epoch average reward: {stats['final_100_avg']:.2f}")
    print(f"Max reward: {stats['max_reward']:.2f}")
    print(f"Min reward: {stats['min_reward']:.2f}")
    print(f"Final exploration sigma: {stats['final_sigma']:.4f}")
    print("Using squared probability distribution - no dead neurons!")

    # Plot if not disabled
    if not args.no_plot:
        trainer.plot_training(show=True, save_path=args.plot)


if __name__ == '__main__':
    main()
