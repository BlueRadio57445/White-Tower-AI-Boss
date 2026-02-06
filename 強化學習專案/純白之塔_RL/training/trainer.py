"""
Training loop for the RL agent.
Coordinates environment, agent, and logging.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Callable
import numpy as np
import os
import time

from core.events import EventBus, EventType
from game.world import GameWorld, Room
from ai.agent import HybridPPOAgent
from ai.features import FeatureExtractor
from ai.reward import RewardCalculator
from ai.export import WeightExporter


@dataclass
class TrainingConfig:
    """Configuration for training."""
    epochs: int = 12000
    steps_per_epoch: int = 50
    render_last_n: int = 10
    world_size: float = 10.0
    use_pygame: bool = True  # Use Pygame renderer (False for ASCII)

    # Agent hyperparameters
    n_features: int = 26
    n_discrete_actions: int = 4
    gamma: float = 0.99
    lmbda: float = 0.95
    epsilon: float = 0.2
    sigma_init: float = 0.6
    sigma_min: float = 0.15
    sigma_decay: float = 0.9998

    # Learning rates
    lr_actor_discrete: float = 0.003
    lr_actor_continuous: float = 0.002
    lr_critic: float = 0.007

    # Export settings
    export_weights: bool = True
    export_path: str = "weights.json"


class Trainer:
    """
    Manages the training loop for the RL agent.
    """

    def __init__(self, config: Optional[TrainingConfig] = None):
        """
        Initialize the trainer.

        Args:
            config: Training configuration
        """
        self.config = config or TrainingConfig()
        self.history: List[float] = []
        self.renderer = None

        # Initialize Pygame renderer if enabled
        if self.config.use_pygame:
            try:
                from rendering import PygameRenderer
                self.renderer = PygameRenderer(world_size=self.config.world_size)
            except ImportError:
                print("Warning: pygame not available, falling back to ASCII rendering")
                self.renderer = None

        # Initialize components
        self.world = GameWorld(Room(size=self.config.world_size))
        self.agent = HybridPPOAgent(
            n_features=self.config.n_features,
            n_discrete_actions=self.config.n_discrete_actions,
            gamma=self.config.gamma,
            lmbda=self.config.lmbda,
            epsilon=self.config.epsilon,
            sigma_init=self.config.sigma_init,
            sigma_min=self.config.sigma_min,
            sigma_decay=self.config.sigma_decay
        )
        self.feature_extractor = FeatureExtractor(self.config.world_size)
        self.reward_calculator = RewardCalculator(self.world.event_bus)

    def train(
        self,
        progress_callback: Optional[Callable[[int, float], None]] = None,
        render: bool = True
    ) -> List[float]:
        """
        Run the full training loop.

        Args:
            progress_callback: Called after each epoch with (epoch, reward)
            render: Whether to render the last few epochs

        Returns:
            List of episode rewards
        """
        action_names = ["MOVE", "LEFT", "RIGHT", "CAST"]

        print("Starting 2D Training... Squared Probability PPO Ready.")
        print(f"Discrete actions: {', '.join(action_names)}")
        print("Continuous action: Aim offset during casting")
        print(f"Features: {self.config.n_features} dimensions")
        print("Advantage: No dead neurons, only uses +, -, *, /")
        print()

        try:
            from tqdm import tqdm
            epoch_iterator = tqdm(range(self.config.epochs))
        except ImportError:
            epoch_iterator = range(self.config.epochs)
            print(f"Training for {self.config.epochs} epochs...")

        for epoch in epoch_iterator:
            total_reward = self._run_episode(
                epoch,
                render and epoch >= self.config.epochs - self.config.render_last_n,
                action_names
            )

            self.history.append(total_reward)

            if progress_callback:
                progress_callback(epoch, total_reward)

        # Export weights if configured
        if self.config.export_weights:
            WeightExporter.to_json(self.agent, self.config.export_path)
            print(f"\nWeights exported to: {self.config.export_path}")

        # Close renderer
        if self.renderer:
            self.renderer.close()

        return self.history

    def _run_episode(
        self,
        epoch: int,
        render: bool,
        action_names: List[str]
    ) -> float:
        """Run a single training episode."""
        self.world.reset()
        self.reward_calculator.reset()
        total_reward = 0.0

        for step in range(self.config.steps_per_epoch):
            # Get observation
            obs = self.feature_extractor.extract(self.world)

            # Get action from agent
            a_d, a_c, prob_d, mu, v, logits = self.agent.get_action(obs)

            # Execute action
            action_event = self.world.execute_action(a_d, a_c)

            # Process game tick
            self.world.tick()

            # Get reward from events
            reward = self.reward_calculator.get_reward()
            event = self.reward_calculator.get_last_event() or action_event

            # Store transition
            self.agent.store_transition((obs, a_d, a_c, prob_d, mu, v, logits, reward))
            total_reward += reward

            # Render if requested
            if render:
                self._render(epoch, total_reward, a_d, a_c, event, action_names)

        # Update agent
        self.agent.update(
            lr_actor_discrete=self.config.lr_actor_discrete,
            lr_actor_continuous=self.config.lr_actor_continuous,
            lr_critic=self.config.lr_critic
        )

        return total_reward

    def _render(
        self,
        epoch: int,
        total_reward: float,
        action: int,
        continuous: float,
        event: str,
        action_names: List[str]
    ) -> bool:
        """
        Render the current game state.

        Returns:
            True to continue, False to stop (window closed)
        """
        if self.renderer:
            return self.renderer.render(
                self.world,
                epoch,
                total_reward,
                self.agent.sigma,
                action,
                continuous,
                event,
                action_names
            )
        else:
            self._render_ascii(epoch, total_reward, action, continuous, event, action_names)
            return True

    def _render_ascii(
        self,
        epoch: int,
        total_reward: float,
        action: int,
        continuous: float,
        event: str,
        action_names: List[str]
    ) -> None:
        """Render the current game state using ASCII."""
        os.system('cls' if os.name == 'nt' else 'clear')

        print(f"Ep: {epoch} | Reward: {total_reward:.1f} | Sigma: {self.agent.sigma:.3f}")
        print(f"Action: {action_names[action]} | Continuous: {continuous:.2f} | Event: {event}")

        # Create canvas
        canvas = np.full((10, 10), ".")

        monster_pos = self.world.get_monster_position()
        blood_pos = self.world.get_blood_pack_position()
        player_pos = self.world.get_player_position()

        # Place entities (clamp to valid indices)
        my, mx = int(np.clip(monster_pos[1], 0, 9)), int(np.clip(monster_pos[0], 0, 9))
        by, bx = int(np.clip(blood_pos[1], 0, 9)), int(np.clip(blood_pos[0], 0, 9))
        py, px = int(np.clip(player_pos[1], 0, 9)), int(np.clip(player_pos[0], 0, 9))

        canvas[my, mx] = "M"
        canvas[by, bx] = "B"
        canvas[py, px] = "A"

        # Draw map with walls
        print("  " + " ".join(["#"] * 12))
        for row in canvas:
            print("  # " + " ".join(row) + " #")
        print("  " + " ".join(["#"] * 12))

        time.sleep(0.05)

    def get_final_stats(self) -> dict:
        """Get training statistics."""
        if not self.history:
            return {}

        return {
            'total_epochs': len(self.history),
            'final_100_avg': np.mean(self.history[-100:]) if len(self.history) >= 100 else np.mean(self.history),
            'max_reward': max(self.history),
            'min_reward': min(self.history),
            'final_sigma': self.agent.sigma
        }

    def plot_training(self, show: bool = True, save_path: Optional[str] = None) -> None:
        """Plot training curve."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available for plotting")
            return

        plt.figure(figsize=(12, 6))
        plt.plot(self.history, color='royalblue', alpha=0.3, label="Raw Episode Reward")

        if len(self.history) >= 50:
            moving_avg = np.convolve(self.history, np.ones(50)/50, mode='valid')
            plt.plot(range(len(moving_avg)), moving_avg, color='crimson',
                    linewidth=2, label="50-Ep Rolling Average")

        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.title("RPG Agent: Squared Probability PPO + GAE Training Report")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Plot saved to: {save_path}")

        if show:
            plt.show()
