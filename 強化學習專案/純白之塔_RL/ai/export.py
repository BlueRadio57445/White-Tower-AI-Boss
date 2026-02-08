"""
Weight export utilities for deploying trained models.
Supports JSON (for Minecraft) and NumPy formats.
"""

import json
from typing import Dict, Any, Optional
from pathlib import Path
import numpy as np

from ai.agent import HybridPPOAgent


class WeightExporter:
    """
    Exports agent weights to various formats for deployment.
    """

    @staticmethod
    def to_json(agent: HybridPPOAgent, path: str, include_metadata: bool = True) -> str:
        """
        Export agent weights to JSON format.

        Args:
            agent: The trained agent
            path: Output file path
            include_metadata: Whether to include model metadata

        Returns:
            The path to the saved file
        """
        weights = agent.get_weights()

        export_data: Dict[str, Any] = {
            'weights': {
                'actor_discrete': weights['w_actor_discrete'].tolist(),
                'aim_actors': [w.tolist() for w in weights['w_aim_actors']],
                'critic': weights['w_critic'].tolist()
            },
            'parameters': {
                'sigma': weights['sigma'],
                'n_features': agent.n_features,
                'n_discrete_actions': agent.n_discrete_actions,
                'n_aim_actors': weights.get('n_aim_actors', 2)
            }
        }

        if include_metadata:
            export_data['metadata'] = {
                'format_version': '2.0',
                'model_type': 'HybridPPOAgent',
                'probability_type': 'squared',
                'description': 'Hardware-friendly PPO with squared probability distribution',
                'skill_system': 'Multi-skill with ring, projectile, and rectangle shapes'
            }

        # Ensure directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        return path

    @staticmethod
    def to_numpy(agent: HybridPPOAgent, path: str) -> str:
        """
        Export agent weights to NumPy .npz format.

        Args:
            agent: The trained agent
            path: Output file path (should end with .npz)

        Returns:
            The path to the saved file
        """
        weights = agent.get_weights()

        # Ensure directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        # Stack aim actors into a single array for easier storage
        aim_actors_stacked = np.stack(weights['w_aim_actors'])

        np.savez(
            path,
            w_actor_discrete=weights['w_actor_discrete'],
            w_aim_actors=aim_actors_stacked,
            w_critic=weights['w_critic'],
            sigma=np.array([weights['sigma']]),
            n_features=np.array([agent.n_features]),
            n_discrete_actions=np.array([agent.n_discrete_actions]),
            n_aim_actors=np.array([weights.get('n_aim_actors', 2)])
        )

        return path

    @staticmethod
    def from_json(path: str) -> Dict[str, Any]:
        """
        Load agent weights from JSON format.

        Args:
            path: Path to the JSON file

        Returns:
            Dictionary containing weights and parameters
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        weights = data['weights']
        params = data['parameters']

        result = {
            'w_actor_discrete': np.array(weights['actor_discrete']),
            'w_critic': np.array(weights['critic']),
            'sigma': params['sigma'],
            'n_features': params['n_features'],
            'n_discrete_actions': params['n_discrete_actions']
        }

        # Handle new format with aim_actors
        if 'aim_actors' in weights:
            result['w_aim_actors'] = [np.array(w) for w in weights['aim_actors']]
            result['n_aim_actors'] = params.get('n_aim_actors', len(weights['aim_actors']))
        # Handle legacy format with actor_continuous_mu
        elif 'actor_continuous_mu' in weights:
            result['w_actor_continuous_mu'] = np.array(weights['actor_continuous_mu'])

        return result

    @staticmethod
    def from_numpy(path: str) -> Dict[str, Any]:
        """
        Load agent weights from NumPy .npz format.

        Args:
            path: Path to the .npz file

        Returns:
            Dictionary containing weights and parameters
        """
        data = np.load(path)

        result = {
            'w_actor_discrete': data['w_actor_discrete'],
            'w_critic': data['w_critic'],
            'sigma': float(data['sigma'][0]),
            'n_features': int(data['n_features'][0]),
            'n_discrete_actions': int(data['n_discrete_actions'][0])
        }

        # Handle new format with aim_actors
        if 'w_aim_actors' in data:
            result['w_aim_actors'] = [data['w_aim_actors'][i] for i in range(len(data['w_aim_actors']))]
            result['n_aim_actors'] = int(data['n_aim_actors'][0]) if 'n_aim_actors' in data else len(result['w_aim_actors'])
        # Handle legacy format with actor_continuous_mu
        elif 'w_actor_continuous_mu' in data:
            result['w_actor_continuous_mu'] = data['w_actor_continuous_mu']

        return result

    @staticmethod
    def load_into_agent(agent: HybridPPOAgent, path: str) -> None:
        """
        Load weights from file into an existing agent.

        Args:
            agent: The agent to load weights into
            path: Path to the weights file (JSON or NPZ)
        """
        if path.endswith('.json'):
            weights = WeightExporter.from_json(path)
        elif path.endswith('.npz'):
            weights = WeightExporter.from_numpy(path)
        else:
            raise ValueError(f"Unsupported file format: {path}")

        agent.set_weights(weights)

    @staticmethod
    def create_agent_from_file(path: str) -> HybridPPOAgent:
        """
        Create a new agent and load weights from file.

        Args:
            path: Path to the weights file

        Returns:
            A new HybridPPOAgent with loaded weights
        """
        if path.endswith('.json'):
            weights = WeightExporter.from_json(path)
        elif path.endswith('.npz'):
            weights = WeightExporter.from_numpy(path)
        else:
            raise ValueError(f"Unsupported file format: {path}")

        n_aim_actors = weights.get('n_aim_actors', 2)

        agent = HybridPPOAgent(
            n_features=weights['n_features'],
            n_discrete_actions=weights['n_discrete_actions'],
            n_aim_actors=n_aim_actors
        )
        agent.set_weights(weights)

        return agent
