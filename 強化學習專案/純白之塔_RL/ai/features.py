"""
Feature extraction for the RL agent.
Converts raw game state into normalized feature vectors.
"""

from typing import Optional
import numpy as np

from game.world import GameWorld
from game.entity import Entity


class FeatureExtractor:
    """
    Extracts observation features from the game world.
    Produces a fixed-size feature vector for the agent.
    """

    def __init__(self, world_size: float = 10.0):
        """
        Initialize the feature extractor.

        Args:
            world_size: Size of the game world for normalization
        """
        self.world_size = world_size
        self.n_features = 15

    def extract(self, world: GameWorld) -> np.ndarray:
        """
        Extract features from the current world state.

        Args:
            world: The game world

        Returns:
            Feature vector of size n_features
        """
        if world.player is None:
            return np.zeros(self.n_features)

        player = world.player
        player_pos = world.get_player_position()
        player_angle = world.get_player_angle()
        monster_pos = world.get_monster_position()
        blood_pos = world.get_blood_pack_position()

        # Relative positions (normalized)
        rel_monster = (monster_pos - player_pos) / self.world_size
        rel_blood = (blood_pos - player_pos) / self.world_size

        monster_dx, monster_dy = rel_monster[0], rel_monster[1]
        blood_dx, blood_dy = rel_blood[0], rel_blood[1]

        # Distances (normalized)
        dist_to_monster = np.linalg.norm(monster_pos - player_pos) / self.world_size
        dist_to_blood = np.linalg.norm(blood_pos - player_pos) / self.world_size

        # Absolute angles to targets
        angle_to_monster = np.arctan2(
            monster_pos[1] - player_pos[1],
            monster_pos[0] - player_pos[0]
        )
        angle_to_blood = np.arctan2(
            blood_pos[1] - player_pos[1],
            blood_pos[0] - player_pos[0]
        )

        # Relative angles (from player's facing direction)
        relative_angle_monster = np.arctan2(
            np.sin(angle_to_monster - player_angle),
            np.cos(angle_to_monster - player_angle)
        )
        relative_angle_blood = np.arctan2(
            np.sin(angle_to_blood - player_angle),
            np.cos(angle_to_blood - player_angle)
        )

        # Monster in sight indicator
        monster_in_sight = 1.0 if (
            abs(relative_angle_monster) < 0.5 and dist_to_monster < 0.6
        ) else 0.0

        # Distance to nearest wall (normalized)
        dist_to_wall = min(
            player_pos[0],
            player_pos[1],
            self.world_size - player_pos[0],
            self.world_size - player_pos[1]
        ) / self.world_size

        # Player facing direction
        cos_angle = np.cos(player_angle)
        sin_angle = np.sin(player_angle)

        # Casting state
        casting_progress = world.get_casting_progress()
        is_ready_to_cast = 1.0 if world.is_player_ready_to_cast() else 0.0

        # Combine all features
        return np.array([
            monster_dx, monster_dy,         # 0-1: Monster relative position
            blood_dx, blood_dy,             # 2-3: Blood pack relative position
            dist_to_monster, dist_to_blood, # 4-5: Distances
            relative_angle_monster,         # 6: Angle to monster
            relative_angle_blood,           # 7: Angle to blood
            cos_angle, sin_angle,           # 8-9: Player facing direction
            monster_in_sight,               # 10: Target in range indicator
            dist_to_wall,                   # 11: Wall proximity
            casting_progress,               # 12: Current cast progress
            is_ready_to_cast,               # 13: Ready to cast indicator
            1.0                             # 14: Bias term
        ])

    def extract_from_raw(
        self,
        agent_pos: np.ndarray,
        agent_angle: float,
        monster_pos: np.ndarray,
        blood_pos: np.ndarray,
        wind_up: int
    ) -> np.ndarray:
        """
        Extract features from raw position data.
        Useful for environments without the full GameWorld.

        Args:
            agent_pos: Player position [x, y]
            agent_angle: Player facing angle
            monster_pos: Monster position [x, y]
            blood_pos: Blood pack position [x, y]
            wind_up: Current wind-up ticks remaining

        Returns:
            Feature vector
        """
        # Relative positions
        rel_monster = (monster_pos - agent_pos) / self.world_size
        rel_blood = (blood_pos - agent_pos) / self.world_size

        monster_dx, monster_dy = rel_monster[0], rel_monster[1]
        blood_dx, blood_dy = rel_blood[0], rel_blood[1]

        # Distances
        dist_to_monster = np.linalg.norm(monster_pos - agent_pos) / self.world_size
        dist_to_blood = np.linalg.norm(blood_pos - agent_pos) / self.world_size

        # Angles
        angle_to_monster = np.arctan2(
            monster_pos[1] - agent_pos[1],
            monster_pos[0] - agent_pos[0]
        )
        angle_to_blood = np.arctan2(
            blood_pos[1] - agent_pos[1],
            blood_pos[0] - agent_pos[0]
        )

        relative_angle_monster = np.arctan2(
            np.sin(angle_to_monster - agent_angle),
            np.cos(angle_to_monster - agent_angle)
        )
        relative_angle_blood = np.arctan2(
            np.sin(angle_to_blood - agent_angle),
            np.cos(angle_to_blood - agent_angle)
        )

        # Derived features
        monster_in_sight = 1.0 if (
            abs(relative_angle_monster) < 0.5 and dist_to_monster < 0.6
        ) else 0.0

        dist_to_wall = min(
            agent_pos[0], agent_pos[1],
            self.world_size - agent_pos[0],
            self.world_size - agent_pos[1]
        ) / self.world_size

        cos_angle = np.cos(agent_angle)
        sin_angle = np.sin(agent_angle)

        casting_progress = wind_up / 4.0
        is_ready_to_cast = 1.0 if wind_up == 0 else 0.0

        return np.array([
            monster_dx, monster_dy,
            blood_dx, blood_dy,
            dist_to_monster, dist_to_blood,
            relative_angle_monster, relative_angle_blood,
            cos_angle, sin_angle,
            monster_in_sight,
            dist_to_wall,
            casting_progress, is_ready_to_cast,
            1.0
        ])
