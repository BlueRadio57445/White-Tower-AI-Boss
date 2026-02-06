"""
Feature extraction for the RL agent.
Converts raw game state into normalized feature vectors.
"""

from typing import List
import numpy as np

from game.world import GameWorld


class FeatureExtractor:
    """
    Extracts observation features from the game world.
    Produces a fixed-size feature vector for the agent.

    Feature layout (26 dimensions):
    [0-3]   Nearest monster: dx, dy, dist, relative_angle
    [4-7]   2nd nearest monster: dx, dy, dist, relative_angle
    [8-11]  3rd nearest monster: dx, dy, dist, relative_angle
    [12-15] 4th nearest monster: dx, dy, dist, relative_angle
    [16-17] Blood pack: dx, dy
    [18]    Blood pack distance
    [19]    Blood pack relative angle
    [20-21] Player facing direction: cos, sin
    [22]    Distance to wall (facing direction)
    [23]    Casting progress
    [24]    Ready to cast indicator
    [25]    Bias term

    Dead monsters have all features set to 0.
    """

    MAX_MONSTERS = 4
    FEATURES_PER_MONSTER = 4

    def __init__(self, world_size: float = 10.0):
        """
        Initialize the feature extractor.

        Args:
            world_size: Size of the game world for normalization
        """
        self.world_size = world_size
        self.n_features = 26

    def _raycast_to_wall(self, pos: np.ndarray, cos_angle: float, sin_angle: float) -> float:
        """
        Calculate distance to wall in the facing direction using ray casting.
        """
        x, y = pos[0], pos[1]
        distances = []

        if cos_angle > 1e-6:
            t = (self.world_size - x) / cos_angle
            distances.append(t)
        elif cos_angle < -1e-6:
            t = -x / cos_angle
            distances.append(t)

        if sin_angle > 1e-6:
            t = (self.world_size - y) / sin_angle
            distances.append(t)
        elif sin_angle < -1e-6:
            t = -y / sin_angle
            distances.append(t)

        return min(distances) if distances else self.world_size

    def _compute_monster_features(
        self,
        player_pos: np.ndarray,
        player_angle: float,
        monster_positions: List[np.ndarray]
    ) -> np.ndarray:
        """
        Compute features for all monsters, sorted by distance.

        Returns:
            Array of shape (MAX_MONSTERS * FEATURES_PER_MONSTER,)
            Dead/missing monsters have features set to 0.
        """
        features = np.zeros(self.MAX_MONSTERS * self.FEATURES_PER_MONSTER)

        if not monster_positions:
            return features

        # Compute distance and features for each monster
        monster_data = []
        for pos in monster_positions:
            rel_pos = (pos - player_pos) / self.world_size
            dist = np.linalg.norm(pos - player_pos) / self.world_size

            angle_to_monster = np.arctan2(
                pos[1] - player_pos[1],
                pos[0] - player_pos[0]
            )
            relative_angle = np.arctan2(
                np.sin(angle_to_monster - player_angle),
                np.cos(angle_to_monster - player_angle)
            )

            monster_data.append((dist, rel_pos[0], rel_pos[1], dist, relative_angle))

        # Sort by distance (nearest first)
        monster_data.sort(key=lambda x: x[0])

        # Fill features
        for i, (_, dx, dy, dist, angle) in enumerate(monster_data):
            if i >= self.MAX_MONSTERS:
                break
            offset = i * self.FEATURES_PER_MONSTER
            features[offset] = dx
            features[offset + 1] = dy
            features[offset + 2] = dist
            features[offset + 3] = angle

        return features

    def extract(self, world: GameWorld) -> np.ndarray:
        """
        Extract features from the current world state.

        Args:
            world: The game world

        Returns:
            Feature vector of size n_features (26)
        """
        if world.player is None:
            return np.zeros(self.n_features)

        player_pos = world.get_player_position()
        player_angle = world.get_player_angle()
        monster_positions = world.get_alive_monster_positions()
        blood_pos = world.get_blood_pack_position()

        # Monster features (sorted by distance, 0 for dead)
        monster_features = self._compute_monster_features(
            player_pos, player_angle, monster_positions
        )

        # Blood pack features
        rel_blood = (blood_pos - player_pos) / self.world_size
        dist_to_blood = np.linalg.norm(blood_pos - player_pos) / self.world_size
        angle_to_blood = np.arctan2(
            blood_pos[1] - player_pos[1],
            blood_pos[0] - player_pos[0]
        )
        relative_angle_blood = np.arctan2(
            np.sin(angle_to_blood - player_angle),
            np.cos(angle_to_blood - player_angle)
        )

        # Player facing direction
        cos_angle = np.cos(player_angle)
        sin_angle = np.sin(player_angle)

        # Distance to wall in facing direction
        dist_to_wall = self._raycast_to_wall(
            player_pos, cos_angle, sin_angle
        ) / self.world_size

        # Casting state
        casting_progress = world.get_casting_progress()
        is_ready_to_cast = 1.0 if world.is_player_ready_to_cast() else 0.0

        # Combine all features
        return np.concatenate([
            monster_features,                    # 0-15: Monster features (4 monsters * 4 features)
            [rel_blood[0], rel_blood[1]],        # 16-17: Blood pack relative position
            [dist_to_blood],                     # 18: Blood pack distance
            [relative_angle_blood],              # 19: Blood pack relative angle
            [cos_angle, sin_angle],              # 20-21: Player facing direction
            [dist_to_wall],                      # 22: Wall distance (facing)
            [casting_progress],                  # 23: Current cast progress
            [is_ready_to_cast],                  # 24: Ready to cast indicator
            [1.0]                                # 25: Bias term
        ])

    def extract_from_raw(
        self,
        agent_pos: np.ndarray,
        agent_angle: float,
        monster_positions: List[np.ndarray],
        blood_pos: np.ndarray,
        wind_up: int
    ) -> np.ndarray:
        """
        Extract features from raw position data.
        Useful for environments without the full GameWorld.

        Args:
            agent_pos: Player position [x, y]
            agent_angle: Player facing angle
            monster_positions: List of monster positions (alive only)
            blood_pos: Blood pack position [x, y]
            wind_up: Current wind-up ticks remaining

        Returns:
            Feature vector
        """
        # Monster features
        monster_features = self._compute_monster_features(
            agent_pos, agent_angle, monster_positions
        )

        # Blood pack features
        rel_blood = (blood_pos - agent_pos) / self.world_size
        dist_to_blood = np.linalg.norm(blood_pos - agent_pos) / self.world_size
        angle_to_blood = np.arctan2(
            blood_pos[1] - agent_pos[1],
            blood_pos[0] - agent_pos[0]
        )
        relative_angle_blood = np.arctan2(
            np.sin(angle_to_blood - agent_angle),
            np.cos(angle_to_blood - agent_angle)
        )

        cos_angle = np.cos(agent_angle)
        sin_angle = np.sin(agent_angle)

        dist_to_wall = self._raycast_to_wall(
            agent_pos, cos_angle, sin_angle
        ) / self.world_size

        casting_progress = wind_up / 4.0
        is_ready_to_cast = 1.0 if wind_up == 0 else 0.0

        return np.concatenate([
            monster_features,
            [rel_blood[0], rel_blood[1]],
            [dist_to_blood],
            [relative_angle_blood],
            [cos_angle, sin_angle],
            [dist_to_wall],
            [casting_progress],
            [is_ready_to_cast],
            [1.0]
        ])
