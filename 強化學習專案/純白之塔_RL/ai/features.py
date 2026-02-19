"""
Feature extraction for the RL agent.

Single source of truth: ai/feature_registry.py
This module is a thin wrapper so existing code (trainer, imports) stays unchanged.

To add or modify features, edit feature_registry.py.
To use a custom feature subset (e.g. for FSS), use SelectiveFeatureExtractor directly.
"""

import numpy as np
from game.world import GameWorld
from ai.feature_registry import SelectiveFeatureExtractor, DEFAULT_GROUP_ORDER


class FeatureExtractor:
    """
    Default feature extractor: all registered groups in DEFAULT_GROUP_ORDER.
    Produces the same 27D vector as the original implementation.

    Feature layout (27 dimensions):
    [0-15]  Monster features (4 monsters, interleaved: dx, dy, dist, angle)
    [16-17] Blood pack (dx, dy)
    [18]    Blood pack distance
    [19]    Blood pack relative angle
    [20-21] Player facing (cos, sin)
    [22]    Distance to wall (facing direction)
    [23-24] Casting info (progress, ready_to_cast)
    [25]    Player health (0-1)
    [26]    Bias term (1.0)
    """

    def __init__(self, world_size: float = 10.0):
        self._extractor = SelectiveFeatureExtractor(DEFAULT_GROUP_ORDER, world_size)
        self.world_size = world_size
        self.n_features = self._extractor.n_features

    def extract(self, world: GameWorld) -> np.ndarray:
        """Extract the 27D feature vector from current world state."""
        return self._extractor.extract(world)
