"""
Action mask loader for curriculum learning.

Loads action masks from JSON files. Masks restrict which discrete actions
are available to the agent during a training stage.

Mask JSON format:
    {
        "name": "missile_only",
        "description": "...",
        "allowed_actions": [0, 1, 2, 3, 5]
    }

Action indices:
    0=FORWARD, 1=BACKWARD, 2=LEFT, 3=RIGHT,
    4=OUTER_SLASH, 5=MISSILE, 6=HAMMER, 7=DASH,
    8=SOUL_CLAW, 9=SOUL_PALM, 10=BLOOD_POOL, 11=SUMMON_PACK
"""

import json
import numpy as np
from pathlib import Path
from typing import Union


N_ACTIONS = 12


class ActionMaskLoader:
    """Loads action masks from JSON files."""

    @staticmethod
    def load(path: Union[str, Path]) -> np.ndarray:
        """
        Load an action mask from a JSON file.

        Args:
            path: Path to the mask JSON file.

        Returns:
            Float array of shape (N_ACTIONS,). 1.0 = allowed, 0.0 = blocked.
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        mask = np.zeros(N_ACTIONS, dtype=float)
        for idx in data["allowed_actions"]:
            if 0 <= idx < N_ACTIONS:
                mask[idx] = 1.0
            else:
                raise ValueError(f"Mask '{path}': action index {idx} out of range [0, {N_ACTIONS})")

        return mask

    @staticmethod
    def load_by_name(name: str, masks_dir: Union[str, Path] = "masks") -> np.ndarray:
        """
        Load a mask by name (without .json extension).

        Args:
            name: Mask filename without extension (e.g., "missile_only").
            masks_dir: Directory containing mask JSON files.

        Returns:
            Float array of shape (N_ACTIONS,).
        """
        path = Path(masks_dir) / f"{name}.json"
        return ActionMaskLoader.load(path)

    @staticmethod
    def no_mask() -> np.ndarray:
        """Return a mask that allows all actions (no restriction)."""
        return np.ones(N_ACTIONS, dtype=float)

    @staticmethod
    def combine(mask_a: np.ndarray, mask_b: np.ndarray) -> np.ndarray:
        """
        Combine two masks: an action is allowed only if BOTH masks allow it.

        Used to merge the curriculum mask (which skills are taught this stage)
        with the world mask (which skills are off cooldown right now).

        Args:
            mask_a: First mask, shape (N_ACTIONS,).
            mask_b: Second mask, shape (N_ACTIONS,).

        Returns:
            Element-wise product, shape (N_ACTIONS,).
        """
        return mask_a * mask_b
