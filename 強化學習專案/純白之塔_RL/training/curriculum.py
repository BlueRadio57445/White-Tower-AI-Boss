"""
Curriculum manager for staged training.

A curriculum defines sequential training stages. Each stage specifies:
  - Epoch range (epoch_start to epoch_end, 0-indexed, inclusive)
  - A list of level file names to randomly sample from each episode
  - One or more action masks to randomly sample from each episode

Curriculum JSON format:
    {
        "name": "my_curriculum",
        "description": "...",
        "stages": [
            {
                "label": "移動基礎",
                "epoch_start": 0,
                "epoch_end": 499,
                "levels": ["01", "02", "03"],
                "mask": "movement_only"
            },
            {
                "label": "技能混合",
                "epoch_start": 500,
                "epoch_end": 999,
                "levels": ["04"],
                "mask": ["missile_only", "hammer_only", "outer_slash_only"]
            },
            ...
        ]
    }

Notes on "mask":
  - String form  "mask": "name"           → fixed mask every episode
  - List form    "mask": ["a", "b", "c"]  → randomly pick one mask per episode

Notes:
  - Epochs are 0-indexed (epoch 0 is the first training epoch).
  - Stages must not overlap. Gaps between stages are allowed.
  - If an epoch falls in a gap, Trainer falls back to its fixed level_config.

Usage:
    from training.curriculum import CurriculumManager

    mgr = CurriculumManager("curricula/missile_training.json")
    print(mgr.describe())

    # In training loop (epoch is 0-indexed):
    if mgr.covers_epoch(epoch):
        level_cfg = mgr.get_level_config(epoch)   # random level from stage
        mask      = mgr.get_mask(epoch)            # action mask for stage
"""

import json
import random

import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

from training.level_config import LevelConfig, LevelLoader
from training.mask_loader import ActionMaskLoader


@dataclass
class CurriculumStage:
    """One training stage in a curriculum."""
    label: str
    epoch_start: int
    epoch_end: int
    level_names: List[str]           # File stems, for display (e.g. ["01", "02"])
    level_configs: List[LevelConfig] # Pre-loaded LevelConfig objects
    mask_names: List[str]            # Mask file stems, for display (e.g. ["missile_only"])
    masks: List[np.ndarray]          # Pre-loaded mask arrays (one per mask_name)


class CurriculumManager:
    """
    Loads a curriculum JSON and provides level configs and action masks
    for each training epoch.

    Epoch numbers match the training loop's 0-indexed counter.
    """

    def __init__(
        self,
        curriculum_path: Union[str, Path],
        levels_dir: Union[str, Path] = "levels",
        masks_dir: Union[str, Path] = "masks",
    ):
        """
        Load a curriculum from a JSON file. All referenced levels and masks
        are pre-loaded at construction time so runtime queries are fast.

        Args:
            curriculum_path: Path to the curriculum JSON file.
            levels_dir: Directory containing level JSON files.
            masks_dir: Directory containing mask JSON files.

        Raises:
            FileNotFoundError: If any referenced level or mask file is missing.
            ValueError: If stages overlap or have invalid epoch ranges.
        """
        self.curriculum_path = Path(curriculum_path)
        self.levels_dir = Path(levels_dir)
        self.masks_dir = Path(masks_dir)

        with open(self.curriculum_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.name: str = data.get("name", "unnamed")
        self.description: str = data.get("description", "")
        self.stages: List[CurriculumStage] = [
            self._load_stage(raw) for raw in data["stages"]
        ]

        self._validate_no_overlap()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_stage(self, raw: dict) -> CurriculumStage:
        label = raw.get("label", "")
        epoch_start = int(raw["epoch_start"])
        epoch_end = int(raw["epoch_end"])

        if epoch_end < epoch_start:
            raise ValueError(
                f"Curriculum '{self.name}': stage '{label}' has "
                f"epoch_end ({epoch_end}) < epoch_start ({epoch_start})."
            )

        level_names = [str(n) for n in raw["levels"]]
        if not level_names:
            raise ValueError(
                f"Curriculum '{self.name}': stage '{label}' has no levels."
            )

        # "mask" accepts a single name (string) or a list of names
        raw_mask = raw["mask"]
        mask_names = [str(raw_mask)] if isinstance(raw_mask, str) else [str(n) for n in raw_mask]
        if not mask_names:
            raise ValueError(
                f"Curriculum '{self.name}': stage '{label}' has no masks."
            )

        level_configs = [
            LevelLoader.from_json(str(self.levels_dir / f"{n}.json"))
            for n in level_names
        ]
        masks = [ActionMaskLoader.load_by_name(n, self.masks_dir) for n in mask_names]

        return CurriculumStage(
            label=label,
            epoch_start=epoch_start,
            epoch_end=epoch_end,
            level_names=level_names,
            level_configs=level_configs,
            mask_names=mask_names,
            masks=masks,
        )

    def _validate_no_overlap(self) -> None:
        sorted_stages = sorted(self.stages, key=lambda s: s.epoch_start)
        for i in range(len(sorted_stages) - 1):
            a, b = sorted_stages[i], sorted_stages[i + 1]
            if a.epoch_end >= b.epoch_start:
                raise ValueError(
                    f"Curriculum '{self.name}': stages overlap. "
                    f"'{a.label or '(unnamed)'}' ends at epoch {a.epoch_end}, "
                    f"'{b.label or '(unnamed)'}' starts at epoch {b.epoch_start}."
                )

    # ------------------------------------------------------------------
    # Epoch queries
    # ------------------------------------------------------------------

    def covers_epoch(self, epoch: int) -> bool:
        """Return True if any stage covers this epoch."""
        return any(s.epoch_start <= epoch <= s.epoch_end for s in self.stages)

    def get_stage(self, epoch: int) -> CurriculumStage:
        """
        Return the stage that covers this epoch.

        Raises:
            ValueError: If no stage covers the given epoch.
        """
        for stage in self.stages:
            if stage.epoch_start <= epoch <= stage.epoch_end:
                return stage
        raise ValueError(
            f"Curriculum '{self.name}': epoch {epoch} is not covered by any stage. "
            f"Defined ranges: {[(s.epoch_start, s.epoch_end) for s in self.stages]}"
        )

    def get_level_config(self, epoch: int) -> LevelConfig:
        """Randomly sample one level from the current stage."""
        return random.choice(self.get_stage(epoch).level_configs)

    def get_mask(self, epoch: int) -> np.ndarray:
        """Randomly sample one action mask from the current stage."""
        return random.choice(self.get_stage(epoch).masks)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def describe(self) -> str:
        """Return a human-readable summary of this curriculum."""
        lines = [f"=== Curriculum: {self.name} ==="]
        if self.description:
            lines.append(f"  {self.description}")
        lines.append(f"  Stages ({len(self.stages)}):")
        for s in self.stages:
            lines.append(
                f"    [epoch {s.epoch_start:>5} - {s.epoch_end:<5}]  "
                f"{s.label or '(unnamed)':<20}  "
                f"levels={s.level_names}  masks={s.mask_names}"
            )
        return "\n".join(lines)
