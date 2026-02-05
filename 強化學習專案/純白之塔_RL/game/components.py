"""
Component system for entities.
Components are pure data containers that can be attached to entities.
"""

from dataclasses import dataclass, field
from typing import Set, Optional
import numpy as np


@dataclass
class Position:
    """
    Position and orientation component.

    Attributes:
        x: X coordinate in world space
        y: Y coordinate in world space
        angle: Facing direction in radians
    """
    x: float = 0.0
    y: float = 0.0
    angle: float = 0.0

    def as_array(self) -> np.ndarray:
        """Return position as numpy array [x, y]."""
        return np.array([self.x, self.y])

    def set_from_array(self, arr: np.ndarray) -> None:
        """Set position from numpy array."""
        self.x = float(arr[0])
        self.y = float(arr[1])

    def distance_to(self, other: 'Position') -> float:
        """Calculate Euclidean distance to another position."""
        return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def angle_to(self, other: 'Position') -> float:
        """Calculate angle to another position."""
        return np.arctan2(other.y - self.y, other.x - self.x)

    def copy(self) -> 'Position':
        """Create a copy of this position."""
        return Position(self.x, self.y, self.angle)


@dataclass
class Health:
    """
    Health component for damageable entities.

    Attributes:
        current: Current health points
        maximum: Maximum health points
    """
    current: float = 100.0
    maximum: float = 100.0

    @property
    def is_alive(self) -> bool:
        """Check if entity is still alive."""
        return self.current > 0

    @property
    def percentage(self) -> float:
        """Get health as percentage (0.0 to 1.0)."""
        return self.current / self.maximum if self.maximum > 0 else 0.0

    def damage(self, amount: float) -> float:
        """
        Apply damage and return actual damage dealt.

        Args:
            amount: Damage to apply

        Returns:
            Actual damage dealt (may be less if health is low)
        """
        actual = min(amount, self.current)
        self.current -= actual
        return actual

    def heal(self, amount: float) -> float:
        """
        Apply healing and return actual amount healed.

        Args:
            amount: Health to restore

        Returns:
            Actual health restored (capped at maximum)
        """
        actual = min(amount, self.maximum - self.current)
        self.current += actual
        return actual


@dataclass
class Skills:
    """
    Skill state component for entities that can cast skills.

    Attributes:
        wind_up_remaining: Ticks remaining in current wind-up (0 = ready)
        aim_angle: Aimed direction for current skill
        current_skill: ID of skill being cast (if any)
    """
    wind_up_remaining: int = 0
    aim_angle: float = 0.0
    current_skill: Optional[str] = None

    @property
    def is_casting(self) -> bool:
        """Check if currently casting a skill."""
        return self.wind_up_remaining > 0

    @property
    def is_ready(self) -> bool:
        """Check if ready to cast a new skill."""
        return self.wind_up_remaining == 0

    def start_cast(self, skill_id: str, wind_up_ticks: int, aim_angle: float) -> None:
        """Begin casting a skill."""
        self.wind_up_remaining = wind_up_ticks
        self.aim_angle = aim_angle
        self.current_skill = skill_id

    def tick(self) -> bool:
        """
        Advance wind-up timer by one tick.

        Returns:
            True if skill completed this tick, False otherwise
        """
        if self.wind_up_remaining > 0:
            self.wind_up_remaining -= 1
            if self.wind_up_remaining == 0:
                completed_skill = self.current_skill
                self.current_skill = None
                return True
        return False


@dataclass
class Tags:
    """
    Tag component for categorizing entities.

    Attributes:
        tags: Set of string tags for this entity
    """
    tags: Set[str] = field(default_factory=set)

    def add(self, tag: str) -> None:
        """Add a tag."""
        self.tags.add(tag)

    def remove(self, tag: str) -> None:
        """Remove a tag if present."""
        self.tags.discard(tag)

    def has(self, tag: str) -> bool:
        """Check if entity has a tag."""
        return tag in self.tags

    def has_any(self, *tags: str) -> bool:
        """Check if entity has any of the given tags."""
        return bool(self.tags.intersection(tags))

    def has_all(self, *tags: str) -> bool:
        """Check if entity has all of the given tags."""
        return all(tag in self.tags for tag in tags)
