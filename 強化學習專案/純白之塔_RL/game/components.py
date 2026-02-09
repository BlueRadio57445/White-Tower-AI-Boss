"""
Component system for entities.
Components are pure data containers that can be attached to entities.
"""

from dataclasses import dataclass, field
from typing import Set, Optional, Dict, Any, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from game.skills import SkillShapeType


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
        current_skill_range: Range of current skill
        current_skill_angle_tolerance: Angle tolerance of current skill
        current_skill_damage: Damage of current skill
        current_skill_shape_type: Shape type of current skill (CONE, RING, RECTANGLE, PROJECTILE)
        current_skill_extra_params: Extra parameters for current skill
        current_skill_wind_up_total: Total wind-up ticks (for progress calculation)
        current_skill_cooldown: Cooldown ticks to apply when current skill completes
        skill_cooldowns: Remaining cooldown ticks per skill_id
    """
    wind_up_remaining: int = 0
    aim_angle: float = 0.0
    current_skill: Optional[str] = None

    # Skill parameters (set when casting starts)
    current_skill_range: float = 6.0
    current_skill_angle_tolerance: float = 0.4
    current_skill_damage: float = 100.0
    current_skill_shape_type: Optional[str] = None  # Store as string to avoid circular import
    current_skill_extra_params: Dict[str, Any] = field(default_factory=dict)
    current_skill_wind_up_total: int = 4
    current_skill_cooldown: int = 0

    # Per-skill cooldown tracking
    skill_cooldowns: Dict[str, int] = field(default_factory=dict)

    @property
    def is_casting(self) -> bool:
        """Check if currently casting a skill (including instant cast skills)."""
        return self.wind_up_remaining > 0 or self.current_skill is not None

    @property
    def is_ready(self) -> bool:
        """Check if ready to cast a new skill."""
        return self.wind_up_remaining == 0

    def start_cast(
        self,
        skill_id: str,
        wind_up_ticks: int,
        aim_angle: float,
        skill_range: float = 6.0,
        angle_tolerance: float = 0.4,
        damage: float = 100.0,
        shape_type: Optional[str] = None,
        extra_params: Optional[Dict[str, Any]] = None,
        cooldown_ticks: int = 0
    ) -> None:
        """Begin casting a skill."""
        self.wind_up_remaining = wind_up_ticks
        self.current_skill_wind_up_total = wind_up_ticks
        self.aim_angle = aim_angle
        self.current_skill = skill_id
        self.current_skill_range = skill_range
        self.current_skill_angle_tolerance = angle_tolerance
        self.current_skill_damage = damage
        self.current_skill_shape_type = shape_type or "cone"
        self.current_skill_extra_params = extra_params or {}
        self.current_skill_cooldown = cooldown_ticks

    def tick(self) -> bool:
        """
        Advance wind-up timer by one tick.

        Returns:
            True if skill completed this tick, False otherwise
        """
        # Handle instant cast skills (wind_up = 0)
        if self.wind_up_remaining == 0 and self.current_skill is not None:
            completed_skill = self.current_skill
            self.current_skill = None
            # Apply cooldown when skill finishes casting
            if completed_skill and self.current_skill_cooldown > 0:
                self.skill_cooldowns[completed_skill] = self.current_skill_cooldown
            self.current_skill_cooldown = 0
            return True

        if self.wind_up_remaining > 0:
            self.wind_up_remaining -= 1
            if self.wind_up_remaining == 0:
                completed_skill = self.current_skill
                self.current_skill = None
                # Apply cooldown when skill finishes casting
                if completed_skill and self.current_skill_cooldown > 0:
                    self.skill_cooldowns[completed_skill] = self.current_skill_cooldown
                self.current_skill_cooldown = 0
                return True
        return False

    def tick_cooldowns(self) -> None:
        """Decrement all per-skill cooldown timers by one tick."""
        for skill_id in list(self.skill_cooldowns.keys()):
            if self.skill_cooldowns[skill_id] > 0:
                self.skill_cooldowns[skill_id] -= 1

    def is_skill_available(self, skill_id: str) -> bool:
        """Check if a skill is off cooldown and can be cast."""
        return self.skill_cooldowns.get(skill_id, 0) == 0


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
