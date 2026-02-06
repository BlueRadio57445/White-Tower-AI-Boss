"""
Monster behavior system - simulating real Minecraft player actions.

Actions are Cartesian product of:
- Movement: {FORWARD, BACKWARD, LEFT, RIGHT, SPRINT_FORWARD, IDLE}
- Turning: {LEFT, RIGHT, TURN_180, NONE}
- Attack: {NO_ATTACK, ATTACK}
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, Any, Optional, Type, List, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from game.entity import Entity
    from game.world import GameWorld


# =============================================================================
# Action Primitives
# =============================================================================

class MovementType(Enum):
    """Movement actions (WASD keys)"""
    IDLE = auto()           # No movement
    FORWARD = auto()        # W - walk forward
    BACKWARD = auto()       # S - walk backward (slower)
    LEFT = auto()           # A - strafe left
    RIGHT = auto()          # D - strafe right
    SPRINT_FORWARD = auto() # W + Sprint - run forward (only forward can sprint)


class TurningType(Enum):
    """Turning actions (mouse movement)"""
    NONE = auto()           # No turning
    LEFT = auto()           # Turn left (positive angle)
    RIGHT = auto()          # Turn right (negative angle)
    TURN_180 = auto()       # Quick 180 degree turn


class AttackType(Enum):
    """Attack actions (mouse click)"""
    NONE = auto()           # No attack
    ATTACK = auto()         # Attack (melee or ranged based on weapon)


@dataclass
class MonsterAction:
    """
    Complete monster action for one tick.

    Monsters can move, turn, and attack simultaneously (unlike Agent).
    """
    movement: MovementType = MovementType.IDLE
    turning: TurningType = TurningType.NONE
    attack: AttackType = AttackType.NONE


# =============================================================================
# Movement Speed Constants
# =============================================================================

class MovementSpeed:
    """Speed constants for different movement types"""
    WALK_FORWARD = 0.25
    WALK_BACKWARD = 0.15    # Slower than forward
    STRAFE = 0.20           # Slightly slower than forward
    SPRINT = 0.45           # Much faster, but only forward


class TurningSpeed:
    """Turning speed constants"""
    NORMAL = 0.3            # Radians per tick
    TURN_180 = np.pi        # Instant 180 (takes 1 tick)


# =============================================================================
# Monster Behavior Base Class
# =============================================================================

class MonsterBehavior(ABC):
    """
    Abstract base class for monster AI behaviors.

    Each behavior decides what action a monster should take based on:
    - Monster's current state (position, health, etc.)
    - Agent's state (position, casting, etc.)
    - World state (other monsters, obstacles, etc.)
    """

    behavior_type: str = "base"

    def __init__(
        self,
        walk_speed: float = MovementSpeed.WALK_FORWARD,
        sprint_speed: float = MovementSpeed.SPRINT,
        turn_speed: float = TurningSpeed.NORMAL,
        attack_cooldown_ticks: int = 20,  # Minecraft: ~1 second
        attack_range: float = 3.0,
        attack_damage: float = 10.0
    ):
        """
        Initialize behavior.

        Args:
            walk_speed: Walking speed (units/tick)
            sprint_speed: Sprinting speed (units/tick)
            turn_speed: Turning speed (radians/tick)
            attack_cooldown_ticks: Ticks between attacks
            attack_range: Attack range (for both melee and ranged)
            attack_damage: Damage per attack
        """
        self.walk_speed = walk_speed
        self.sprint_speed = sprint_speed
        self.turn_speed = turn_speed
        self.attack_cooldown_ticks = attack_cooldown_ticks
        self.attack_range = attack_range
        self.attack_damage = attack_damage

        # Internal state
        self._attack_cooldown_remaining = 0
        self._internal_state: Dict[str, Any] = {}

    @abstractmethod
    def decide_action(
        self,
        entity: 'Entity',
        world: 'GameWorld'
    ) -> MonsterAction:
        """
        Decide what action to take this tick.

        Args:
            entity: The monster entity
            world: The game world

        Returns:
            MonsterAction with movement, turning, and attack decisions
        """
        pass

    def update_cooldowns(self):
        """Update cooldown timers (called every tick)"""
        if self._attack_cooldown_remaining > 0:
            self._attack_cooldown_remaining -= 1

    def can_attack(self) -> bool:
        """Check if attack is off cooldown"""
        return self._attack_cooldown_remaining <= 0

    def use_attack(self):
        """Consume attack cooldown"""
        self._attack_cooldown_remaining = self.attack_cooldown_ticks

    def reset(self):
        """Reset behavior state (for new episode)"""
        self._attack_cooldown_remaining = 0
        self._internal_state.clear()

    # =========================================================================
    # Helper Methods for Subclasses
    # =========================================================================

    def _get_angle_to_target(
        self,
        entity_pos: np.ndarray,
        entity_angle: float,
        target_pos: np.ndarray
    ) -> float:
        """
        Get angle difference to target (positive = target is to the left).

        Returns:
            Angle in radians, normalized to [-pi, pi]
        """
        delta = target_pos - entity_pos
        target_angle = np.arctan2(delta[1], delta[0])
        angle_diff = target_angle - entity_angle

        # Normalize to [-pi, pi]
        while angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        while angle_diff < -np.pi:
            angle_diff += 2 * np.pi

        return angle_diff

    def _get_distance_to_target(
        self,
        entity_pos: np.ndarray,
        target_pos: np.ndarray
    ) -> float:
        """Get distance to target"""
        return np.linalg.norm(target_pos - entity_pos)

    def _decide_turning(self, angle_diff: float) -> TurningType:
        """
        Decide turning action based on angle difference.

        Args:
            angle_diff: Angle to target (positive = left)

        Returns:
            TurningType
        """
        # If facing almost opposite direction, use 180 turn
        if abs(angle_diff) > 2.5:  # ~143 degrees
            return TurningType.TURN_180
        # Otherwise, turn toward target
        elif angle_diff > 0.1:
            return TurningType.LEFT
        elif angle_diff < -0.1:
            return TurningType.RIGHT
        else:
            return TurningType.NONE

    def _is_facing_target(self, angle_diff: float, tolerance: float = 0.3) -> bool:
        """Check if roughly facing the target"""
        return abs(angle_diff) < tolerance

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Serialize behavior to dictionary"""
        return {
            "type": self.behavior_type,
            "walk_speed": self.walk_speed,
            "sprint_speed": self.sprint_speed,
            "turn_speed": self.turn_speed,
            "attack_cooldown_ticks": self.attack_cooldown_ticks,
            "attack_range": self.attack_range,
            "attack_damage": self.attack_damage
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MonsterBehavior':
        """Create behavior from dictionary"""
        params = {k: v for k, v in data.items() if k != "type"}
        return cls(**params)


# =============================================================================
# Behavior Registry
# =============================================================================

class BehaviorRegistry:
    """Registry for monster behavior types"""

    _behaviors: Dict[str, Type[MonsterBehavior]] = {}

    @classmethod
    def register(cls, behavior_class: Type[MonsterBehavior]) -> Type[MonsterBehavior]:
        """Register a behavior class (can be used as decorator)"""
        cls._behaviors[behavior_class.behavior_type] = behavior_class
        return behavior_class

    @classmethod
    def create(cls, behavior_type: str, **params) -> MonsterBehavior:
        """Create a behavior instance by type name"""
        if behavior_type not in cls._behaviors:
            raise KeyError(f"Unknown behavior type: {behavior_type}")
        return cls._behaviors[behavior_type](**params)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MonsterBehavior:
        """Create behavior from dictionary"""
        behavior_type = data.get("type", "stationary")
        behavior_class = cls._behaviors.get(behavior_type)

        if behavior_class is None:
            raise KeyError(f"Unknown behavior type: {behavior_type}")

        return behavior_class.from_dict(data)

    @classmethod
    def list_types(cls) -> List[str]:
        """Get list of registered behavior types"""
        return list(cls._behaviors.keys())


# =============================================================================
# Action Executor
# =============================================================================

class MonsterActionExecutor:
    """
    Executes MonsterAction and updates entity state.

    This is separate from behavior to allow different physics implementations.
    """

    def __init__(self, room_size: float = 10.0):
        self.room_size = room_size

    def execute(
        self,
        entity: 'Entity',
        action: MonsterAction,
        behavior: MonsterBehavior
    ) -> Dict[str, Any]:
        """
        Execute action and update entity.

        Args:
            entity: The monster entity
            action: The action to execute
            behavior: The behavior (for speed constants)

        Returns:
            Dict with execution results (hit_wall, attacked, etc.)
        """
        result = {
            "hit_wall": False,
            "attacked": False,
            "attack_damage": 0.0
        }

        # 1. Execute turning (instantaneous)
        self._execute_turning(entity, action.turning, behavior.turn_speed)

        # 2. Execute movement
        hit_wall = self._execute_movement(entity, action.movement, behavior)
        result["hit_wall"] = hit_wall

        # 3. Execute attack (if cooldown ready)
        if action.attack == AttackType.ATTACK and behavior.can_attack():
            behavior.use_attack()
            result["attacked"] = True
            result["attack_damage"] = behavior.attack_damage

        # 4. Update cooldowns
        behavior.update_cooldowns()

        return result

    def _execute_turning(
        self,
        entity: 'Entity',
        turning: TurningType,
        turn_speed: float
    ):
        """Apply turning to entity"""
        if turning == TurningType.NONE:
            return

        if turning == TurningType.TURN_180:
            entity.angle += np.pi
        elif turning == TurningType.LEFT:
            entity.angle += turn_speed
        elif turning == TurningType.RIGHT:
            entity.angle -= turn_speed

        # Normalize angle to [-pi, pi]
        while entity.angle > np.pi:
            entity.angle -= 2 * np.pi
        while entity.angle < -np.pi:
            entity.angle += 2 * np.pi

    def _execute_movement(
        self,
        entity: 'Entity',
        movement: MovementType,
        behavior: MonsterBehavior
    ) -> bool:
        """
        Apply movement to entity.

        Returns:
            True if hit wall
        """
        if movement == MovementType.IDLE:
            return False

        # Calculate movement vector based on type
        cos_a = np.cos(entity.angle)
        sin_a = np.sin(entity.angle)

        # Forward/backward vectors
        forward = np.array([cos_a, sin_a])
        right = np.array([sin_a, -cos_a])  # Perpendicular to forward

        if movement == MovementType.FORWARD:
            velocity = forward * behavior.walk_speed
        elif movement == MovementType.BACKWARD:
            velocity = -forward * MovementSpeed.WALK_BACKWARD
        elif movement == MovementType.LEFT:
            velocity = -right * MovementSpeed.STRAFE
        elif movement == MovementType.RIGHT:
            velocity = right * MovementSpeed.STRAFE
        elif movement == MovementType.SPRINT_FORWARD:
            velocity = forward * behavior.sprint_speed
        else:
            velocity = np.array([0.0, 0.0])

        # Apply movement
        current_pos = entity.position.as_array()
        new_pos = current_pos + velocity

        # Check boundaries
        hit_wall = False
        margin = 0.3

        if new_pos[0] < margin:
            new_pos[0] = margin
            hit_wall = True
        elif new_pos[0] > self.room_size - margin:
            new_pos[0] = self.room_size - margin
            hit_wall = True

        if new_pos[1] < margin:
            new_pos[1] = margin
            hit_wall = True
        elif new_pos[1] > self.room_size - margin:
            new_pos[1] = self.room_size - margin
            hit_wall = True

        # Update position component
        entity.position.x = new_pos[0]
        entity.position.y = new_pos[1]
        return hit_wall
