"""
Monster behavior system.
Defines different AI behaviors for monsters.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Tuple
import numpy as np

from game.entity import Entity
from game.physics import PhysicsSystem


class MonsterBehavior(ABC):
    """Abstract base class for monster behaviors."""

    @abstractmethod
    def update(
        self,
        monster: Entity,
        player: Optional[Entity],
        physics: PhysicsSystem,
        dt: float = 1.0
    ) -> None:
        """
        Update monster based on behavior.

        Args:
            monster: The monster entity
            player: The player entity (if visible)
            physics: Physics system for movement
            dt: Time delta
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get the behavior name."""
        pass


class IdleBehavior(MonsterBehavior):
    """
    Idle behavior - monster stands still.
    This is the default behavior used in the original implementation.
    """

    def update(
        self,
        monster: Entity,
        player: Optional[Entity],
        physics: PhysicsSystem,
        dt: float = 1.0
    ) -> None:
        """Monster does nothing."""
        pass

    def get_name(self) -> str:
        return "idle"


class WanderBehavior(MonsterBehavior):
    """
    Wander behavior - monster moves randomly.
    """

    def __init__(
        self,
        move_speed: float = 0.3,
        direction_change_chance: float = 0.1
    ):
        """
        Initialize wander behavior.

        Args:
            move_speed: Movement speed per tick
            direction_change_chance: Probability of changing direction each tick
        """
        self.move_speed = move_speed
        self.direction_change_chance = direction_change_chance
        self._current_direction: Optional[np.ndarray] = None

    def update(
        self,
        monster: Entity,
        player: Optional[Entity],
        physics: PhysicsSystem,
        dt: float = 1.0
    ) -> None:
        """Move in a random direction, occasionally changing."""
        if not monster.has_position():
            return

        # Change direction randomly or if no direction set
        if self._current_direction is None or \
           np.random.random() < self.direction_change_chance:
            angle = np.random.uniform(0, 2 * np.pi)
            self._current_direction = np.array([
                np.cos(angle),
                np.sin(angle)
            ])

        # Move in current direction
        movement = self._current_direction * self.move_speed * dt
        success = physics.move_entity(monster, movement)

        # Change direction if hit wall
        if not success:
            self._current_direction = None

    def get_name(self) -> str:
        return "wander"


class ChaseBehavior(MonsterBehavior):
    """
    Chase behavior - monster moves toward player.
    """

    def __init__(
        self,
        chase_speed: float = 0.4,
        detection_range: float = 8.0,
        give_up_range: float = 12.0
    ):
        """
        Initialize chase behavior.

        Args:
            chase_speed: Movement speed when chasing
            detection_range: Distance at which monster starts chasing
            give_up_range: Distance at which monster stops chasing
        """
        self.chase_speed = chase_speed
        self.detection_range = detection_range
        self.give_up_range = give_up_range
        self._is_chasing = False

    def update(
        self,
        monster: Entity,
        player: Optional[Entity],
        physics: PhysicsSystem,
        dt: float = 1.0
    ) -> None:
        """Chase the player if in range."""
        if player is None or not monster.has_position() or not player.has_position():
            return

        distance = monster.position.distance_to(player.position)

        # Start chasing if player comes close
        if distance < self.detection_range:
            self._is_chasing = True
        # Stop chasing if player gets too far
        elif distance > self.give_up_range:
            self._is_chasing = False

        if self._is_chasing:
            # Calculate direction to player
            direction = np.array([
                player.position.x - monster.position.x,
                player.position.y - monster.position.y
            ])

            # Normalize and scale by speed
            length = np.linalg.norm(direction)
            if length > 0:
                direction = direction / length * self.chase_speed * dt
                physics.move_entity(monster, direction)

    def get_name(self) -> str:
        return "chase"


class PatrolBehavior(MonsterBehavior):
    """
    Patrol behavior - monster follows a set path.
    """

    def __init__(
        self,
        waypoints: List[Tuple[float, float]],
        patrol_speed: float = 0.3,
        waypoint_threshold: float = 0.5
    ):
        """
        Initialize patrol behavior.

        Args:
            waypoints: List of (x, y) positions to patrol between
            patrol_speed: Movement speed while patrolling
            waypoint_threshold: Distance to consider waypoint reached
        """
        self.waypoints = [np.array(wp) for wp in waypoints]
        self.patrol_speed = patrol_speed
        self.waypoint_threshold = waypoint_threshold
        self._current_waypoint_index = 0

    def update(
        self,
        monster: Entity,
        player: Optional[Entity],
        physics: PhysicsSystem,
        dt: float = 1.0
    ) -> None:
        """Follow patrol path."""
        if not monster.has_position() or len(self.waypoints) == 0:
            return

        current_pos = monster.position.as_array()
        target = self.waypoints[self._current_waypoint_index]

        # Check if reached current waypoint
        distance = np.linalg.norm(target - current_pos)
        if distance < self.waypoint_threshold:
            # Move to next waypoint
            self._current_waypoint_index = (
                self._current_waypoint_index + 1
            ) % len(self.waypoints)
            target = self.waypoints[self._current_waypoint_index]

        # Move toward target
        direction = target - current_pos
        length = np.linalg.norm(direction)
        if length > 0:
            direction = direction / length * self.patrol_speed * dt
            physics.move_entity(monster, direction)

    def get_name(self) -> str:
        return "patrol"

    def set_waypoints(self, waypoints: List[Tuple[float, float]]) -> None:
        """Update patrol waypoints."""
        self.waypoints = [np.array(wp) for wp in waypoints]
        self._current_waypoint_index = 0
