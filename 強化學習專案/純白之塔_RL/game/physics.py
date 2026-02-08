"""
Physics system - handles movement, collision detection, and boundary checking.
"""

from typing import Optional, List, Tuple
import numpy as np

from core.events import EventBus, GameEvent, EventType
from game.entity import Entity
from game.components import Position


class PhysicsSystem:
    """
    Handles all physics-related operations including movement,
    collision detection, and boundary checking.
    """

    def __init__(self, event_bus: EventBus, world_size: float = 10.0):
        """
        Initialize the physics system.

        Args:
            event_bus: Event bus for publishing physics events
            world_size: Size of the square world (0 to world_size)
        """
        self.event_bus = event_bus
        self.world_size = world_size
        self.boundary_min = 0.0
        self.boundary_max = world_size - 1.0

    def move_entity(self, entity: Entity, direction: np.ndarray) -> bool:
        """
        Move an entity in a direction, respecting boundaries.

        Args:
            entity: The entity to move
            direction: Movement vector [dx, dy]

        Returns:
            True if movement was successful (no wall collision)
        """
        if not entity.has_position():
            return False

        pos = entity.position
        old_pos = pos.as_array()
        intended_pos = old_pos + direction

        # Clamp to boundaries
        new_pos = np.clip(intended_pos, self.boundary_min, self.boundary_max)
        pos.set_from_array(new_pos)

        # Check for wall collision
        hit_wall = not np.allclose(intended_pos, new_pos)

        # Publish movement event
        self.event_bus.publish(GameEvent(
            EventType.ENTITY_MOVED,
            source_entity=entity,
            data={
                'old_position': old_pos,
                'new_position': new_pos,
                'intended_position': intended_pos
            }
        ))

        # Publish wall collision event if applicable
        if hit_wall:
            self.event_bus.publish(GameEvent(
                EventType.ENTITY_HIT_WALL,
                source_entity=entity,
                data={
                    'position': new_pos,
                    'intended_position': intended_pos
                }
            ))

        return not hit_wall

    def move_forward(self, entity: Entity, speed: float = 0.6) -> bool:
        """
        Move entity forward based on its facing angle.

        Args:
            entity: The entity to move
            speed: Movement speed

        Returns:
            True if movement was successful
        """
        if not entity.has_position():
            return False

        angle = entity.position.angle
        direction = np.array([np.cos(angle), np.sin(angle)]) * speed
        return self.move_entity(entity, direction)

    def move_backward(self, entity: Entity, speed: float = 0.6) -> bool:
        """
        Move entity backward based on its facing angle.

        Args:
            entity: The entity to move
            speed: Movement speed

        Returns:
            True if movement was successful
        """
        if not entity.has_position():
            return False

        angle = entity.position.angle
        direction = np.array([np.cos(angle), np.sin(angle)]) * (-speed)
        return self.move_entity(entity, direction)

    def rotate_entity(self, entity: Entity, angle_delta: float) -> None:
        """
        Rotate an entity by a given angle.

        Args:
            entity: The entity to rotate
            angle_delta: Angle change in radians (positive = counter-clockwise)
        """
        if not entity.has_position():
            return

        entity.position.angle += angle_delta

        self.event_bus.publish(GameEvent(
            EventType.ENTITY_ROTATED,
            source_entity=entity,
            data={'angle_delta': angle_delta, 'new_angle': entity.position.angle}
        ))

    def check_collision(self, entity1: Entity, entity2: Entity,
                       threshold: float = 1.0) -> bool:
        """
        Check if two entities are colliding (within threshold distance).

        Args:
            entity1: First entity
            entity2: Second entity
            threshold: Collision distance threshold

        Returns:
            True if entities are colliding
        """
        if not entity1.has_position() or not entity2.has_position():
            return False

        distance = entity1.position.distance_to(entity2.position)
        return distance < threshold

    def check_pickups(self, collector: Entity, items: List[Entity],
                     threshold: float = 1.0) -> List[Entity]:
        """
        Check for item pickups and publish collection events.

        Args:
            collector: Entity that can collect items
            items: List of collectible items
            threshold: Collection distance threshold

        Returns:
            List of collected items
        """
        collected = []

        for item in items:
            if not item.is_alive:
                continue

            if self.check_collision(collector, item, threshold):
                item.despawn()
                collected.append(item)

                self.event_bus.publish(GameEvent(
                    EventType.ITEM_COLLECTED,
                    source_entity=collector,
                    target_entity=item,
                    data={'item_type': item.entity_type}
                ))

        return collected

    def get_distance(self, entity1: Entity, entity2: Entity) -> float:
        """
        Get distance between two entities.

        Args:
            entity1: First entity
            entity2: Second entity

        Returns:
            Euclidean distance, or inf if either lacks position
        """
        if not entity1.has_position() or not entity2.has_position():
            return float('inf')

        return entity1.position.distance_to(entity2.position)

    def get_angle_between(self, from_entity: Entity, to_entity: Entity) -> float:
        """
        Get angle from one entity to another.

        Args:
            from_entity: Source entity
            to_entity: Target entity

        Returns:
            Angle in radians, or 0 if either lacks position
        """
        if not from_entity.has_position() or not to_entity.has_position():
            return 0.0

        return from_entity.position.angle_to(to_entity.position)

    def get_relative_angle(self, from_entity: Entity, to_entity: Entity) -> float:
        """
        Get relative angle from entity's facing direction to target.

        Args:
            from_entity: Source entity (with facing direction)
            to_entity: Target entity

        Returns:
            Relative angle in radians (-pi to pi), 0 means directly ahead
        """
        if not from_entity.has_position() or not to_entity.has_position():
            return 0.0

        absolute_angle = from_entity.position.angle_to(to_entity.position)
        facing_angle = from_entity.position.angle

        # Normalize to -pi to pi
        relative = np.arctan2(
            np.sin(absolute_angle - facing_angle),
            np.cos(absolute_angle - facing_angle)
        )
        return relative

    def get_distance_to_nearest_wall(self, entity: Entity) -> float:
        """
        Get distance to the nearest wall.

        Args:
            entity: The entity to check

        Returns:
            Distance to nearest wall, normalized by world size
        """
        if not entity.has_position():
            return 1.0

        pos = entity.position
        distances = [
            pos.x - self.boundary_min,      # Left wall
            pos.y - self.boundary_min,      # Bottom wall
            self.boundary_max - pos.x,      # Right wall
            self.boundary_max - pos.y       # Top wall
        ]
        return min(distances) / self.world_size
