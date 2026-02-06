"""
Projectile system for ranged attacks.
Handles projectile spawning, movement, collision, and lifetime management.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum, auto
import numpy as np

from core.events import EventBus, GameEvent, EventType


class ProjectileType(Enum):
    """Types of projectiles in the game."""
    ARROW = auto()       # Bow attack - fast, lower damage
    MAGIC_BOLT = auto()  # Staff attack - slower, higher damage


# Projectile configuration by type
PROJECTILE_CONFIG: Dict[ProjectileType, Dict[str, Any]] = {
    ProjectileType.ARROW: {
        "speed": 0.8,
        "damage": 15.0,
        "radius": 0.3,
        "max_lifetime": 50,  # ticks
    },
    ProjectileType.MAGIC_BOLT: {
        "speed": 0.6,
        "damage": 20.0,
        "radius": 0.4,
        "max_lifetime": 60,  # ticks
    },
}


@dataclass
class Projectile:
    """
    Represents a projectile in the game world.

    Attributes:
        position: Current position as numpy array [x, y]
        direction: Movement direction as numpy array [dx, dy] (normalized)
        speed: Movement speed per tick
        damage: Damage dealt on hit
        owner_id: Entity ID of the shooter (to avoid friendly fire)
        projectile_type: Type of projectile (arrow/magic bolt)
        radius: Collision radius
        lifetime: Remaining lifetime in ticks
        alive: Whether the projectile is still active
    """
    position: np.ndarray
    direction: np.ndarray
    speed: float
    damage: float
    owner_id: int
    projectile_type: ProjectileType
    radius: float = 0.3
    lifetime: int = 50
    alive: bool = True
    id: int = field(default_factory=lambda: Projectile._next_id())

    _id_counter: int = 0

    @classmethod
    def _next_id(cls) -> int:
        cls._id_counter += 1
        return cls._id_counter

    @classmethod
    def reset_id_counter(cls) -> None:
        cls._id_counter = 0

    def update(self) -> None:
        """Update projectile position and lifetime."""
        if not self.alive:
            return

        # Move projectile
        self.position = self.position + self.direction * self.speed

        # Decrease lifetime
        self.lifetime -= 1
        if self.lifetime <= 0:
            self.alive = False

    def despawn(self) -> None:
        """Mark projectile as dead."""
        self.alive = False


class ProjectileManager:
    """
    Manages all projectiles in the game world.

    Responsibilities:
    - Spawn projectiles
    - Update projectile positions
    - Check collisions with entities
    - Check boundary collisions
    - Clean up dead projectiles
    """

    def __init__(self, event_bus: EventBus, world_size: float = 10.0):
        """
        Initialize the projectile manager.

        Args:
            event_bus: Event bus for publishing projectile events
            world_size: Size of the game world (for boundary checks)
        """
        self.event_bus = event_bus
        self.world_size = world_size
        self.projectiles: List[Projectile] = []

    def spawn_projectile(
        self,
        position: np.ndarray,
        direction: np.ndarray,
        owner_id: int,
        projectile_type: ProjectileType,
        damage_override: Optional[float] = None
    ) -> Projectile:
        """
        Spawn a new projectile.

        Args:
            position: Starting position [x, y]
            direction: Direction to fire (will be normalized)
            owner_id: ID of the entity that fired the projectile
            projectile_type: Type of projectile
            damage_override: Optional damage override

        Returns:
            The spawned projectile
        """
        config = PROJECTILE_CONFIG[projectile_type]

        # Normalize direction
        dir_norm = np.linalg.norm(direction)
        if dir_norm > 0:
            direction = direction / dir_norm
        else:
            direction = np.array([1.0, 0.0])  # Default to right

        damage = damage_override if damage_override is not None else config["damage"]

        projectile = Projectile(
            position=position.copy(),
            direction=direction,
            speed=config["speed"],
            damage=damage,
            owner_id=owner_id,
            projectile_type=projectile_type,
            radius=config["radius"],
            lifetime=config["max_lifetime"],
        )

        self.projectiles.append(projectile)

        self.event_bus.publish(GameEvent(
            EventType.PROJECTILE_SPAWNED,
            data={
                "projectile_id": projectile.id,
                "projectile_type": projectile_type.name,
                "owner_id": owner_id,
                "position": position.copy(),
            }
        ))

        return projectile

    def update(self, target_entity) -> List[Dict[str, Any]]:
        """
        Update all projectiles and check collisions.

        Args:
            target_entity: The entity to check collisions against (typically the player/agent)

        Returns:
            List of hit events (projectile hit target)
        """
        hits = []

        for projectile in self.projectiles:
            if not projectile.alive:
                continue

            # Update position
            projectile.update()

            if not projectile.alive:
                # Died from timeout
                self._publish_despawn(projectile, "timeout")
                continue

            # Check boundary collision
            if self._check_boundary_collision(projectile):
                projectile.despawn()
                self._publish_despawn(projectile, "boundary")
                continue

            # Check collision with target
            if target_entity and self._check_entity_collision(projectile, target_entity):
                hit_info = {
                    "projectile": projectile,
                    "damage": projectile.damage,
                    "projectile_type": projectile.projectile_type,
                }
                hits.append(hit_info)

                self.event_bus.publish(GameEvent(
                    EventType.PROJECTILE_HIT,
                    target_entity=target_entity,
                    data={
                        "projectile_id": projectile.id,
                        "damage": projectile.damage,
                        "projectile_type": projectile.projectile_type.name,
                    }
                ))

                projectile.despawn()

        # Clean up dead projectiles
        self._cleanup()

        return hits

    def _check_boundary_collision(self, projectile: Projectile) -> bool:
        """Check if projectile hit world boundary."""
        x, y = projectile.position
        r = projectile.radius

        return (x - r < 0 or x + r > self.world_size or
                y - r < 0 or y + r > self.world_size)

    def _check_entity_collision(self, projectile: Projectile, entity) -> bool:
        """Check if projectile collides with an entity."""
        if not entity.is_alive or not entity.has_position():
            return False

        # Don't hit the owner
        if entity.id == projectile.owner_id:
            return False

        entity_pos = entity.position.as_array()
        distance = np.linalg.norm(projectile.position - entity_pos)

        # Entity radius is assumed to be 0.5
        entity_radius = 0.5
        return distance < (projectile.radius + entity_radius)

    def _publish_despawn(self, projectile: Projectile, reason: str) -> None:
        """Publish projectile despawn event."""
        self.event_bus.publish(GameEvent(
            EventType.PROJECTILE_DESPAWNED,
            data={
                "projectile_id": projectile.id,
                "reason": reason,
                "position": projectile.position.copy(),
            }
        ))

    def _cleanup(self) -> None:
        """Remove dead projectiles from the list."""
        self.projectiles = [p for p in self.projectiles if p.alive]

    def get_active_projectiles(self) -> List[Projectile]:
        """Get all active projectiles."""
        return [p for p in self.projectiles if p.alive]

    def clear(self) -> None:
        """Clear all projectiles."""
        self.projectiles.clear()
        Projectile.reset_id_counter()
