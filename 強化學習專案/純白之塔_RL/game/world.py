"""
World and room management - contains entities and coordinates game systems.
"""

from typing import List, Optional, Dict, Any, Callable
from dataclasses import dataclass, field
import numpy as np

from core.events import EventBus, GameEvent, EventType
from game.entity import Entity, EntityFactory
from game.physics import PhysicsSystem
from game.skills import SkillExecutor, SkillRegistry


@dataclass
class Room:
    """
    Represents a single game room/area.

    Attributes:
        id: Room identifier
        size: Room dimensions (square)
        spawn_points: Named spawn locations
    """
    id: str = "default"
    size: float = 10.0
    spawn_points: Dict[str, tuple] = field(default_factory=dict)

    def __post_init__(self):
        if not self.spawn_points:
            self.spawn_points = {
                'player': (5.0, 5.0),
                'monster': (8.0, 8.0),
                'blood_pack': (2.0, 8.0)
            }

    def random_position(self, margin: float = 1.0) -> np.ndarray:
        """Get a random position within the room."""
        return np.random.uniform(margin, self.size - margin, 2)


class GameWorld:
    """
    Main game world that coordinates all game systems.
    """

    def __init__(self, room: Optional[Room] = None):
        """
        Initialize the game world.

        Args:
            room: The room configuration (default room if None)
        """
        self.room = room or Room()
        self.event_bus = EventBus()

        # Initialize systems
        self.physics = PhysicsSystem(self.event_bus, self.room.size)
        self.skill_registry = SkillRegistry()
        self.skill_executor = SkillExecutor(self.event_bus, self.skill_registry)

        # Entity tracking
        self.entities: List[Entity] = []
        self.player: Optional[Entity] = None
        self.monsters: List[Entity] = []
        self.items: List[Entity] = []

        # World state
        self.tick_count: int = 0

    def reset(self) -> None:
        """Reset the world to initial state."""
        self.entities.clear()
        self.monsters.clear()
        self.items.clear()
        self.tick_count = 0

        EntityFactory.reset_id_counter()

        # Create player
        px, py = self.room.spawn_points.get('player', (1.0, 1.0))
        self.player = EntityFactory.create_player(px, py)
        self.entities.append(self.player)

        # Create monster
        mx, my = self.room.spawn_points.get('monster', (8.0, 8.0))
        monster = EntityFactory.create_monster(mx, my)
        self.monsters.append(monster)
        self.entities.append(monster)

        # Create blood pack
        bx, by = self.room.spawn_points.get('blood_pack', (2.0, 8.0))
        blood = EntityFactory.create_blood_pack(bx, by)
        self.items.append(blood)
        self.entities.append(blood)

        # Publish episode start
        self.event_bus.publish(GameEvent(
            EventType.EPISODE_START,
            data={'tick': 0}
        ))

    def tick(self) -> None:
        """Execute one game tick."""
        self.tick_count += 1

        # Publish tick event
        self.event_bus.publish(GameEvent(
            EventType.TICK,
            data={'tick': self.tick_count}
        ))

        # Process skill casting for player
        if self.player and self.player.has_skills() and self.player.skills.is_casting:
            event = self.skill_executor.tick(self.player, self.monsters)

            # Respawn killed monsters
            if event and event.event_type == EventType.SKILL_CAST_COMPLETE:
                self._respawn_killed_monsters()

        # Check item pickups
        if self.player:
            collected = self.physics.check_pickups(self.player, self.items)
            for item in collected:
                self._respawn_item(item)

        # Clean up dead entities
        self._cleanup_entities()

    def _respawn_killed_monsters(self) -> None:
        """Respawn any dead monsters at random positions."""
        for monster in self.monsters:
            if not monster.health.is_alive:
                # Reset health and position
                monster.health.current = monster.health.maximum
                monster.is_alive = True
                new_pos = self.room.random_position()
                monster.position.set_from_array(new_pos)

                self.event_bus.publish(GameEvent(
                    EventType.ENTITY_SPAWNED,
                    source_entity=monster,
                    data={'reason': 'respawn'}
                ))

    def _respawn_item(self, item: Entity) -> None:
        """Respawn a collected item at a random position."""
        # Create new blood pack
        new_pos = self.room.random_position()
        new_item = EntityFactory.create_blood_pack(new_pos[0], new_pos[1])
        self.items.append(new_item)
        self.entities.append(new_item)

        self.event_bus.publish(GameEvent(
            EventType.ITEM_SPAWNED,
            source_entity=new_item,
            data={'item_type': new_item.entity_type}
        ))

    def _cleanup_entities(self) -> None:
        """Remove dead entities from tracking lists."""
        self.entities = [e for e in self.entities if e.is_alive]
        self.items = [i for i in self.items if i.is_alive]

    def execute_action(self, action_discrete: int, action_continuous: float) -> str:
        """
        Execute a player action.

        Args:
            action_discrete: 0=forward, 1=left, 2=right, 3=cast
            action_continuous: Aim offset (only used for casting)

        Returns:
            Event string describing what happened
        """
        if self.player is None:
            return ""

        event = ""

        if action_discrete == 0:  # Move forward
            success = self.physics.move_forward(self.player, speed=0.6)
            if not success:
                event = "HIT WALL!"

        elif action_discrete == 1:  # Rotate left
            self.physics.rotate_entity(self.player, 0.4)

        elif action_discrete == 2:  # Rotate right
            self.physics.rotate_entity(self.player, -0.4)

        elif action_discrete == 3:  # Cast skill
            if self.player.skills.is_ready:
                self.skill_executor.start_cast(
                    self.player,
                    "basic_attack",
                    aim_offset=action_continuous
                )
                event = "CASTING..."

        return event

    def get_player_position(self) -> np.ndarray:
        """Get player position as numpy array."""
        if self.player and self.player.has_position():
            return self.player.position.as_array()
        return np.array([0.0, 0.0])

    def get_player_angle(self) -> float:
        """Get player facing angle."""
        if self.player and self.player.has_position():
            return self.player.position.angle
        return 0.0

    def get_monster_position(self) -> np.ndarray:
        """Get first monster position as numpy array."""
        if self.monsters and self.monsters[0].has_position():
            return self.monsters[0].position.as_array()
        return np.array([0.0, 0.0])

    def get_blood_pack_position(self) -> np.ndarray:
        """Get first blood pack position as numpy array."""
        alive_items = [i for i in self.items if i.is_alive]
        if alive_items and alive_items[0].has_position():
            return alive_items[0].position.as_array()
        return np.array([0.0, 0.0])

    def get_casting_progress(self) -> float:
        """Get player's current casting progress (0-1)."""
        if self.player and self.player.has_skills():
            return self.player.skills.wind_up_remaining / 4.0
        return 0.0

    def is_player_ready_to_cast(self) -> bool:
        """Check if player can cast a skill."""
        if self.player and self.player.has_skills():
            return self.player.skills.is_ready
        return False
