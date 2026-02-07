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
from game.projectile import ProjectileManager, ProjectileType
from game.player import Player, PlayerConfig, DEFAULT_PLAYER_CONFIG
from game.behaviors import (
    BehaviorRegistry,
    MonsterActionExecutor,
    StationaryBehavior,
    BerserkerBehavior,
    HitAndRunBehavior,
    OrbitMeleeBehavior,
    OrbitRangedBehavior,
)


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
                'player': (8.0, 5.0),
                'monsters': [(2.0, 2.0), (8.0, 2.0), (2.0, 8.0), (8.0, 8.0)],
                'blood_pack': (5.0, 8.0)
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
        self.monster_action_executor = MonsterActionExecutor(self.room.size)
        self.projectile_manager = ProjectileManager(self.event_bus, self.room.size)

        # Entity tracking
        self.entities: List[Entity] = []
        self.player: Optional[Player] = None
        self.monsters: List[Entity] = []
        self.items: List[Entity] = []

        # Player configuration (can be customized before reset)
        self.player_config: PlayerConfig = DEFAULT_PLAYER_CONFIG

        # World state
        self.tick_count: int = 0

    def reset(self) -> None:
        """Reset the world to initial state."""
        self.entities.clear()
        self.monsters.clear()
        self.items.clear()
        self.tick_count = 0

        EntityFactory.reset_id_counter()
        self.projectile_manager.clear()

        # Create player
        px, py = self.room.spawn_points.get('player', (1.0, 1.0))
        self.player = Player(self.player_config)
        player_entity = self.player.spawn(px, py)
        self.entities.append(player_entity)

        # Create monsters (4 monsters with different behaviors)
        monster_spawns = self.room.spawn_points.get(
            'monsters', [(2.0, 2.0), (8.0, 2.0), (2.0, 8.0), (8.0, 8.0)]
        )

        # Default behaviors for each monster (simulating different player types)
        default_behaviors = [
            BerserkerBehavior(),                              # 狂戰士：正面衝鋒
            HitAndRunBehavior(),                              # 偷傷害：打一下就跑
            OrbitMeleeBehavior(clockwise=True),               # 近戰繞圈
            OrbitRangedBehavior(weapon_type="bow"),           # 遠程繞圈（弓箭手）
        ]

        for i, (mx, my) in enumerate(monster_spawns):
            behavior = default_behaviors[i] if i < len(default_behaviors) else StationaryBehavior()
            monster = EntityFactory.create_monster(mx, my, movement_behavior=behavior)
            self.monsters.append(monster)
            self.entities.append(monster)

        # Create blood pack
        bx, by = self.room.spawn_points.get('blood_pack', (5.0, 8.0))
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

        # Process monster movements
        self._update_monster_movements()

        # Process skill casting for player
        if self.player and self.player.has_skills() and self.player.skills.is_casting:
            self.skill_executor.tick(self.player.entity, self.monsters)

        # Check item pickups
        if self.player:
            collected = self.physics.check_pickups(self.player.entity, self.items)
            for item in collected:
                self._process_item_pickup(item)
                self._respawn_item(item)

        # Update projectiles and check collisions with player
        self._update_projectiles()

        # Clean up dead entities
        self._cleanup_entities()

    def _update_monster_movements(self) -> None:
        """Update all monster behaviors (movement + turning + attack)."""
        for monster in self.monsters:
            if not monster.is_alive or not monster.has_movement_behavior():
                continue

            behavior = monster.movement_behavior

            # 1. Behavior decides the action
            action = behavior.decide_action(monster, self)

            # 2. Execute the action
            result = self.monster_action_executor.execute(monster, action, behavior)

            # 3. Handle attack results
            if result.get("attacked"):
                self._process_monster_attack(monster, behavior, result)

    def _process_monster_attack(self, monster: Entity, behavior, result: dict) -> None:
        """Process monster attack results."""
        if not self.player or not self.player.is_alive:
            return

        # Check if player is in range
        if not monster.has_position() or not self.player.has_position():
            return

        monster_pos = monster.position.as_array()
        player_pos = self.player.position.as_array()
        distance = np.linalg.norm(player_pos - monster_pos)

        # Check if attack hits (must be in range and facing target)
        if distance > behavior.attack_range:
            return

        angle_diff = behavior._get_angle_to_target(
            monster_pos, monster.position.angle, player_pos
        )
        if not behavior._is_facing_target(angle_diff, tolerance=0.5):
            return

        damage = result.get("attack_damage", 0)

        # Check if this is a ranged attack (OrbitRangedBehavior)
        if hasattr(behavior, 'weapon_type'):
            # Ranged attack: spawn projectile
            weapon_type = behavior.weapon_type
            if weapon_type == "bow":
                proj_type = ProjectileType.ARROW
            else:  # staff
                proj_type = ProjectileType.MAGIC_BOLT

            # Direction from monster to player
            direction = player_pos - monster_pos
            self.projectile_manager.spawn_projectile(
                position=monster_pos,
                direction=direction,
                owner_id=monster.id,
                projectile_type=proj_type,
                damage_override=damage
            )
        else:
            # Melee attack: direct damage
            if self.player.has_health():
                self.player.health.damage(damage)

                self.event_bus.publish(GameEvent(
                    EventType.DAMAGE_TAKEN,
                    source_entity=monster,
                    target_entity=self.player.entity,
                    data={"damage": damage}
                ))

                # Check player death
                if not self.player.health.is_alive:
                    self.player.despawn()
                    self.event_bus.publish(GameEvent(
                        EventType.AGENT_DIED,
                        source_entity=monster,
                        target_entity=self.player.entity
                    ))

    def _process_item_pickup(self, item: Entity) -> None:
        """Process the effects of picking up an item."""
        if not self.player or not self.player.is_alive:
            return

        # Handle blood pack healing
        if item.has_tag("blood_pack"):
            heal_amount = item.get_component("heal_amount") or 30.0

            if self.player.has_health():
                self.player.health.heal(heal_amount)

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
        # Check if any monsters were alive before cleanup
        had_monsters = len(self.monsters) > 0

        self.entities = [e for e in self.entities if e.is_alive]
        self.monsters = [m for m in self.monsters if m.is_alive]
        self.items = [i for i in self.items if i.is_alive]

        # Check if all monsters are now dead (victory condition)
        if had_monsters and len(self.monsters) == 0:
            self.event_bus.publish(GameEvent(
                EventType.ALL_ENEMIES_DEAD,
                data={'tick': self.tick_count}
            ))

    def _update_projectiles(self) -> None:
        """Update all projectiles and handle collisions with player."""
        if not self.player or not self.player.is_alive:
            return

        hits = self.projectile_manager.update(self.player.entity)

        for hit_info in hits:
            damage = hit_info["damage"]

            if self.player.has_health():
                self.player.health.damage(damage)

                self.event_bus.publish(GameEvent(
                    EventType.DAMAGE_TAKEN,
                    source_entity=None,  # Projectile source
                    target_entity=self.player.entity,
                    data={"damage": damage, "source": "projectile"}
                ))

                # Check player death
                if not self.player.health.is_alive:
                    self.player.despawn()
                    self.event_bus.publish(GameEvent(
                        EventType.AGENT_DIED,
                        target_entity=self.player.entity,
                        data={"source": "projectile"}
                    ))

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

        return self.player.execute_action(
            action_discrete,
            action_continuous,
            self.physics,
            self.skill_executor
        )

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

    def get_alive_monster_positions(self) -> List[np.ndarray]:
        """Get positions of all alive monsters."""
        positions = []
        for monster in self.monsters:
            if monster.is_alive and monster.has_position():
                positions.append(monster.position.as_array())
        return positions

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

    def get_player_health_percentage(self) -> float:
        """Get player health as percentage (0-1)."""
        if self.player:
            return self.player.health_percentage
        return 0.0

    def get_player_current_health(self) -> float:
        """Get player current health."""
        if self.player:
            return self.player.current_health
        return 0.0

    def get_player_max_health(self) -> float:
        """Get player max health."""
        if self.player:
            return self.player.max_health
        return 0.0

    def is_player_ready_to_cast(self) -> bool:
        """Check if player can cast a skill."""
        if self.player and self.player.has_skills():
            return self.player.skills.is_ready
        return False

    def get_active_projectiles(self) -> List:
        """Get all active projectiles for rendering."""
        return self.projectile_manager.get_active_projectiles()
