"""
Player class - encapsulates player configuration and action execution.

Separates player-specific attributes (speed, skills) from the game world,
providing a clean interface: Agent -> Player -> World
"""

from typing import Dict, Optional, TYPE_CHECKING
from dataclasses import dataclass, field

from game.entity import Entity, EntityFactory
from game.components import Position, Health, Skills

if TYPE_CHECKING:
    from game.physics import PhysicsSystem
    from game.skills import SkillExecutor


@dataclass
class SkillConfig:
    """
    Configuration for a single skill.

    Basic parameters are defined as fields.
    Skill-specific parameters (e.g., inner_radius for ring AOE)
    should be stored in extra_params.
    """
    skill_id: str
    name: str = ""

    # Cooldown
    cooldown_ticks: int = 30              # Individual CD in ticks

    # Wind-up
    wind_up_ticks: int = 4
    can_move_during_wind_up: bool = True  # Can move while casting

    # Aiming
    requires_aim: bool = False            # Needs aim input
    aim_actor_count: int = 0              # Number of aim actors (0-2)

    # Damage/Range (basic parameters)
    damage: float = 100.0
    range: float = 6.0
    angle_tolerance: float = 0.4          # For simple cone/line skills

    # Extra parameters for special skills
    # e.g., {"inner_radius": 3.0, "outer_radius": 4.5} for ring AOE
    extra_params: Dict[str, float] = field(default_factory=dict)


@dataclass
class PlayerConfig:
    """
    Player configuration.

    Contains all player-specific attributes that were previously
    hardcoded in world.py and skills.py.
    """
    # Movement attributes
    move_speed: float = 0.6
    turn_speed: float = 0.4

    # Health attributes
    max_health: float = 100.0

    # Skill configurations
    skills: Dict[str, SkillConfig] = field(default_factory=dict)

    def __post_init__(self):
        """Add default skill if none provided."""
        if not self.skills:
            self.skills = {
                "basic_attack": SkillConfig(
                    skill_id="basic_attack",
                    name="Basic Attack",
                    cooldown_ticks=0,  # No cooldown for basic attack
                    wind_up_ticks=4,
                    can_move_during_wind_up=False,
                    requires_aim=True,
                    aim_actor_count=1,
                    damage=100.0,
                    range=6.0,
                    angle_tolerance=0.4
                )
            }


# Default configuration for convenience
DEFAULT_PLAYER_CONFIG = PlayerConfig()


class Player:
    """
    Player class - encapsulates player configuration and action execution.

    Wraps an Entity and provides high-level action methods.
    Uses property delegation for backward compatibility with existing code
    that accesses player.position, player.health, etc.
    """

    def __init__(self, config: Optional[PlayerConfig] = None):
        """
        Initialize player with configuration.

        Args:
            config: Player configuration (uses default if None)
        """
        self.config = config or PlayerConfig()
        self.entity: Optional[Entity] = None

    # ===== Property Delegation (Backward Compatibility) =====

    @property
    def position(self) -> Position:
        """Get position component from underlying entity."""
        if self.entity is None:
            raise RuntimeError("Player entity not spawned")
        return self.entity.position

    @property
    def health(self) -> Health:
        """Get health component from underlying entity."""
        if self.entity is None:
            raise RuntimeError("Player entity not spawned")
        return self.entity.health

    @property
    def skills(self) -> Skills:
        """Get skills component from underlying entity."""
        if self.entity is None:
            raise RuntimeError("Player entity not spawned")
        return self.entity.skills

    @property
    def is_alive(self) -> bool:
        """Check if player is alive."""
        return self.entity.is_alive if self.entity else False

    @property
    def max_health(self) -> float:
        """Get maximum health from config."""
        return self.config.max_health

    @property
    def current_health(self) -> float:
        """Get current health."""
        if self.entity and self.entity.has_health():
            return self.entity.health.current
        return 0.0

    @property
    def health_percentage(self) -> float:
        """Get health as percentage (0.0 to 1.0)."""
        if self.entity and self.entity.has_health():
            return self.entity.health.percentage
        return 0.0

    @property
    def id(self) -> int:
        """Get entity ID."""
        if self.entity is None:
            raise RuntimeError("Player entity not spawned")
        return self.entity.id

    def has_position(self) -> bool:
        """Check if player has position component."""
        return self.entity.has_position() if self.entity else False

    def has_health(self) -> bool:
        """Check if player has health component."""
        return self.entity.has_health() if self.entity else False

    def has_skills(self) -> bool:
        """Check if player has skills component."""
        return self.entity.has_skills() if self.entity else False

    def has_tag(self, tag: str) -> bool:
        """Check if player has a specific tag."""
        return self.entity.has_tag(tag) if self.entity else False

    def despawn(self) -> None:
        """Mark player as no longer alive."""
        if self.entity:
            self.entity.despawn()

    # ===== Core Methods =====

    def spawn(self, x: float, y: float) -> Entity:
        """
        Spawn the player entity at specified position.

        Args:
            x: Initial X position
            y: Initial Y position

        Returns:
            The created Entity
        """
        self.entity = Entity("player")
        self.entity.position = Position(x, y, angle=0.0)
        self.entity.health = Health(
            current=self.config.max_health,
            maximum=self.config.max_health
        )
        self.entity.skills = Skills()
        self.entity.add_tag("player")
        self.entity.add_tag("controllable")

        return self.entity

    def execute_action(
        self,
        action_discrete: int,
        action_continuous: float,
        physics: 'PhysicsSystem',
        skill_executor: 'SkillExecutor'
    ) -> str:
        """
        Execute a player action.

        Args:
            action_discrete: 0=forward, 1=left, 2=right, 3=cast
            action_continuous: Aim offset (only used for casting)
            physics: Physics system for movement
            skill_executor: Skill executor for casting

        Returns:
            Event string describing what happened
        """
        if self.entity is None:
            return ""

        event = ""

        if action_discrete == 0:  # Move forward
            success = physics.move_forward(self.entity, speed=self.config.move_speed)
            if not success:
                event = "HIT WALL!"

        elif action_discrete == 1:  # Rotate left
            physics.rotate_entity(self.entity, self.config.turn_speed)

        elif action_discrete == 2:  # Rotate right
            physics.rotate_entity(self.entity, -self.config.turn_speed)

        elif action_discrete == 3:  # Cast skill
            if self.entity.skills.is_ready:
                # Use default skill for now
                skill_config = self.get_skill_config("basic_attack")
                if skill_config:
                    skill_executor.start_cast(
                        self.entity,
                        skill_config,
                        aim_offset=action_continuous
                    )
                    event = "CASTING..."

        return event

    def get_skill_config(self, skill_id: str) -> Optional[SkillConfig]:
        """
        Get configuration for a specific skill.

        Args:
            skill_id: The skill identifier

        Returns:
            SkillConfig if found, None otherwise
        """
        return self.config.skills.get(skill_id)
