"""
Player class - encapsulates player configuration and action execution.

Separates player-specific attributes (speed, skills) from the game world,
providing a clean interface: Agent -> Player -> World
"""

from typing import Dict, Optional, List, TYPE_CHECKING
from dataclasses import dataclass, field

from game.entity import Entity, EntityFactory
from game.components import Position, Health, Skills
from game.skills import SkillShapeType

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
    aim_actor_index: int = -1             # Index of aim actor (-1 = no aim, 0 = aim_missile, 1 = aim_hammer)

    # Damage/Range (basic parameters)
    damage: float = 100.0
    range: float = 6.0
    angle_tolerance: float = 0.4          # For simple cone/line skills

    # Shape type
    shape_type: SkillShapeType = SkillShapeType.CONE

    # Extra parameters for special skills
    # RING: {"inner_radius": 3.0, "outer_radius": 4.5}
    # RECTANGLE: {"length": 5.0, "width": 0.8, "tip_range_start": 4.0, "tip_damage": 50.0}
    # PROJECTILE: {"speed": 1.5, "radius": 0.5, "max_range": 15.0}
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
        """Add default skills if none provided."""
        if not self.skills:
            self.skills = {
                # Action 3: 外圈刮 (Outer Slash) - 環形 AOE，不需要瞄準
                "outer_slash": SkillConfig(
                    skill_id="outer_slash",
                    name="外圈刮",
                    cooldown_ticks=30,
                    wind_up_ticks=5,
                    can_move_during_wind_up=True,
                    requires_aim=False,
                    aim_actor_index=-1,
                    damage=30.0,
                    range=4.5,  # outer_radius
                    angle_tolerance=0.0,  # Not used for ring
                    shape_type=SkillShapeType.RING,
                    extra_params={"inner_radius": 3.0, "outer_radius": 4.5}
                ),
                # Action 4: 飛彈 (Missile) - 投射物，使用 aim_actor 0
                "missile": SkillConfig(
                    skill_id="missile",
                    name="飛彈",
                    cooldown_ticks=25,
                    wind_up_ticks=5,
                    can_move_during_wind_up=False,
                    requires_aim=True,
                    aim_actor_index=0,  # Uses aim_missile actor
                    damage=40.0,
                    range=15.0,  # max_range
                    angle_tolerance=0.0,  # Not used for projectile
                    shape_type=SkillShapeType.PROJECTILE,
                    extra_params={"speed": 1.5, "radius": 0.5, "max_range": 15.0}
                ),
                # Action 5: 鐵錘 (Hammer) - 長方形範圍，使用 aim_actor 1
                "hammer": SkillConfig(
                    skill_id="hammer",
                    name="鐵錘",
                    cooldown_ticks=35,
                    wind_up_ticks=5,
                    can_move_during_wind_up=True,
                    requires_aim=True,
                    aim_actor_index=1,  # Uses aim_hammer actor
                    damage=25.0,
                    range=5.0,  # length
                    angle_tolerance=0.0,  # Not used for rectangle
                    shape_type=SkillShapeType.RECTANGLE,
                    extra_params={
                        "length": 5.0,
                        "width": 0.8,
                        "tip_range_start": 4.0,
                        "tip_damage": 50.0
                    }
                ),
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

    # Mapping from discrete action to skill ID
    SKILL_ACTION_MAP = {
        4: "outer_slash",  # 外圈刮
        5: "missile",      # 飛彈
        6: "hammer",       # 鐵錘
    }

    def execute_action(
        self,
        action_discrete: int,
        aim_values: List[float],
        physics: 'PhysicsSystem',
        skill_executor: 'SkillExecutor'
    ) -> str:
        """
        Execute a player action.

        Args:
            action_discrete: 0=forward, 1=backward, 2=left, 3=right, 4=outer_slash, 5=missile, 6=hammer
            aim_values: List of aim values for each aim actor
            physics: Physics system for movement
            skill_executor: Skill executor for casting

        Returns:
            Event string describing what happened
        """
        if self.entity is None:
            return ""

        event = ""

        # Check if movement is blocked during wind-up
        is_movement_action = action_discrete in (0, 1, 2, 3)
        if is_movement_action and self._is_movement_blocked():
            return "WIND-UP..."  # Can't move during wind-up

        if action_discrete == 0:  # Move forward
            success = physics.move_forward(self.entity, speed=self.config.move_speed)
            if not success:
                event = "HIT WALL!"

        elif action_discrete == 1:  # Move backward
            success = physics.move_backward(self.entity, speed=self.config.move_speed)
            if not success:
                event = "HIT WALL!"

        elif action_discrete == 2:  # Rotate left
            physics.rotate_entity(self.entity, self.config.turn_speed)

        elif action_discrete == 3:  # Rotate right
            physics.rotate_entity(self.entity, -self.config.turn_speed)

        elif action_discrete in self.SKILL_ACTION_MAP:  # Cast skill (4, 5, 6)
            if self.entity.skills.is_ready:
                skill_id = self.SKILL_ACTION_MAP[action_discrete]
                skill_config = self.get_skill_config(skill_id)
                if skill_config:
                    # Get aim offset from the appropriate actor
                    aim_offset = 0.0
                    if skill_config.requires_aim and skill_config.aim_actor_index >= 0:
                        if skill_config.aim_actor_index < len(aim_values):
                            aim_offset = aim_values[skill_config.aim_actor_index]

                    skill_executor.start_cast(
                        self.entity,
                        skill_config,
                        aim_offset=aim_offset
                    )
                    event = f"CASTING {skill_config.name}..."

        return event

    def _is_movement_blocked(self) -> bool:
        """
        Check if movement is blocked (e.g., during wind-up of a skill
        that doesn't allow movement).

        Returns:
            True if movement is currently blocked
        """
        if not self.entity or not self.entity.has_skills():
            return False

        skills = self.entity.skills
        if not skills.is_casting:
            return False

        # Get the current skill config
        current_skill_id = skills.current_skill
        if current_skill_id:
            skill_config = self.get_skill_config(current_skill_id)
            if skill_config and not skill_config.can_move_during_wind_up:
                return True

        return False

    def get_skill_config(self, skill_id: str) -> Optional[SkillConfig]:
        """
        Get configuration for a specific skill.

        Args:
            skill_id: The skill identifier

        Returns:
            SkillConfig if found, None otherwise
        """
        return self.config.skills.get(skill_id)
