"""
Skill system - defines skills, handles casting, wind-up, and damage calculation.
"""

from typing import Dict, List, Optional, Callable, Any, Union, TYPE_CHECKING
from dataclasses import dataclass, field
import numpy as np

from core.events import EventBus, GameEvent, EventType
from game.entity import Entity

if TYPE_CHECKING:
    from game.player import SkillConfig


@dataclass
class SkillDefinition:
    """
    Definition of a skill's properties.

    Attributes:
        id: Unique skill identifier
        name: Display name
        wind_up_ticks: Ticks needed to cast
        range: Maximum effective range
        angle_tolerance: Angular accuracy needed (radians)
        damage: Base damage dealt
        tags: Skill category tags
    """
    id: str
    name: str
    wind_up_ticks: int = 4
    range: float = 6.0
    angle_tolerance: float = 0.4  # radians
    damage: float = 100.0
    tags: List[str] = field(default_factory=list)


class SkillRegistry:
    """
    Registry of all available skills.
    """

    def __init__(self):
        self._skills: Dict[str, SkillDefinition] = {}
        self._register_default_skills()

    def _register_default_skills(self) -> None:
        """Register the default skill set."""
        self.register(SkillDefinition(
            id="basic_attack",
            name="Basic Attack",
            wind_up_ticks=4,
            range=6.0,
            angle_tolerance=0.4,
            damage=100.0,
            tags=["attack", "ranged"]
        ))

    def register(self, skill: SkillDefinition) -> None:
        """
        Register a skill definition.

        Args:
            skill: The skill to register
        """
        self._skills[skill.id] = skill

    def get(self, skill_id: str) -> Optional[SkillDefinition]:
        """
        Get a skill definition by ID.

        Args:
            skill_id: The skill identifier

        Returns:
            The skill definition or None
        """
        return self._skills.get(skill_id)

    def list_skills(self) -> List[str]:
        """Get list of all registered skill IDs."""
        return list(self._skills.keys())


class SkillExecutor:
    """
    Handles skill casting, wind-up timing, and hit detection.
    """

    def __init__(self, event_bus: EventBus, skill_registry: SkillRegistry):
        """
        Initialize the skill executor.

        Args:
            event_bus: Event bus for skill events
            skill_registry: Registry of skill definitions
        """
        self.event_bus = event_bus
        self.skill_registry = skill_registry

    def start_cast(
        self,
        caster: Entity,
        skill: Union[str, 'SkillConfig'],
        aim_offset: float = 0.0
    ) -> bool:
        """
        Begin casting a skill.

        Args:
            caster: The entity casting the skill
            skill: Either a skill_id string (legacy) or a SkillConfig object
            aim_offset: Offset from current facing direction

        Returns:
            True if cast started successfully
        """
        if not caster.has_skills() or not caster.has_position():
            return False

        if caster.skills.is_casting:
            return False

        # Handle both legacy skill_id and new SkillConfig
        if isinstance(skill, str):
            # Legacy: lookup in registry
            skill_def = self.skill_registry.get(skill)
            if skill_def is None:
                return False
            skill_id = skill
            wind_up_ticks = skill_def.wind_up_ticks
            skill_range = skill_def.range
            angle_tolerance = skill_def.angle_tolerance
            damage = skill_def.damage
        else:
            # New: use SkillConfig directly
            skill_id = skill.skill_id
            wind_up_ticks = skill.wind_up_ticks
            skill_range = skill.range
            angle_tolerance = skill.angle_tolerance
            damage = skill.damage

        aim_angle = caster.position.angle + np.clip(aim_offset, -0.5, 0.5)
        caster.skills.start_cast(
            skill_id, wind_up_ticks, aim_angle,
            skill_range=skill_range,
            angle_tolerance=angle_tolerance,
            damage=damage
        )

        self.event_bus.publish(GameEvent(
            EventType.SKILL_CAST_START,
            source_entity=caster,
            data={
                'skill_id': skill_id,
                'aim_angle': aim_angle,
                'wind_up_ticks': wind_up_ticks
            }
        ))

        return True

    def tick(self, caster: Entity, targets: List[Entity]) -> Optional[GameEvent]:
        """
        Process one tick of skill execution.

        Args:
            caster: The casting entity
            targets: Potential target entities

        Returns:
            Hit or miss event if skill completed, None otherwise
        """
        if not caster.has_skills():
            return None

        # Read skill parameters BEFORE tick() clears them
        skill_id = caster.skills.current_skill or "basic_attack"
        skill_range = caster.skills.current_skill_range
        angle_tolerance = caster.skills.current_skill_angle_tolerance
        damage = caster.skills.current_skill_damage

        completed = caster.skills.tick()

        if not completed:
            return None

        # Find hit targets using stored parameters
        hit_target = self._check_hit(caster, targets, skill_range, angle_tolerance)

        if hit_target is not None:
            event = GameEvent(
                EventType.SKILL_CAST_COMPLETE,
                source_entity=caster,
                target_entity=hit_target,
                data={
                    'skill_id': skill_id,
                    'damage': damage
                }
            )

            # Apply damage if target has health
            if hit_target.has_health():
                hit_target.health.damage(damage)

                if not hit_target.health.is_alive:
                    hit_target.despawn()
                    self.event_bus.publish(GameEvent(
                        EventType.ENTITY_KILLED,
                        source_entity=caster,
                        target_entity=hit_target,
                        data={'skill_id': skill_id}
                    ))

            self.event_bus.publish(event)
            return event
        else:
            event = GameEvent(
                EventType.SKILL_MISSED,
                source_entity=caster,
                data={'skill_id': skill_id}
            )
            self.event_bus.publish(event)
            return event

    def _check_hit(
        self,
        caster: Entity,
        targets: List[Entity],
        skill_range: float,
        angle_tolerance: float
    ) -> Optional[Entity]:
        """
        Check if skill hits any target.

        Args:
            caster: The casting entity
            targets: Potential targets
            skill_range: Maximum range of the skill
            angle_tolerance: Angular accuracy tolerance

        Returns:
            The hit target or None
        """
        if not caster.has_position() or not caster.has_skills():
            return None

        aim_angle = caster.skills.aim_angle
        caster_pos = caster.position.as_array()

        for target in targets:
            if not target.is_alive or not target.has_position():
                continue

            if not target.has_tag("targetable"):
                continue

            target_pos = target.position.as_array()
            vec = target_pos - caster_pos
            distance = np.linalg.norm(vec)

            if distance > skill_range:
                continue

            angle_to_target = np.arctan2(vec[1], vec[0])
            angle_diff = np.arctan2(
                np.sin(angle_to_target - aim_angle),
                np.cos(angle_to_target - aim_angle)
            )

            if abs(angle_diff) < angle_tolerance:
                return target

        return None

    def process_casting_entity(self, caster: Entity,
                               targets: List[Entity]) -> Optional[GameEvent]:
        """
        Process a single tick for a casting entity.
        Convenience method that combines tick logic.

        Args:
            caster: Entity that may be casting
            targets: List of potential targets

        Returns:
            Event if skill completed this tick
        """
        if not caster.has_skills() or not caster.skills.is_casting:
            return None

        return self.tick(caster, targets)
