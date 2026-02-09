"""
Skill system - defines skills, handles casting, wind-up, and damage calculation.
"""

from typing import Dict, List, Optional, Callable, Any, Union, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from core.events import EventBus, GameEvent, EventType
from game.entity import Entity

if TYPE_CHECKING:
    from game.player import SkillConfig


class SkillShapeType(Enum):
    """技能範圍形狀類型"""
    CONE = "cone"           # 扇形 (現有 basic_attack)
    RING = "ring"           # 環形 (外圈刮)
    RECTANGLE = "rectangle" # 長方形 (鐵錘, 靈魂爪, 靈魂掌)
    PROJECTILE = "projectile" # 投射物 (飛彈)
    DASH = "dash"           # 位移 (閃現)


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

    def __init__(self, event_bus: EventBus, skill_registry: SkillRegistry, projectile_manager=None):
        """
        Initialize the skill executor.

        Args:
            event_bus: Event bus for skill events
            skill_registry: Registry of skill definitions
            projectile_manager: Optional projectile manager for missile skills
        """
        self.event_bus = event_bus
        self.skill_registry = skill_registry
        self.projectile_manager = projectile_manager

    def set_projectile_manager(self, projectile_manager) -> None:
        """Set the projectile manager (called by GameWorld after init)."""
        self.projectile_manager = projectile_manager

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
            shape_type = SkillShapeType.CONE
            extra_params = {}
            cooldown_ticks = 0
        else:
            # New: use SkillConfig directly
            skill_id = skill.skill_id
            wind_up_ticks = skill.wind_up_ticks
            skill_range = skill.range
            angle_tolerance = skill.angle_tolerance
            damage = skill.damage
            shape_type = skill.shape_type
            extra_params = skill.extra_params
            cooldown_ticks = skill.cooldown_ticks

        aim_angle = caster.position.angle + np.clip(aim_offset, -0.5, 0.5)
        caster.skills.start_cast(
            skill_id, wind_up_ticks, aim_angle,
            skill_range=skill_range,
            angle_tolerance=angle_tolerance,
            damage=damage,
            shape_type=shape_type.value,
            extra_params=extra_params,
            cooldown_ticks=cooldown_ticks
        )

        self.event_bus.publish(GameEvent(
            EventType.SKILL_CAST_START,
            source_entity=caster,
            data={
                'skill_id': skill_id,
                'aim_angle': aim_angle,
                'wind_up_ticks': wind_up_ticks,
                'shape_type': shape_type.value
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
        skill_id = caster.skills.current_skill or "outer_slash"
        skill_range = caster.skills.current_skill_range
        angle_tolerance = caster.skills.current_skill_angle_tolerance
        damage = caster.skills.current_skill_damage
        shape_type_str = caster.skills.current_skill_shape_type or "cone"
        extra_params = caster.skills.current_skill_extra_params or {}

        completed = caster.skills.tick()

        if not completed:
            return None

        # Convert shape type string to enum
        shape_type = SkillShapeType(shape_type_str)

        # Handle projectile skills differently
        if shape_type == SkillShapeType.PROJECTILE:
            return self._handle_projectile_skill(caster, skill_id, damage, extra_params)

        # Handle dash skills differently (no damage, just movement)
        if shape_type == SkillShapeType.DASH:
            return self._handle_dash_skill(caster, skill_id, extra_params)

        # Find hit targets based on shape type
        hit_results = self._check_hit_by_shape(
            caster, targets, skill_range, angle_tolerance,
            damage, shape_type, extra_params
        )

        if hit_results:
            # Apply damage to all hit targets
            for hit_target, hit_damage in hit_results:
                if hit_target.has_health():
                    hit_target.health.damage(hit_damage)

                    if not hit_target.health.is_alive:
                        hit_target.despawn()
                        self.event_bus.publish(GameEvent(
                            EventType.ENTITY_KILLED,
                            source_entity=caster,
                            target_entity=hit_target,
                            data={'skill_id': skill_id}
                        ))

            # Return event for first hit target
            first_target = hit_results[0][0]
            event = GameEvent(
                EventType.SKILL_CAST_COMPLETE,
                source_entity=caster,
                target_entity=first_target,
                data={
                    'skill_id': skill_id,
                    'damage': damage,
                    'hit_count': len(hit_results)
                }
            )
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

    def _handle_projectile_skill(
        self,
        caster: Entity,
        skill_id: str,
        damage: float,
        extra_params: Dict[str, Any]
    ) -> GameEvent:
        """Handle projectile skill completion by spawning a projectile."""
        if self.projectile_manager is None:
            # No projectile manager, return miss
            event = GameEvent(
                EventType.SKILL_MISSED,
                source_entity=caster,
                data={'skill_id': skill_id, 'reason': 'no_projectile_manager'}
            )
            self.event_bus.publish(event)
            return event

        from game.projectile import ProjectileType

        aim_angle = caster.skills.aim_angle
        caster_pos = caster.position.as_array()

        # Calculate direction from aim angle
        direction = np.array([np.cos(aim_angle), np.sin(aim_angle)])

        # Spawn projectile - hit detection will be done by ProjectileManager
        # Do NOT publish SKILL_CAST_COMPLETE here - reward is given when projectile hits
        self.projectile_manager.spawn_projectile(
            position=caster_pos,
            direction=direction,
            owner_id=caster.id,
            projectile_type=ProjectileType.SKILL_MISSILE,
            damage_override=damage
        )

        # Return a projectile spawned event (no reward)
        event = GameEvent(
            EventType.PROJECTILE_SPAWNED,
            source_entity=caster,
            data={'skill_id': skill_id, 'projectile_spawned': True}
        )
        # Don't publish here - already published by projectile_manager
        return event

    def _handle_dash_skill(
        self,
        caster: Entity,
        skill_id: str,
        extra_params: Dict[str, Any]
    ) -> GameEvent:
        """
        Handle dash skill completion by moving the caster.

        Args:
            caster: The entity casting the skill
            skill_id: Skill identifier
            extra_params: Contains dash_distance, dash_direction_offset, dash_facing_offset

        Returns:
            Event for dash completion
        """
        if not caster.has_position():
            event = GameEvent(
                EventType.SKILL_MISSED,
                source_entity=caster,
                data={'skill_id': skill_id, 'reason': 'no_position'}
            )
            self.event_bus.publish(event)
            return event

        dash_distance = extra_params.get("dash_distance", 3.0)
        # Note: aim_angle already includes the dash_direction_offset
        # The facing offset will be applied after movement
        dash_direction = caster.skills.aim_angle
        dash_facing_offset = extra_params.get("dash_facing_offset", 0.0)

        # Calculate new position
        current_pos = caster.position.as_array()
        direction_vec = np.array([np.cos(dash_direction), np.sin(dash_direction)])
        new_pos = current_pos + direction_vec * dash_distance

        # Boundary check (0-10 world size)
        new_pos = np.clip(new_pos, 0.1, 9.9)

        # Update position and facing
        caster.position.x = new_pos[0]
        caster.position.y = new_pos[1]
        caster.position.angle = caster.position.angle + np.clip(dash_facing_offset, -np.pi, np.pi)

        # Normalize angle to [-π, π]
        caster.position.angle = np.arctan2(
            np.sin(caster.position.angle),
            np.cos(caster.position.angle)
        )

        # Publish dash complete event
        event = GameEvent(
            EventType.SKILL_CAST_COMPLETE,
            source_entity=caster,
            data={
                'skill_id': skill_id,
                'dash_distance': dash_distance,
                'new_position': new_pos.tolist()
            }
        )
        self.event_bus.publish(event)
        return event

    def _check_hit_by_shape(
        self,
        caster: Entity,
        targets: List[Entity],
        skill_range: float,
        angle_tolerance: float,
        base_damage: float,
        shape_type: SkillShapeType,
        extra_params: Dict[str, Any]
    ) -> List[Tuple[Entity, float]]:
        """
        Check hits based on skill shape type.

        Returns:
            List of (target, damage) tuples for all hit targets
        """
        if shape_type == SkillShapeType.CONE:
            hit = self._check_cone_hit(caster, targets, skill_range, angle_tolerance)
            return [(hit, base_damage)] if hit else []
        elif shape_type == SkillShapeType.RING:
            return self._check_ring_hit(caster, targets, extra_params, base_damage)
        elif shape_type == SkillShapeType.RECTANGLE:
            return self._check_rectangle_hit(caster, targets, extra_params, base_damage)
        else:
            return []

    def _check_cone_hit(
        self,
        caster: Entity,
        targets: List[Entity],
        skill_range: float,
        angle_tolerance: float
    ) -> Optional[Entity]:
        """
        Check if cone skill hits any target (legacy method).

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

    def _check_ring_hit(
        self,
        caster: Entity,
        targets: List[Entity],
        extra_params: Dict[str, Any],
        base_damage: float
    ) -> List[Tuple[Entity, float]]:
        """
        Check if ring (annulus) skill hits any targets.
        Hits all targets within inner_radius <= distance <= outer_radius.

        Args:
            caster: The casting entity
            targets: Potential targets
            extra_params: Contains inner_radius and outer_radius
            base_damage: Base damage to deal

        Returns:
            List of (target, damage) tuples
        """
        if not caster.has_position():
            return []

        inner_radius = extra_params.get("inner_radius", 3.0)
        outer_radius = extra_params.get("outer_radius", 4.5)
        caster_pos = caster.position.as_array()

        hit_results = []
        for target in targets:
            if not target.is_alive or not target.has_position():
                continue

            if not target.has_tag("targetable"):
                continue

            target_pos = target.position.as_array()
            distance = np.linalg.norm(target_pos - caster_pos)

            if inner_radius <= distance <= outer_radius:
                hit_results.append((target, base_damage))

        return hit_results

    def _check_rectangle_hit(
        self,
        caster: Entity,
        targets: List[Entity],
        extra_params: Dict[str, Any],
        base_damage: float
    ) -> List[Tuple[Entity, float]]:
        """
        Check if rectangle skill hits any targets.
        Deals more damage at the tip (far end) of the rectangle.
        Also handles pull/push effects if specified in extra_params.

        Args:
            caster: The casting entity
            targets: Potential targets
            extra_params: Contains length, width, tip_range_start, tip_damage, pull_distance, push_distance
            base_damage: Base damage to deal

        Returns:
            List of (target, damage) tuples
        """
        if not caster.has_position() or not caster.has_skills():
            return []

        length = extra_params.get("length", 5.0)
        width = extra_params.get("width", 0.8)
        tip_start = extra_params.get("tip_range_start", 4.0)
        tip_damage = extra_params.get("tip_damage", 50.0)
        pull_distance = extra_params.get("pull_distance", 0.0)
        push_distance = extra_params.get("push_distance", 0.0)

        aim_angle = caster.skills.aim_angle
        caster_pos = caster.position.as_array()

        # Unit vectors for rectangle coordinate system
        forward = np.array([np.cos(aim_angle), np.sin(aim_angle)])
        right = np.array([np.sin(aim_angle), -np.cos(aim_angle)])

        hit_results = []
        for target in targets:
            if not target.is_alive or not target.has_position():
                continue

            if not target.has_tag("targetable"):
                continue

            target_pos = target.position.as_array()
            diff = target_pos - caster_pos

            # Project onto rectangle axes
            forward_dist = np.dot(diff, forward)
            right_dist = np.dot(diff, right)

            # Check if within rectangle
            if 0 < forward_dist <= length and abs(right_dist) <= width / 2:
                # Determine damage based on position
                damage = tip_damage if forward_dist >= tip_start else base_damage
                hit_results.append((target, damage))

                # Apply pull or push effect
                if pull_distance > 0:
                    # Pull towards caster
                    direction_to_caster = caster_pos - target_pos
                    dist_to_caster = np.linalg.norm(direction_to_caster)
                    if dist_to_caster > 0:
                        direction_to_caster /= dist_to_caster
                        new_pos = target_pos + direction_to_caster * pull_distance
                        new_pos = np.clip(new_pos, 0.1, 9.9)
                        target.position.x = new_pos[0]
                        target.position.y = new_pos[1]

                elif push_distance > 0:
                    # Push away from caster
                    direction_from_caster = target_pos - caster_pos
                    dist_to_caster = np.linalg.norm(direction_from_caster)
                    if dist_to_caster > 0:
                        direction_from_caster /= dist_to_caster
                        new_pos = target_pos + direction_from_caster * push_distance
                        new_pos = np.clip(new_pos, 0.1, 9.9)
                        target.position.x = new_pos[0]
                        target.position.y = new_pos[1]

        return hit_results

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
