"""
Entity system - base class and factory for game entities.
Entities are containers for components with a unique ID.
"""

from typing import Dict, Any, Optional, Type, TypeVar, List
from dataclasses import dataclass, field
import numpy as np

from .components import Position, Health, Skills, Tags

T = TypeVar('T')


class Entity:
    """
    Base entity class. Entities are identified by a unique ID
    and contain various components.
    """

    _next_id: int = 0

    def __init__(self, entity_type: str = "generic"):
        """
        Create a new entity.

        Args:
            entity_type: Type identifier for this entity
        """
        self.id: int = Entity._next_id
        Entity._next_id += 1
        self.entity_type: str = entity_type
        self.is_alive: bool = True

        # Core components - initialized on demand
        self._position: Optional[Position] = None
        self._health: Optional[Health] = None
        self._skills: Optional[Skills] = None
        self._tags: Tags = Tags()

        # Custom components storage
        self._components: Dict[str, Any] = {}

    @property
    def position(self) -> Position:
        """Get position component, creating if needed."""
        if self._position is None:
            self._position = Position()
        return self._position

    @position.setter
    def position(self, value: Position) -> None:
        self._position = value

    def has_position(self) -> bool:
        """Check if entity has a position component."""
        return self._position is not None

    @property
    def health(self) -> Health:
        """Get health component, creating if needed."""
        if self._health is None:
            self._health = Health()
        return self._health

    @health.setter
    def health(self, value: Health) -> None:
        self._health = value

    def has_health(self) -> bool:
        """Check if entity has a health component."""
        return self._health is not None

    @property
    def skills(self) -> Skills:
        """Get skills component, creating if needed."""
        if self._skills is None:
            self._skills = Skills()
        return self._skills

    @skills.setter
    def skills(self, value: Skills) -> None:
        self._skills = value

    def has_skills(self) -> bool:
        """Check if entity has a skills component."""
        return self._skills is not None

    @property
    def tags(self) -> Tags:
        """Get tags component."""
        return self._tags

    def has_tag(self, tag: str) -> bool:
        """Check if entity has a specific tag."""
        return self._tags.has(tag)

    def add_tag(self, tag: str) -> 'Entity':
        """Add a tag to this entity. Returns self for chaining."""
        self._tags.add(tag)
        return self

    def set_component(self, name: str, component: Any) -> 'Entity':
        """
        Set a custom component.

        Args:
            name: Component identifier
            component: The component object

        Returns:
            Self for method chaining
        """
        self._components[name] = component
        return self

    def get_component(self, name: str, component_type: Type[T] = object) -> Optional[T]:
        """
        Get a custom component by name.

        Args:
            name: Component identifier
            component_type: Expected type (for type hints)

        Returns:
            The component if found, None otherwise
        """
        return self._components.get(name)

    def has_component(self, name: str) -> bool:
        """Check if entity has a custom component."""
        return name in self._components

    def despawn(self) -> None:
        """Mark this entity as no longer alive."""
        self.is_alive = False

    def __repr__(self) -> str:
        pos_str = f"({self.position.x:.1f}, {self.position.y:.1f})" if self._position else "no-pos"
        return f"Entity({self.id}, {self.entity_type}, {pos_str})"


class EntityFactory:
    """
    Factory for creating pre-configured entities.
    Provides templates for common entity types.
    """

    @staticmethod
    def create_player(x: float = 1.0, y: float = 1.0) -> Entity:
        """
        Create a player entity.

        Args:
            x: Initial X position
            y: Initial Y position

        Returns:
            Configured player entity
        """
        player = Entity("player")
        player.position = Position(x, y, angle=0.0)
        player.skills = Skills()
        player.add_tag("player")
        player.add_tag("controllable")
        return player

    @staticmethod
    def create_monster(x: float, y: float, monster_type: str = "basic") -> Entity:
        """
        Create a monster entity.

        Args:
            x: Initial X position
            y: Initial Y position
            monster_type: Type of monster for behavior lookup

        Returns:
            Configured monster entity
        """
        monster = Entity("monster")
        monster.position = Position(x, y)
        monster.health = Health(current=100, maximum=100)
        monster.add_tag("monster")
        monster.add_tag("targetable")
        monster.add_tag(f"monster_{monster_type}")
        return monster

    @staticmethod
    def create_blood_pack(x: float, y: float) -> Entity:
        """
        Create a blood pack (health pickup) entity.

        Args:
            x: Initial X position
            y: Initial Y position

        Returns:
            Configured blood pack entity
        """
        blood = Entity("blood_pack")
        blood.position = Position(x, y)
        blood.add_tag("item")
        blood.add_tag("blood_pack")
        blood.add_tag("collectible")
        return blood

    @staticmethod
    def reset_id_counter() -> None:
        """Reset the entity ID counter. Useful for testing."""
        Entity._next_id = 0
