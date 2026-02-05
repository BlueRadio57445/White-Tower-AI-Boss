"""
Monster registry - defines and manages monster templates.
"""

from typing import Dict, Optional, List, Callable
from dataclasses import dataclass, field

from game.entity import Entity, EntityFactory
from monsters.behaviors import MonsterBehavior, IdleBehavior


@dataclass
class MonsterTemplate:
    """
    Template for creating monsters of a specific type.

    Attributes:
        id: Unique identifier for this monster type
        name: Display name
        health: Maximum health points
        behavior_factory: Function that creates the behavior instance
        tags: Additional tags for categorization
    """
    id: str
    name: str
    health: float = 100.0
    behavior_factory: Callable[[], MonsterBehavior] = field(
        default_factory=lambda: IdleBehavior
    )
    tags: List[str] = field(default_factory=list)


class MonsterRegistry:
    """
    Registry of monster templates.
    Used to create and manage different monster types.
    """

    def __init__(self):
        self._templates: Dict[str, MonsterTemplate] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register default monster types."""
        self.register(MonsterTemplate(
            id="basic",
            name="Basic Monster",
            health=100.0,
            behavior_factory=lambda: IdleBehavior(),
            tags=["basic"]
        ))

    def register(self, template: MonsterTemplate) -> None:
        """
        Register a monster template.

        Args:
            template: The monster template to register
        """
        self._templates[template.id] = template

    def get(self, monster_id: str) -> Optional[MonsterTemplate]:
        """
        Get a monster template by ID.

        Args:
            monster_id: The monster type identifier

        Returns:
            The template or None if not found
        """
        return self._templates.get(monster_id)

    def list_monsters(self) -> List[str]:
        """Get list of all registered monster IDs."""
        return list(self._templates.keys())

    def create_monster(
        self,
        monster_id: str,
        x: float,
        y: float
    ) -> tuple:
        """
        Create a monster entity from a template.

        Args:
            monster_id: The monster type to create
            x: X position
            y: Y position

        Returns:
            Tuple of (entity, behavior) or (None, None) if template not found
        """
        template = self.get(monster_id)
        if template is None:
            return None, None

        # Create entity
        entity = EntityFactory.create_monster(x, y, monster_id)
        entity.health.maximum = template.health
        entity.health.current = template.health

        # Add template tags
        for tag in template.tags:
            entity.add_tag(tag)

        # Create behavior
        behavior = template.behavior_factory()

        return entity, behavior

    def unregister(self, monster_id: str) -> bool:
        """
        Remove a monster template.

        Args:
            monster_id: The monster type to remove

        Returns:
            True if removed, False if not found
        """
        if monster_id in self._templates:
            del self._templates[monster_id]
            return True
        return False
