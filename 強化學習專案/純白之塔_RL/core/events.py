"""
Event system for decoupling game logic from agent logic.
Allows the reward calculator to respond to game events without
tight coupling to the game implementation.
"""

from enum import Enum, auto
from typing import Callable, Dict, List, Any, Optional
from dataclasses import dataclass, field


class EventType(Enum):
    """All event types in the game."""
    # World events
    TICK = auto()                    # Called every game tick
    EPISODE_START = auto()           # Episode begins
    EPISODE_END = auto()             # Episode ends

    # Movement events
    ENTITY_MOVED = auto()            # Entity changed position
    ENTITY_HIT_WALL = auto()         # Entity collided with boundary
    ENTITY_ROTATED = auto()          # Entity changed facing direction

    # Combat events
    SKILL_CAST_START = auto()        # Skill wind-up begins
    SKILL_CAST_COMPLETE = auto()     # Skill successfully hit target
    SKILL_MISSED = auto()            # Skill missed target
    ENTITY_DAMAGED = auto()          # Entity took damage
    ENTITY_KILLED = auto()           # Entity died

    # Item events
    ITEM_COLLECTED = auto()          # Item was picked up
    ITEM_SPAWNED = auto()            # New item appeared

    # Entity lifecycle
    ENTITY_SPAWNED = auto()          # New entity created
    ENTITY_DESPAWNED = auto()        # Entity removed from world


@dataclass
class GameEvent:
    """
    Represents a game event with associated data.

    Attributes:
        event_type: The type of event
        source_entity: The entity that caused the event (if any)
        target_entity: The entity affected by the event (if any)
        data: Additional event-specific data
    """
    event_type: EventType
    source_entity: Optional[Any] = None
    target_entity: Optional[Any] = None
    data: Dict[str, Any] = field(default_factory=dict)


class EventBus:
    """
    Central event dispatcher for the game.
    Allows subscribers to listen for specific event types.
    """

    def __init__(self):
        self._subscribers: Dict[EventType, List[Callable[[GameEvent], None]]] = {}
        self._global_subscribers: List[Callable[[GameEvent], None]] = []

    def subscribe(self, event_type: EventType, callback: Callable[[GameEvent], None]) -> None:
        """
        Subscribe to a specific event type.

        Args:
            event_type: The type of event to listen for
            callback: Function to call when event occurs
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)

    def subscribe_all(self, callback: Callable[[GameEvent], None]) -> None:
        """
        Subscribe to all events.

        Args:
            callback: Function to call for any event
        """
        self._global_subscribers.append(callback)

    def unsubscribe(self, event_type: EventType, callback: Callable[[GameEvent], None]) -> None:
        """
        Unsubscribe from a specific event type.

        Args:
            event_type: The type of event
            callback: The callback to remove
        """
        if event_type in self._subscribers:
            try:
                self._subscribers[event_type].remove(callback)
            except ValueError:
                pass

    def unsubscribe_all(self, callback: Callable[[GameEvent], None]) -> None:
        """
        Unsubscribe from all events.

        Args:
            callback: The callback to remove
        """
        try:
            self._global_subscribers.remove(callback)
        except ValueError:
            pass

    def publish(self, event: GameEvent) -> None:
        """
        Publish an event to all subscribers.

        Args:
            event: The event to publish
        """
        # Notify type-specific subscribers
        if event.event_type in self._subscribers:
            for callback in self._subscribers[event.event_type]:
                callback(event)

        # Notify global subscribers
        for callback in self._global_subscribers:
            callback(event)

    def clear(self) -> None:
        """Remove all subscribers."""
        self._subscribers.clear()
        self._global_subscribers.clear()
