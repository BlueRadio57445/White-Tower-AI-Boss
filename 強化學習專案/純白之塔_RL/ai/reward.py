"""
Reward calculation based on game events.
Subscribes to the event bus to accumulate rewards.
"""

from typing import Dict, Optional
from core.events import EventBus, GameEvent, EventType


class RewardCalculator:
    """
    Calculates rewards based on game events.
    Subscribes to the event bus and accumulates rewards per step.
    """

    # Default reward values
    DEFAULT_REWARDS: Dict[EventType, float] = {
        EventType.TICK: -0.01,
        EventType.ENTITY_HIT_WALL: -2.0,
        EventType.ITEM_COLLECTED: 25.0,
        EventType.SKILL_CAST_COMPLETE: 12.0,
        EventType.SKILL_MISSED: 0.0,
        EventType.ENTITY_KILLED: 0.0,  # Already counted in SKILL_CAST_COMPLETE
        EventType.AGENT_DIED: -200.0,
        EventType.ALL_ENEMIES_DEAD: 300.0,
    }

    def __init__(
        self,
        event_bus: EventBus,
        reward_config: Optional[Dict[EventType, float]] = None
    ):
        """
        Initialize the reward calculator.

        Args:
            event_bus: Event bus to subscribe to
            reward_config: Custom reward values (optional)
        """
        self.event_bus = event_bus
        self.rewards = {**self.DEFAULT_REWARDS}

        if reward_config:
            self.rewards.update(reward_config)

        self.accumulated_reward: float = 0.0
        self.last_event: str = ""

        # Episode termination state
        self._episode_done: bool = False
        self._win: bool = False

        # Subscribe to relevant events
        self._subscribe()

    def _subscribe(self) -> None:
        """Subscribe to all reward-generating events."""
        self.event_bus.subscribe(EventType.TICK, self._on_tick)
        self.event_bus.subscribe(EventType.ENTITY_HIT_WALL, self._on_hit_wall)
        self.event_bus.subscribe(EventType.ITEM_COLLECTED, self._on_item_collected)
        self.event_bus.subscribe(EventType.SKILL_CAST_COMPLETE, self._on_skill_hit)
        self.event_bus.subscribe(EventType.SKILL_MISSED, self._on_skill_missed)
        self.event_bus.subscribe(EventType.AGENT_DIED, self._on_agent_died)
        self.event_bus.subscribe(EventType.ALL_ENEMIES_DEAD, self._on_all_enemies_dead)

    def _on_tick(self, event: GameEvent) -> None:
        """Handle tick event (time penalty)."""
        self.accumulated_reward += self.rewards[EventType.TICK]

    def _on_hit_wall(self, event: GameEvent) -> None:
        """Handle wall collision event."""
        # Only penalize if source is the player
        if event.source_entity and event.source_entity.has_tag("player"):
            self.accumulated_reward += self.rewards[EventType.ENTITY_HIT_WALL]
            self.last_event = "HIT WALL!"

    def _on_item_collected(self, event: GameEvent) -> None:
        """Handle item collection event."""
        if event.target_entity and event.target_entity.has_tag("blood_pack"):
            self.accumulated_reward += self.rewards[EventType.ITEM_COLLECTED]
            self.last_event = "EAT BLOOD!"

    def _on_skill_hit(self, event: GameEvent) -> None:
        """Handle successful skill hit event."""
        # Check if this is a summon_pack skill
        skill_id = event.data.get('skill_id', '') if event.data else ''

        if skill_id == 'summon_pack':
            # Special reward for summon: +3.0 per pack spawned
            packs_spawned = event.data.get('packs_spawned', 0)
            if packs_spawned > 0:
                self.accumulated_reward += 3.0 * packs_spawned
                self.last_event = f"SUMMONED {packs_spawned} PACKS!"
            else:
                # No reward if summon failed (hit limit)
                self.last_event = "SUMMON LIMIT!"
        else:
            # Normal skill reward (damage skills)
            self.accumulated_reward += self.rewards[EventType.SKILL_CAST_COMPLETE]
            self.last_event = "KILLED MONSTER!"

    def _on_skill_missed(self, event: GameEvent) -> None:
        """Handle skill miss event."""
        self.accumulated_reward += self.rewards[EventType.SKILL_MISSED]
        self.last_event = "MISSED..."

    def _on_agent_died(self, event: GameEvent) -> None:
        """Handle agent death event."""
        self.accumulated_reward += self.rewards[EventType.AGENT_DIED]
        self.last_event = "AGENT DIED!"
        self._episode_done = True
        self._win = False

    def _on_all_enemies_dead(self, event: GameEvent) -> None:
        """Handle all enemies killed event (victory)."""
        self.accumulated_reward += self.rewards[EventType.ALL_ENEMIES_DEAD]
        self.last_event = "VICTORY!"
        self._episode_done = True
        self._win = True

    def get_reward(self) -> float:
        """
        Get the accumulated reward and reset.

        Returns:
            Total accumulated reward since last call
        """
        reward = self.accumulated_reward
        self.accumulated_reward = 0.0
        return reward

    def get_last_event(self) -> str:
        """
        Get the last significant event description.

        Returns:
            Event description string
        """
        event = self.last_event
        self.last_event = ""
        return event

    def peek_reward(self) -> float:
        """
        Get accumulated reward without resetting.

        Returns:
            Current accumulated reward
        """
        return self.accumulated_reward

    def reset(self) -> None:
        """Reset accumulated reward and event."""
        self.accumulated_reward = 0.0
        self.last_event = ""
        self._episode_done = False
        self._win = False

    def is_episode_done(self) -> bool:
        """Check if episode has ended (agent died or all enemies killed)."""
        return self._episode_done

    def is_win(self) -> bool:
        """Check if episode ended in victory."""
        return self._win

    def set_reward(self, event_type: EventType, value: float) -> None:
        """
        Set a custom reward value for an event type.

        Args:
            event_type: The event type
            value: The reward value
        """
        self.rewards[event_type] = value

    def unsubscribe(self) -> None:
        """Unsubscribe from all events."""
        self.event_bus.unsubscribe(EventType.TICK, self._on_tick)
        self.event_bus.unsubscribe(EventType.ENTITY_HIT_WALL, self._on_hit_wall)
        self.event_bus.unsubscribe(EventType.ITEM_COLLECTED, self._on_item_collected)
        self.event_bus.unsubscribe(EventType.SKILL_CAST_COMPLETE, self._on_skill_hit)
        self.event_bus.unsubscribe(EventType.SKILL_MISSED, self._on_skill_missed)
        self.event_bus.unsubscribe(EventType.AGENT_DIED, self._on_agent_died)
        self.event_bus.unsubscribe(EventType.ALL_ENEMIES_DEAD, self._on_all_enemies_dead)
