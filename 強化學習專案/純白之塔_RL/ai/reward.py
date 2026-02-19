"""
Reward calculation based on game events.
Subscribes to the event bus to accumulate rewards.

Reward structure:
    TICK:              -0.01 per tick (time pressure)
    HIT_WALL:          -2.0 (collision penalty)
    DAMAGE_DEALT:      +1.0 × damage (per hit)
    DAMAGE_TAKEN:      -0.01 × damage (melee) / -1.0 × damage (projectile)
    HEAL:              +1.0 × actual_heal (blood pack collected, capped by missing HP)
    KILL_ENEMY:        +20.0 (per kill)
    SUMMON_BLOOD_PACK: +3.0 per pack spawned
    AGENT_DIED:        -200.0
    ALL_ENEMIES_DEAD:  +300.0
"""

from core.events import EventBus, GameEvent, EventType


class RewardCalculator:
    """
    Calculates rewards based on game events.
    Subscribes to the event bus and accumulates rewards per step.
    """

    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.accumulated_reward: float = 0.0
        self.last_event: str = ""

        # Episode termination state
        self._episode_done: bool = False
        self._win: bool = False

        self._subscribe()

    def _subscribe(self) -> None:
        """Subscribe to all reward-generating events."""
        self.event_bus.subscribe(EventType.TICK, self._on_tick)
        self.event_bus.subscribe(EventType.ENTITY_HIT_WALL, self._on_hit_wall)
        self.event_bus.subscribe(EventType.ITEM_COLLECTED, self._on_item_collected)
        self.event_bus.subscribe(EventType.SKILL_CAST_COMPLETE, self._on_skill_hit)
        self.event_bus.subscribe(EventType.ENTITY_KILLED, self._on_entity_killed)
        self.event_bus.subscribe(EventType.DAMAGE_TAKEN, self._on_damage_taken)
        self.event_bus.subscribe(EventType.AGENT_DIED, self._on_agent_died)
        self.event_bus.subscribe(EventType.ALL_ENEMIES_DEAD, self._on_all_enemies_dead)

    def _on_tick(self, event: GameEvent) -> None:
        """Handle tick event (time penalty)."""
        self.accumulated_reward += -0.01

    def _on_hit_wall(self, event: GameEvent) -> None:
        """Handle wall collision event."""
        if event.source_entity and event.source_entity.has_tag("player"):
            self.accumulated_reward += -2.0
            self.last_event = "HIT WALL!"

    def _on_item_collected(self, event: GameEvent) -> None:
        """Handle blood pack collection (HEAL: +1.0 × actual_heal)."""
        if event.target_entity and event.target_entity.has_tag("blood_pack"):
            actual_heal = event.data.get('actual_heal', 0.0) if event.data else 0.0
            if actual_heal > 0:
                self.accumulated_reward += 1.0 * actual_heal
                self.last_event = "EAT BLOOD!"

    def _on_skill_hit(self, event: GameEvent) -> None:
        """Handle skill hit events.

        - Damage skills: DAMAGE_DEALT = +1.0 × damage
        - Summon skill:  SUMMON_BLOOD_PACK = +3.0 per pack spawned
        - Dash/other:    No reward (no 'damage' field in data)
        """
        skill_id = event.data.get('skill_id', '') if event.data else ''

        if skill_id == 'summon_pack':
            packs_spawned = event.data.get('packs_spawned', 0)
            if packs_spawned > 0:
                self.accumulated_reward += 10.0 * packs_spawned
                self.last_event = f"SUMMONED {packs_spawned} PACKS!"
            else:
                self.last_event = "SUMMON LIMIT!"
        else:
            # DAMAGE_DEALT: +1.0 × damage
            damage = event.data.get('damage', 0.0) if event.data else 0.0
            if damage > 0:
                self.accumulated_reward += 1.0 * damage
                self.last_event = "HIT MONSTER!"

    def _on_entity_killed(self, event: GameEvent) -> None:
        """Handle enemy kill event (KILL_ENEMY: +20.0)."""
        self.accumulated_reward += 120.0
        self.last_event = "KILLED MONSTER!"

    def _on_damage_taken(self, event: GameEvent) -> None:
        """Handle player taking damage.

        - Melee:     -0.01 × damage
        - Projectile: -1.0 × damage
        """
        damage = event.data.get('damage', 0.0) if event.data else 0.0
        if damage > 0:
            source = event.data.get('source', 'melee') if event.data else 'melee'
            multiplier = 0.01 if source == 'melee' else 1.0
            self.accumulated_reward -= multiplier * damage
            self.last_event = "TOOK DAMAGE!"

    def _on_agent_died(self, event: GameEvent) -> None:
        """Handle agent death event."""
        self.accumulated_reward += -200.0
        self.last_event = "AGENT DIED!"
        self._episode_done = True
        self._win = False

    def _on_all_enemies_dead(self, event: GameEvent) -> None:
        """Handle all enemies killed event (victory)."""
        self.accumulated_reward += 400.0
        self.last_event = "VICTORY!"
        self._episode_done = True
        self._win = True

    def get_reward(self) -> float:
        """Get the accumulated reward and reset."""
        reward = self.accumulated_reward
        self.accumulated_reward = 0.0
        return reward

    def get_last_event(self) -> str:
        """Get the last significant event description."""
        event = self.last_event
        self.last_event = ""
        return event

    def peek_reward(self) -> float:
        """Get accumulated reward without resetting."""
        return self.accumulated_reward

    def reset(self) -> None:
        """Reset accumulated reward and episode state."""
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

    def unsubscribe(self) -> None:
        """Unsubscribe from all events."""
        self.event_bus.unsubscribe(EventType.TICK, self._on_tick)
        self.event_bus.unsubscribe(EventType.ENTITY_HIT_WALL, self._on_hit_wall)
        self.event_bus.unsubscribe(EventType.ITEM_COLLECTED, self._on_item_collected)
        self.event_bus.unsubscribe(EventType.SKILL_CAST_COMPLETE, self._on_skill_hit)
        self.event_bus.unsubscribe(EventType.ENTITY_KILLED, self._on_entity_killed)
        self.event_bus.unsubscribe(EventType.DAMAGE_TAKEN, self._on_damage_taken)
        self.event_bus.unsubscribe(EventType.AGENT_DIED, self._on_agent_died)
        self.event_bus.unsubscribe(EventType.ALL_ENEMIES_DEAD, self._on_all_enemies_dead)
