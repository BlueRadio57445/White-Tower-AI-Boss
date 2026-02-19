"""
Monster behavior system - simulating real Minecraft player actions.

Actions are Cartesian product of:
- Movement: {FORWARD, BACKWARD, LEFT, RIGHT, SPRINT_FORWARD, IDLE}
- Turning: {LEFT, RIGHT, TURN_180, NONE}
- Attack: {NO_ATTACK, ATTACK}

Available behaviors:
- stationary: Stands still, faces and attacks Agent
- berserker: Rushes at Agent, fights to the death
- hit_and_run: Hit once then flee, repeat
- orbit_melee: Circle around Agent at close range (melee)
- orbit_ranged: Circle around Agent at long range (kiting)
- opportunist: Waits at safe distance, rushes in when Boss is casting
- adaptive: Changes strategy based on own health percentage
- backstab: Always tries to get behind the Boss to attack
- blood_pack_disruptor_melee: Guards blood packs (melee, delegates to OrbitMeleeBehavior)
- blood_pack_disruptor_ranged: Guards blood packs (ranged, delegates to OrbitRangedBehavior, spawns projectiles)
"""

from game.behaviors.base import (
    # Enums
    MovementType,
    TurningType,
    AttackType,
    # Data classes
    MonsterAction,
    # Speed constants
    MovementSpeed,
    TurningSpeed,
    # Base class
    MonsterBehavior,
    # Registry
    BehaviorRegistry,
    # Executor
    MonsterActionExecutor,
)

# Import all behaviors to register them
from game.behaviors.stationary import StationaryBehavior
from game.behaviors.berserker import BerserkerBehavior
from game.behaviors.hit_and_run import HitAndRunBehavior
from game.behaviors.orbit import OrbitMeleeBehavior, OrbitRangedBehavior
from game.behaviors.opportunist import OpportunistBehavior
from game.behaviors.adaptive import AdaptiveBehavior
from game.behaviors.backstab import BackstabBehavior
from game.behaviors.blood_pack_disruptor import BloodPackDisruptorMeleeBehavior, BloodPackDisruptorRangedBehavior

__all__ = [
    # Enums
    'MovementType',
    'TurningType',
    'AttackType',
    # Data classes
    'MonsterAction',
    # Speed constants
    'MovementSpeed',
    'TurningSpeed',
    # Base class
    'MonsterBehavior',
    # Registry
    'BehaviorRegistry',
    # Executor
    'MonsterActionExecutor',
    # Behaviors
    'StationaryBehavior',
    'BerserkerBehavior',
    'HitAndRunBehavior',
    'OrbitMeleeBehavior',
    'OrbitRangedBehavior',
    'OpportunistBehavior',
    'AdaptiveBehavior',
    'BackstabBehavior',
    'BloodPackDisruptorMeleeBehavior',
    'BloodPackDisruptorRangedBehavior',
]
