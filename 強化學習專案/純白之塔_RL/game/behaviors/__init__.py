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
]
