# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

純白之塔_RL is a reinforcement learning framework for training a PPO agent in a 2D RPG environment. The agent learns to navigate, aim, and cast skills at monsters while collecting items. Designed for training in Python with eventual deployment to Minecraft.

**Key Design Choice**: Uses "squared probability distribution" (`P_i = logit_i² / Σ(logit_k²)`) instead of softmax to avoid dead neuron problems. This is hardware-friendly, using only `+`, `-`, `*`, `/` operations.

## Commands

```bash
# Run training (Pygame visualization, 12000 epochs default)
cd 純白之塔_RL
python main.py

# Quick training run
python main.py --epochs 100

# ASCII rendering (no pygame dependency)
python main.py --ascii

# Training without visualization
python main.py --no-render

# Export weights to custom path
python main.py --export my_weights.json
```

## Architecture

### Data Flow
```
GameWorld.tick() → EventBus → RewardCalculator → reward
     ↓
FeatureExtractor.extract() → 26-dim observation → HybridPPOAgent → (discrete_action, continuous_action)
     ↓
GameWorld.execute_action() → PhysicsSystem / SkillExecutor
```

### Module Responsibilities

| Module | Purpose |
|--------|---------|
| `core/` | Event system (`EventBus`) and math utilities (`squared_prob`, `gaussian_log_prob`) |
| `game/` | Game simulation: `Entity`/`Components`, `GameWorld`, `PhysicsSystem`, `SkillExecutor` |
| `game/behaviors/` | Monster behaviors: `stationary`, `berserker`, `hit_and_run`, `orbit_melee`, `orbit_ranged` |
| `ai/` | Agent logic: `HybridPPOAgent`, `FeatureExtractor`, `RewardCalculator`, `WeightExporter` |
| `training/` | Training loop: `Trainer`, `TrainingConfig` |
| `rendering/` | Visualization: `PygameRenderer` (direction indicators, skill range fan, casting bar) |

### Action Space

- **Discrete (4 actions)**: MOVE_FORWARD, ROTATE_LEFT, ROTATE_RIGHT, CAST_SKILL
- **Continuous (1 value)**: Aim offset during casting (±0.5 radians from facing direction)

### Observation Features (26 dimensions)

Monsters are sorted by distance (nearest first). Dead monsters have features set to 0.

```
[0-3]   Nearest monster (dx, dy, dist, relative_angle)
[4-7]   2nd nearest monster (dx, dy, dist, relative_angle)
[8-11]  3rd nearest monster (dx, dy, dist, relative_angle)
[12-15] 4th nearest monster (dx, dy, dist, relative_angle)
[16-17] Blood pack relative position (dx, dy)
[18]    Blood pack distance
[19]    Blood pack relative angle
[20-21] Player facing direction (cos, sin)
[22]    Distance to wall (facing direction)
[23]    Casting progress (0-1)
[24]    Ready to cast indicator
[25]    Bias term (always 1.0)
```

### Monster Behavior System

Monsters simulate real Minecraft players with composite actions (movement × turning × attack per tick).

**Action Space** (in `game/behaviors/base.py`):
- `MovementType`: IDLE, FORWARD, BACKWARD, LEFT, RIGHT, SPRINT_FORWARD
- `TurningType`: NONE, LEFT, RIGHT, TURN_180
- `AttackType`: NONE, ATTACK

**Available Behaviors** (in `game/behaviors/`):

| Behavior | Type | Strategy |
|----------|------|----------|
| `stationary` | Melee | Stand still, face and attack |
| `berserker` | Melee | Rush forward, never retreat |
| `hit_and_run` | Melee | Attack once, 180° turn and flee |
| `orbit_melee` | Melee | Strafe in circles, close range |
| `orbit_ranged` | Ranged | Kite at distance, fire projectiles (bow/staff) |

**Usage**:
```python
from game.behaviors import BerserkerBehavior, OrbitRangedBehavior

monster = EntityFactory.create_monster(
    x=8.0, y=8.0,
    movement_behavior=BerserkerBehavior(attack_damage=15.0)
)

archer = EntityFactory.create_monster(
    x=2.0, y=2.0,
    movement_behavior=OrbitRangedBehavior(weapon_type="bow")
)
```

### Skill System

Skills have `wind_up_ticks` (casting time), `range`, and `angle_tolerance`. The `basic_attack` skill:
- Range: 6.0 world units
- Angle tolerance: ±0.4 radians (~23°)
- Wind-up: 4 ticks

### Projectile System

Ranged monsters (`OrbitRangedBehavior`) fire projectiles instead of dealing direct damage.

**Projectile Types** (in `game/projectile.py`):

| Type | Speed | Damage | Radius | Lifetime |
|------|-------|--------|--------|----------|
| ARROW | 0.8 | 15 | 0.3 | 50 ticks |
| MAGIC_BOLT | 0.6 | 20 | 0.4 | 60 ticks |

**Key Classes**:
- `Projectile`: dataclass with position, direction, speed, damage, owner_id, lifetime
- `ProjectileManager`: handles spawn, update, collision detection, boundary checks

**Events**:
- `PROJECTILE_SPAWNED`: Fired when projectile is created
- `PROJECTILE_HIT`: Fired when projectile hits Agent
- `PROJECTILE_DESPAWNED`: Fired when projectile removed (boundary/timeout)

**Usage**:
```python
# Access projectiles from GameWorld
projectiles = world.get_active_projectiles()
for proj in projectiles:
    print(proj.position, proj.projectile_type)
```

### Reward Structure (in `ai/reward.py`)

| Event | Reward |
|-------|--------|
| TICK | -0.01 |
| HIT_WALL | -2.0 |
| ITEM_COLLECTED | +12.0 |
| SKILL_HIT | +25.0 |

## Conventions

- Entity data uses component pattern: `entity.position`, `entity.health`, `entity.skills`
- World coordinates: 0-10 float range, Y increases upward
- Angles: radians, 0 = right, π/2 = up
- All positions stored as `np.ndarray([x, y])`
