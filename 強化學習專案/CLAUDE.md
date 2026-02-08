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

# Developer mode (keyboard control for debugging)
python main.py --dev
```

### Developer Mode

Developer mode allows manual control of the player for debugging game logic. The game world is frozen until keyboard input is received.

**Controls**:
- `W`: Move forward
- `A`: Turn left
- `D`: Turn right
- `P`: Pass (advance one tick without action)
- `1`: Cast 外圈刮 (Outer Slash) - Ring AOE, no aim needed
- `2`: Cast 飛彈 (Missile) - Projectile, uses mouse aim
- `3`: Cast 鐵錘 (Hammer) - Rectangle, uses mouse aim
- `R`: Reset world
- `ESC`: Quit

**Mouse**: Controls aim direction during casting for 飛彈 and 鐵錘 (aim offset clamped to ±0.5 radians)

## Architecture

### Data Flow

**Training Mode** (Agent → Player → World):
```
GameWorld.tick() → EventBus → RewardCalculator → reward
     ↓
FeatureExtractor.extract() → 27-dim observation → HybridPPOAgent → (discrete_action, continuous_action)
     ↓
GameWorld.execute_action() → Player.execute_action() → PhysicsSystem / SkillExecutor
```

**Developer Mode** (Keyboard → Player → World):
```
Keyboard Input → DevMode._wait_for_input() → (discrete_action)
Mouse Position → DevMode._calculate_aim_offset() → (continuous_action)
     ↓
GameWorld.execute_action() → Player.execute_action() → PhysicsSystem / SkillExecutor
     ↓
GameWorld.tick() (only when input received)
```

### Module Responsibilities

| Module | Purpose |
|--------|---------|
| `core/` | Event system (`EventBus`) and math utilities (`squared_prob`, `gaussian_log_prob`) |
| `game/` | Game simulation: `Entity`/`Components`, `GameWorld`, `PhysicsSystem`, `SkillExecutor` |
| `game/player.py` | Player configuration and action execution: `Player`, `PlayerConfig`, `SkillConfig` |
| `game/behaviors/` | Monster behaviors: `stationary`, `berserker`, `hit_and_run`, `orbit_melee`, `orbit_ranged` |
| `ai/` | Agent logic: `HybridPPOAgent`, `FeatureExtractor`, `RewardCalculator`, `WeightExporter` |
| `training/` | Training loop: `Trainer`, `TrainingConfig` |
| `rendering/` | Visualization: `PygameRenderer` (direction indicators, skill range fan, casting bar) |
| `dev_mode.py` | Developer mode: keyboard control for debugging (`DevMode`, `DevModeRenderer`) |

### Action Space

- **Discrete (6 actions)**: MOVE_FORWARD, ROTATE_LEFT, ROTATE_RIGHT, OUTER_SLASH, MISSILE, HAMMER
- **Continuous (2 actors)**: aim_missile (for 飛彈), aim_hammer (for 鐵錘)

**Skill to Action Mapping**:
| Action | Skill | Shape | Aim Actor |
|--------|-------|-------|-----------|
| 3 | 外圈刮 | Ring (3.0-4.5 range) | None |
| 4 | 飛彈 | Projectile | aim_actor[0] |
| 5 | 鐵錘 | Rectangle (tip bonus) | aim_actor[1] |

### Observation Features (27 dimensions)

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
[25]    Player health (0-1)
[26]    Bias term (always 1.0)
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

### Player System

Player configuration is centralized in `game/player.py`, separating player attributes from game world logic.

**Architecture**: `Agent (AI) → Player (config) → World (physics)`

**Key Classes**:
- `PlayerConfig`: Movement speed, turn speed, max health, skill configurations
- `SkillConfig`: Per-skill parameters (wind_up, range, angle_tolerance, damage)
- `Player`: Wraps Entity, provides `execute_action()` method

**Usage**:
```python
from game.player import Player, PlayerConfig, SkillConfig

config = PlayerConfig(
    move_speed=0.6,
    turn_speed=0.4,
    max_health=100.0,
    skills={
        "basic_attack": SkillConfig(
            skill_id="basic_attack",
            wind_up_ticks=4,
            range=6.0,
            angle_tolerance=0.4,
            damage=100.0
        )
    }
)

# GameWorld uses Player internally
world.player_config = config
world.reset()
```

**Property Delegation**: `player.position`, `player.health`, `player.skills` delegate to `player.entity.*` for backward compatibility.

### Skill System

Skills are defined in `game/player.py` with `SkillConfig`. Each skill has a `SkillShapeType` that determines hit detection logic.

**Shape Types** (in `game/skills.py`):
- `CONE`: Fan-shaped area (legacy)
- `RING`: Annulus/donut shape (外圈刮)
- `RECTANGLE`: Rectangle with tip bonus damage (鐵錘)
- `PROJECTILE`: Spawns projectile, hit determined later (飛彈)

**Current Skills**:

| Skill | Wind-up | Shape | Parameters |
|-------|---------|-------|------------|
| 外圈刮 | 10 ticks | RING | inner=3.0, outer=4.5, damage=30 |
| 飛彈 | 5 ticks | PROJECTILE | speed=1.5, damage=40 |
| 鐵錘 | 15 ticks | RECTANGLE | length=5.0, width=0.8, base=25, tip=50 |

**Important**: Projectile skills only give reward when the projectile actually hits a monster, not when cast.

### Projectile System

Projectiles are used by both ranged monsters and the player's 飛彈 skill.

**Projectile Types** (in `game/projectile.py`):

| Type | Speed | Damage | Radius | Lifetime | Owner |
|------|-------|--------|--------|----------|-------|
| ARROW | 0.8 | 15 | 0.3 | 50 ticks | Monster |
| MAGIC_BOLT | 0.6 | 20 | 0.4 | 60 ticks | Monster |
| SKILL_MISSILE | 1.5 | 40 | 0.5 | 40 ticks | Player |

**Key Classes**:
- `Projectile`: dataclass with position, direction, speed, damage, owner_id, lifetime
- `ProjectileManager`: handles spawn, update, collision detection, boundary checks

**Collision Logic**:
- Monster projectiles (ARROW, MAGIC_BOLT) → check collision with player
- Player projectiles (SKILL_MISSILE) → check collision with monsters, publish `SKILL_CAST_COMPLETE` on hit

**Events**:
- `PROJECTILE_SPAWNED`: Fired when projectile is created
- `PROJECTILE_HIT`: Fired when monster projectile hits player
- `SKILL_CAST_COMPLETE`: Fired when player projectile hits monster (triggers reward)
- `PROJECTILE_DESPAWNED`: Fired when projectile removed (boundary/timeout)

**Usage**:
```python
# Access projectiles from GameWorld
projectiles = world.get_active_projectiles()
for proj in projectiles:
    print(proj.position, proj.projectile_type)
```

### Reward Structure (in `ai/reward.py`)

| Event | Reward | Note |
|-------|--------|------|
| TICK | -0.01 | Time penalty |
| HIT_WALL | -2.0 | Player hits boundary |
| ITEM_COLLECTED | +25.0 | Blood pack collected |
| SKILL_CAST_COMPLETE | +12.0 | Skill hits monster (including projectile) |
| AGENT_DIED | -200.0 | Player death |
| ALL_ENEMIES_DEAD | +300.0 | Victory |

**Important**: Projectile skills (飛彈) only trigger `SKILL_CAST_COMPLETE` when the projectile actually hits a monster, not when cast. Projectiles hitting walls or timing out give no reward.

## Conventions

- Entity data uses component pattern: `entity.position`, `entity.health`, `entity.skills`
- World coordinates: 0-10 float range, Y increases upward
- Angles: radians, 0 = right, π/2 = up
- All positions stored as `np.ndarray([x, y])`
