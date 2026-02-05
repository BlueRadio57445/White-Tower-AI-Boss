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
FeatureExtractor.extract() → 15-dim observation → HybridPPOAgent → (discrete_action, continuous_action)
     ↓
GameWorld.execute_action() → PhysicsSystem / SkillExecutor
```

### Module Responsibilities

| Module | Purpose |
|--------|---------|
| `core/` | Event system (`EventBus`) and math utilities (`squared_prob`, `gaussian_log_prob`) |
| `game/` | Game simulation: `Entity`/`Components`, `GameWorld`, `PhysicsSystem`, `SkillExecutor` |
| `ai/` | Agent logic: `HybridPPOAgent`, `FeatureExtractor`, `RewardCalculator`, `WeightExporter` |
| `training/` | Training loop: `Trainer`, `TrainingConfig` |
| `rendering/` | Visualization: `PygameRenderer` (direction indicators, skill range fan, casting bar) |

### Action Space

- **Discrete (4 actions)**: MOVE_FORWARD, ROTATE_LEFT, ROTATE_RIGHT, CAST_SKILL
- **Continuous (1 value)**: Aim offset during casting (±0.5 radians from facing direction)

### Observation Features (15 dimensions)

```
[0-1]  Monster relative position (dx, dy)
[2-3]  Blood pack relative position (dx, dy)
[4-5]  Distances to monster/blood (normalized)
[6-7]  Relative angles to monster/blood
[8-9]  Player facing direction (cos, sin)
[10]   Monster in attack range indicator
[11]   Distance to nearest wall
[12]   Casting progress (0-1)
[13]   Ready to cast indicator
[14]   Bias term (always 1.0)
```

### Skill System

Skills have `wind_up_ticks` (casting time), `range`, and `angle_tolerance`. The `basic_attack` skill:
- Range: 6.0 world units
- Angle tolerance: ±0.4 radians (~23°)
- Wind-up: 4 ticks

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
