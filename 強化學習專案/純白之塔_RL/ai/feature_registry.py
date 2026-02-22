"""
Feature group registry for Forward Stepwise Selection.

A "feature group" is the minimal unit that FSS adds or removes at once.
Each group has a name, a fixed output dimension count, and an extraction function.

Registering a new feature group:
    @feature_group("my_feature", n_dims=2)
    def my_feature(world, world_size):
        return np.array([value_a, value_b])

The function must return a np.ndarray of exactly n_dims elements.
world_size is passed for normalization (typically 10.0).

All groups that should be tested by FSS are registered here.
SelectiveFeatureExtractor takes a list of group names and builds the
corresponding feature vector by concatenating their outputs.

Current groups and dimensions:
    monster_dist              4D   distance to each monster (nearest first)
    monster_dx_dy             8D   (dx, dy) to each monster (nearest first)
    monster_relative_angle    4D   relative angle to each monster (nearest first)
    monster_health            4D   health ratio of each monster (nearest first)
    monster_damage_taken      4D   damage taken ratio (1 - health) of each monster (nearest first)
    monster_angle_sincos      8D   (sin, cos) of relative angle to each monster (nearest first)
    monster_all              16D   all monster features grouped by type (dist ×4, dx_dy ×4, angle ×4)
    ring_hit_count            1D   number of monsters in 外圈刮 ring range (inner=3.0, outer=4.5), normalized
    movable_cast_state        3D   binary flags: [is_casting_outer_slash, is_casting_hammer, is_in_blood_pool]
    enemy_proj_count          1D   number of active enemy projectiles, normalized by 8
    nearest_enemy_proj        4D   nearest enemy projectile [dist, dx, dy, heading_toward]; [1,0,0,0] if none
    blood_pack_dist           3D   distance to each blood pack (nearest first)
    blood_pack_dx_dy          6D   (dx, dy) to each blood pack (nearest first)
    blood_pack_relative_angle 3D   relative angle to each blood pack (nearest first)
    blood_pack_angle_sincos   6D   (sin, cos) of relative angle to each blood pack (nearest first)
    blood_pack_all           12D   all blood pack features grouped by type (dist ×3, dx_dy ×3, angle ×3)
    player_facing             2D   (cos, sin) of player facing angle
    wall_dist                 1D   distance to wall in facing direction
    casting_info              9D   (casting_progress, cooldown_ratio ×8 skills)
    player_health             1D   player health ratio (0-1)
    player_damage_taken       1D   player damage taken ratio (1 - health)
    total_monster_hp          1D   sum of monster health ratios / MAX_MONSTERS
    alive_monster_count       1D   number of alive monsters / MAX_MONSTERS
    ready_skill_count         1D   fraction of skills fully off cooldown
    hp_advantage              1D   player_health - total_monster_hp; positive = player winning
    blood_pack_count          1D   number of alive blood packs, normalized by MAX_BLOOD_PACKS
    bias                      1D   constant 1.0
    -------------------------------------------------------
    ALL GROUPS               56D   (DEFAULT_GROUP_ORDER total)
"""

from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np

from game.world import GameWorld
from game.projectile import ProjectileType

# ---------------------------------------------------------------------------
# Registry internals
# ---------------------------------------------------------------------------

@dataclass
class FeatureGroup:
    name: str
    n_dims: int
    fn: Callable  # (world: GameWorld, world_size: float) -> np.ndarray


_REGISTRY: Dict[str, FeatureGroup] = {}


def feature_group(name: str, n_dims: int):
    """Decorator to register a feature group."""
    def decorator(fn: Callable) -> Callable:
        _REGISTRY[name] = FeatureGroup(name=name, n_dims=n_dims, fn=fn)
        return fn
    return decorator


def list_all_groups() -> List[str]:
    """Return names of all registered feature groups."""
    return list(_REGISTRY.keys())


def get_group(name: str) -> FeatureGroup:
    """Return a registered group by name, or raise KeyError."""
    if name not in _REGISTRY:
        raise KeyError(
            f"Feature group '{name}' not found. "
            f"Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[name]


# ---------------------------------------------------------------------------
# Shared helpers (used by multiple groups)
# ---------------------------------------------------------------------------

MAX_MONSTERS = 4
MAX_BLOOD_PACKS = 3


def _get_sorted_monster_data(world: GameWorld, world_size: float):
    """
    Compute and sort monster data by distance (nearest first).

    Returns list of (dist, dx, dy, rel_angle, health_ratio) tuples,
    zero-padded to MAX_MONSTERS entries.

    Results are cached on world._feature_cache (per extract() call) when available.
    """
    cache = getattr(world, '_feature_cache', None)
    cache_key = f'monster_{world_size}'
    if cache is not None and cache_key in cache:
        return cache[cache_key]

    player_pos = world.get_player_position()
    player_angle = world.get_player_angle()

    data = []
    for pos, health_ratio in world.get_alive_monster_positions_and_health():
        dx = (pos[0] - player_pos[0]) / world_size
        dy = (pos[1] - player_pos[1]) / world_size
        dist = np.linalg.norm(pos - player_pos) / world_size
        angle_to = np.arctan2(pos[1] - player_pos[1], pos[0] - player_pos[0])
        rel_angle = np.arctan2(
            np.sin(angle_to - player_angle),
            np.cos(angle_to - player_angle)
        )
        data.append((dist, dx, dy, rel_angle, health_ratio))

    data.sort(key=lambda x: x[0])

    # Zero-pad
    while len(data) < MAX_MONSTERS:
        data.append((0.0, 0.0, 0.0, 0.0, 0.0))

    result = data[:MAX_MONSTERS]
    if cache is not None:
        cache[cache_key] = result
    return result


def _get_sorted_blood_pack_data(world: GameWorld, world_size: float):
    """
    Compute and sort blood pack data by distance (nearest first).

    Returns list of (dist, dx, dy, rel_angle) tuples,
    zero-padded to MAX_BLOOD_PACKS entries.

    Results are cached on world._feature_cache (per extract() call) when available.
    """
    cache = getattr(world, '_feature_cache', None)
    cache_key = f'blood_{world_size}'
    if cache is not None and cache_key in cache:
        return cache[cache_key]

    player_pos = world.get_player_position()
    player_angle = world.get_player_angle()
    blood_positions = world.get_alive_blood_pack_positions()

    data = []
    for pos in blood_positions:
        dx = (pos[0] - player_pos[0]) / world_size
        dy = (pos[1] - player_pos[1]) / world_size
        dist = np.linalg.norm(pos - player_pos) / world_size
        angle_to = np.arctan2(pos[1] - player_pos[1], pos[0] - player_pos[0])
        rel_angle = np.arctan2(
            np.sin(angle_to - player_angle),
            np.cos(angle_to - player_angle)
        )
        data.append((dist, dx, dy, rel_angle))

    data.sort(key=lambda x: x[0])

    # Zero-pad
    while len(data) < MAX_BLOOD_PACKS:
        data.append((0.0, 0.0, 0.0, 0.0))

    result = data[:MAX_BLOOD_PACKS]
    if cache is not None:
        cache[cache_key] = result
    return result


def _raycast_to_wall(pos: np.ndarray, cos_a: float, sin_a: float, world_size: float) -> float:
    """Distance to nearest wall in the given direction."""
    x, y = pos[0], pos[1]
    distances = []
    if cos_a > 1e-6:
        distances.append((world_size - x) / cos_a)
    elif cos_a < -1e-6:
        distances.append(-x / cos_a)
    if sin_a > 1e-6:
        distances.append((world_size - y) / sin_a)
    elif sin_a < -1e-6:
        distances.append(-y / sin_a)
    return min(distances) if distances else world_size


# ---------------------------------------------------------------------------
# Feature group definitions
# ---------------------------------------------------------------------------

@feature_group("monster_dist", n_dims=4)
def monster_dist(world: GameWorld, world_size: float) -> np.ndarray:
    """Distance to each monster (nearest first). 4D."""
    data = _get_sorted_monster_data(world, world_size)
    return np.array([d[0] for d in data])


@feature_group("monster_dx_dy", n_dims=8)
def monster_dx_dy(world: GameWorld, world_size: float) -> np.ndarray:
    """Relative (dx, dy) to each monster (nearest first). 8D."""
    data = _get_sorted_monster_data(world, world_size)
    return np.array([v for d in data for v in (d[1], d[2])])


@feature_group("monster_relative_angle", n_dims=4)
def monster_relative_angle(world: GameWorld, world_size: float) -> np.ndarray:
    """Relative angle to each monster (nearest first). 4D."""
    data = _get_sorted_monster_data(world, world_size)
    return np.array([d[3] for d in data])


@feature_group("monster_health", n_dims=4)
def monster_health(world: GameWorld, world_size: float) -> np.ndarray:
    """Health ratio of each monster (nearest first). 4D. (0.0 = dead, 1.0 = full hp)"""
    data = _get_sorted_monster_data(world, world_size)
    return np.array([d[4] for d in data])


@feature_group("monster_damage_taken", n_dims=4)
def monster_damage_taken(world: GameWorld, world_size: float) -> np.ndarray:
    """
    Damage taken ratio (1 - health_ratio) for each monster (nearest first). 4D.
    - 0.0 = full hp or monster doesn't exist
    - 0.5 = monster at 50% hp (took 50% damage)
    - 1.0 = monster dead
    """
    data = _get_sorted_monster_data(world, world_size)
    result = []
    for d in data:
        health_ratio = d[4]
        # If monster exists (health_ratio > 0 or distance > 0), compute damage taken
        # Otherwise (zero-padded entry), set to 0
        if d[0] > 0 or health_ratio > 0:  # d[0] is distance
            result.append(1.0 - health_ratio)
        else:
            result.append(0.0)
    return np.array(result)


@feature_group("monster_angle_sincos", n_dims=8)
def monster_angle_sincos(world: GameWorld, world_size: float) -> np.ndarray:
    """
    Sin/cos decomposition of relative angle to each monster (nearest first). 8D.
    Output: [sin1,cos1, sin2,cos2, sin3,cos3, sin4,cos4]
    Avoids the ±π discontinuity of raw angles.
    cos(angle) encodes how "ahead" the monster is; sin(angle) encodes left/right offset.
    """
    data = _get_sorted_monster_data(world, world_size)
    return np.array([v for d in data for v in (np.sin(d[3]), np.cos(d[3]))])


@feature_group("blood_pack_dist", n_dims=3)
def blood_pack_dist(world: GameWorld, world_size: float) -> np.ndarray:
    """Distance to each blood pack (nearest first). 3D."""
    data = _get_sorted_blood_pack_data(world, world_size)
    return np.array([d[0] for d in data])


@feature_group("blood_pack_dx_dy", n_dims=6)
def blood_pack_dx_dy(world: GameWorld, world_size: float) -> np.ndarray:
    """Relative (dx, dy) to each blood pack (nearest first). 6D."""
    data = _get_sorted_blood_pack_data(world, world_size)
    return np.array([v for d in data for v in (d[1], d[2])])


@feature_group("blood_pack_relative_angle", n_dims=3)
def blood_pack_relative_angle(world: GameWorld, world_size: float) -> np.ndarray:
    """Relative angle to each blood pack (nearest first). 3D."""
    data = _get_sorted_blood_pack_data(world, world_size)
    return np.array([d[3] for d in data])


@feature_group("blood_pack_angle_sincos", n_dims=6)
def blood_pack_angle_sincos(world: GameWorld, world_size: float) -> np.ndarray:
    """
    Sin/cos decomposition of relative angle to each blood pack (nearest first). 6D.
    Output: [sin1,cos1, sin2,cos2, sin3,cos3]
    Avoids the ±π discontinuity of raw angles.
    """
    data = _get_sorted_blood_pack_data(world, world_size)
    return np.array([v for d in data for v in (np.sin(d[3]), np.cos(d[3]))])


@feature_group("blood_pack_all", n_dims=12)
def blood_pack_all(world: GameWorld, world_size: float) -> np.ndarray:
    """
    All blood pack features grouped by type. 12D.
    Output: [dist1..3, dx1,dy1..dx3,dy3, angle1..3]
    Equivalent to blood_pack_dist + blood_pack_dx_dy + blood_pack_relative_angle.
    """
    data = _get_sorted_blood_pack_data(world, world_size)
    dists  = np.array([d[0] for d in data])
    dx_dys = np.array([v for d in data for v in (d[1], d[2])])
    angles = np.array([d[3] for d in data])
    return np.concatenate([dists, dx_dys, angles])


@feature_group("player_facing", n_dims=2)
def player_facing(world: GameWorld, world_size: float) -> np.ndarray:
    """Player facing direction (cos, sin). 2D."""
    angle = world.get_player_angle()
    return np.array([np.cos(angle), np.sin(angle)])


@feature_group("wall_dist", n_dims=1)
def wall_dist(world: GameWorld, world_size: float) -> np.ndarray:
    """Distance to wall in player facing direction. 1D."""
    pos = world.get_player_position()
    angle = world.get_player_angle()
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    dist = _raycast_to_wall(pos, cos_a, sin_a, world_size) / world_size
    return np.array([dist])


@feature_group("casting_info", n_dims=9)
def casting_info(world: GameWorld, world_size: float) -> np.ndarray:
    """
    Casting state for the player. 9D.
    [casting_progress, cd_outer_slash, cd_missile, cd_hammer, cd_dash,
     cd_soul_claw, cd_soul_palm, cd_blood_pool, cd_summon_pack]
    Cooldown ratio = 1 - remaining / max (1.0 = ready, 0.0 = just used).
    """
    progress = world.get_casting_progress()
    cooldowns = world.get_skill_cooldown_ratios()  # 8D, SKILL_ACTION_MAP order
    return np.concatenate([[progress], cooldowns])


@feature_group("player_health", n_dims=1)
def player_health(world: GameWorld, world_size: float) -> np.ndarray:
    """Player health ratio (0-1). 1D."""
    return np.array([world.get_player_health_percentage()])


@feature_group("player_damage_taken", n_dims=1)
def player_damage_taken(world: GameWorld, world_size: float) -> np.ndarray:
    """
    Player damage taken ratio (1 - health_ratio). 1D.
    - 0.0 = full hp
    - 0.5 = player at 50% hp (took 50% damage)
    - 1.0 = player dead
    """
    health_ratio = world.get_player_health_percentage()
    return np.array([1.0 - health_ratio])


@feature_group("total_monster_hp", n_dims=1)
def total_monster_hp(world: GameWorld, world_size: float) -> np.ndarray:
    """
    Sum of all monster health ratios, normalized by MAX_MONSTERS. 1D.
    0.0 = all monsters dead, 1.0 = all monsters at full HP.
    Critic use: linear predictor of distance to ALL_ENEMIES_DEAD (+300).
    """
    data = _get_sorted_monster_data(world, world_size)
    total = sum(d[4] for d in data if d[0] > 0)  # d[4]=health_ratio, d[0]>0 = alive
    return np.array([total / MAX_MONSTERS])


@feature_group("alive_monster_count", n_dims=1)
def alive_monster_count(world: GameWorld, world_size: float) -> np.ndarray:
    """
    Number of alive monsters, normalized by MAX_MONSTERS. 1D.
    Critic use: linear predictor of KILL_ENEMY rewards remaining.
    """
    data = _get_sorted_monster_data(world, world_size)
    count = sum(1 for d in data if d[0] > 0)
    return np.array([count / MAX_MONSTERS])


@feature_group("ready_skill_count", n_dims=1)
def ready_skill_count(world: GameWorld, world_size: float) -> np.ndarray:
    """
    Fraction of skills fully off cooldown (ratio == 1.0). 1D.
    0.0 = no skills ready, 1.0 = all 8 skills ready.
    Critic use: counts immediately castable skills, unlike mean which conflates
    "all skills half-ready" with "half of skills fully ready".
    """
    cooldowns = world.get_skill_cooldown_ratios()  # 8D, 1.0=ready
    count = sum(1 for r in cooldowns if r >= 1.0)
    return np.array([count / len(cooldowns)])


@feature_group("hp_advantage", n_dims=1)
def hp_advantage(world: GameWorld, world_size: float) -> np.ndarray:
    """
    player_health - total_monster_hp_normalized. 1D.
    Positive = player winning, negative = monsters winning.
    Critic use: single-value game state summary.
    """
    player_hp = world.get_player_health_percentage()
    data = _get_sorted_monster_data(world, world_size)
    monster_hp = sum(d[4] for d in data if d[0] > 0) / MAX_MONSTERS
    return np.array([player_hp - monster_hp])


@feature_group("blood_pack_count", n_dims=1)
def blood_pack_count(world: GameWorld, world_size: float) -> np.ndarray:
    """
    Number of alive blood packs, normalized by MAX_BLOOD_PACKS. 1D.
    0.0 = no blood packs, 1.0 = maximum blood packs present.
    """
    data = _get_sorted_blood_pack_data(world, world_size)
    count = sum(1 for d in data if d[0] > 0)
    return np.array([count / MAX_BLOOD_PACKS])


@feature_group("bias", n_dims=1)
def bias(world: GameWorld, world_size: float) -> np.ndarray:
    """Constant bias term. 1D."""
    return np.array([1.0])


@feature_group("movable_cast_state", n_dims=3)
def movable_cast_state(world: GameWorld, world_size: float) -> np.ndarray:
    """
    Binary flags for skills that allow movement during wind-up. 3D.
    [is_casting_outer_slash, is_casting_hammer, is_in_blood_pool]
    血池包含 wind-up 期間 (current_skill == "blood_pool") 和池中期間 (in_blood_pool)。
    """
    skills = world.player.skills
    current = skills.current_skill
    return np.array([
        1.0 if current == "outer_slash" else 0.0,
        1.0 if current == "hammer" else 0.0,
        1.0 if (current == "blood_pool" or skills.in_blood_pool) else 0.0,
    ])


def _get_enemy_projectiles(world: GameWorld):
    """Return active projectiles that belong to monsters (not the player)."""
    return [
        p for p in world.get_active_projectiles()
        if p.projectile_type != ProjectileType.SKILL_MISSILE
    ]


@feature_group("enemy_proj_count", n_dims=1)
def enemy_proj_count(world: GameWorld, world_size: float) -> np.ndarray:
    """
    Number of active enemy projectiles. 1D.
    Normalized by 8 (assumed max concurrent projectiles in typical encounters).
    """
    count = len(_get_enemy_projectiles(world))
    return np.array([count / 8.0])


@feature_group("nearest_enemy_proj", n_dims=4)
def nearest_enemy_proj(world: GameWorld, world_size: float) -> np.ndarray:
    """
    Nearest enemy projectile info. 4D: [dist, dx, dy, heading_toward].
    - dist: normalized distance (0-1)
    - dx, dy: relative position normalized by world_size
    - heading_toward: dot product of projectile direction with unit vector
      from projectile to player (1.0 = flying directly at player, -1.0 = flying away)
    Returns [1, 0, 0, 0] when no enemy projectile exists.
    """
    player_pos = world.get_player_position()
    enemy_projs = _get_enemy_projectiles(world)

    if not enemy_projs:
        return np.array([1.0, 0.0, 0.0, 0.0])

    nearest = min(enemy_projs, key=lambda p: np.linalg.norm(p.position - player_pos))

    diff = player_pos - nearest.position  # vector from proj to player
    dist = np.linalg.norm(diff)
    dx = (nearest.position[0] - player_pos[0]) / world_size
    dy = (nearest.position[1] - player_pos[1]) / world_size
    dist_norm = dist / world_size

    if dist > 1e-6:
        heading_toward = float(np.dot(nearest.direction, diff / dist))
    else:
        heading_toward = 0.0

    return np.array([dist_norm, dx, dy, heading_toward])


@feature_group("ring_hit_count", n_dims=1)
def ring_hit_count(world: GameWorld, world_size: float) -> np.ndarray:
    """
    Number of monsters within 外圈刮 (outer slash) ring AOE range. 1D.
    Reads inner_radius and outer_radius from world.player_config.
    Normalized by MAX_MONSTERS to [0, 1].
    """
    outer_slash = world.player_config.skills.get("outer_slash")
    inner = outer_slash.extra_params["inner_radius"] / world_size
    outer = outer_slash.extra_params["outer_radius"] / world_size
    data = _get_sorted_monster_data(world, world_size)
    count = sum(1 for d in data if inner <= d[0] <= outer)
    return np.array([count / MAX_MONSTERS])


@feature_group("monster_all", n_dims=16)
def monster_all(world: GameWorld, world_size: float) -> np.ndarray:
    """
    All monster features grouped by type. 16D.
    Output: [dist1..4, dx1,dy1..dx4,dy4, angle1..4]
    Equivalent to monster_dist + monster_dx_dy + monster_relative_angle.

    For FSS, prefer the fine-grained groups (monster_dist, monster_dx_dy, etc.)
    which let you test each feature type independently.
    """
    data = _get_sorted_monster_data(world, world_size)
    dists  = np.array([d[0] for d in data])
    dx_dys = np.array([v for d in data for v in (d[1], d[2])])
    angles = np.array([d[3] for d in data])
    return np.concatenate([dists, dx_dys, angles])


# ---------------------------------------------------------------------------
# Selective extractor
# ---------------------------------------------------------------------------

DEFAULT_GROUP_ORDER = [
    "monster_all",          # 16D
    "blood_pack_all",       # 12D
    "player_facing",        # 2D
    "wall_dist",            # 1D
    "casting_info",         # 9D
    "player_health",        # 1D
    "ring_hit_count",       # 1D
    "movable_cast_state",   # 3D
    "bias",                 # 1D
]                           # total = 46D


class SelectiveFeatureExtractor:
    """
    Builds a feature vector from a chosen subset of registered feature groups.
    Designed for use with FSS: each FSS trial creates a fresh instance with
    a different group selection.

    Args:
        group_names: Ordered list of feature group names to include.
        world_size: World size for normalization (default 10.0).
    """

    def __init__(self, group_names: List[str], world_size: float = 10.0):
        self.group_names = list(group_names)
        self.world_size = world_size
        self.groups = [get_group(name) for name in group_names]
        self.n_features = sum(g.n_dims for g in self.groups)

    def extract(self, world: GameWorld) -> np.ndarray:
        """Extract feature vector from current world state."""
        if world.player is None:
            return np.zeros(self.n_features)
        world._feature_cache = {}
        try:
            parts = [g.fn(world, self.world_size) for g in self.groups]
        finally:
            del world._feature_cache
        return np.concatenate(parts)

    def describe(self) -> str:
        """Human-readable summary of selected groups."""
        lines = [f"SelectiveFeatureExtractor ({self.n_features}D):"]
        offset = 0
        for g in self.groups:
            lines.append(f"  [{offset:>2} - {offset + g.n_dims - 1:<2}]  {g.name}  ({g.n_dims}D)")
            offset += g.n_dims
        return "\n".join(lines)
