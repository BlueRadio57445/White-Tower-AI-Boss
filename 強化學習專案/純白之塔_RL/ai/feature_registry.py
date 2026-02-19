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
    monster_angle_sincos      8D   (sin, cos) of relative angle to each monster (nearest first)
    monster_all              16D   all monster features grouped by type (dist ×4, dx_dy ×4, angle ×4)
    blood_pack_dist           3D   distance to each blood pack (nearest first)
    blood_pack_dx_dy          6D   (dx, dy) to each blood pack (nearest first)
    blood_pack_relative_angle 3D   relative angle to each blood pack (nearest first)
    blood_pack_angle_sincos   6D   (sin, cos) of relative angle to each blood pack (nearest first)
    blood_pack_all           12D   all blood pack features grouped by type (dist ×3, dx_dy ×3, angle ×3)
    player_facing             2D   (cos, sin) of player facing angle
    wall_dist                 1D   distance to wall in facing direction
    casting_info              2D   (casting_progress, ready_to_cast)
    player_health             1D   player health ratio (0-1)
    bias                      1D   constant 1.0
    -------------------------------------------------------
    ALL GROUPS               35D   (DEFAULT_GROUP_ORDER total)
"""

from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np

from game.world import GameWorld

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

    Returns list of (dist, dx, dy, rel_angle) tuples,
    zero-padded to MAX_MONSTERS entries.
    """
    player_pos = world.get_player_position()
    player_angle = world.get_player_angle()
    monster_positions = world.get_alive_monster_positions()

    data = []
    for pos in monster_positions:
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
    while len(data) < MAX_MONSTERS:
        data.append((0.0, 0.0, 0.0, 0.0))

    return data[:MAX_MONSTERS]


def _get_sorted_blood_pack_data(world: GameWorld, world_size: float):
    """
    Compute and sort blood pack data by distance (nearest first).

    Returns list of (dist, dx, dy, rel_angle) tuples,
    zero-padded to MAX_BLOOD_PACKS entries.
    """
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

    return data[:MAX_BLOOD_PACKS]


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


@feature_group("casting_info", n_dims=2)
def casting_info(world: GameWorld, world_size: float) -> np.ndarray:
    """[casting_progress, ready_to_cast]. 2D."""
    progress = world.get_casting_progress()
    ready = 1.0 if world.is_player_ready_to_cast() else 0.0
    return np.array([progress, ready])


@feature_group("player_health", n_dims=1)
def player_health(world: GameWorld, world_size: float) -> np.ndarray:
    """Player health ratio (0-1). 1D."""
    return np.array([world.get_player_health_percentage()])


@feature_group("bias", n_dims=1)
def bias(world: GameWorld, world_size: float) -> np.ndarray:
    """Constant bias term. 1D."""
    return np.array([1.0])


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
    "monster_all",          # 16D  [dist1,dx1,dy1,angle1, ...] interleaved
    "blood_pack_all",       # 12D  [dist1,dx1,dy1,angle1, ...] interleaved
    "player_facing",        # 2D
    "wall_dist",            # 1D
    "casting_info",         # 2D
    "player_health",        # 1D
    "bias",                 # 1D
]                           # total = 35D


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
        parts = [g.fn(world, self.world_size) for g in self.groups]
        return np.concatenate(parts)

    def describe(self) -> str:
        """Human-readable summary of selected groups."""
        lines = [f"SelectiveFeatureExtractor ({self.n_features}D):"]
        offset = 0
        for g in self.groups:
            lines.append(f"  [{offset:>2} - {offset + g.n_dims - 1:<2}]  {g.name}  ({g.n_dims}D)")
            offset += g.n_dims
        return "\n".join(lines)
