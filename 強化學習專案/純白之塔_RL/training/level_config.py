"""
Level configuration and loader for map-based training.

JSON format:
{
    "id": "level_id",
    "name": "Level Name",
    "room": { "size": 10.0 },
    "player": { "x": 8.0, "y": 5.0, "angle": 3.14159 },
    "monsters": [
        {
            "x": 5.0, "y": 5.0, "health": 100.0,
            "behavior": { "type": "stationary", "attack_damage": 10.0 }
        }
    ],
    "blood_packs": [
        { "x": 2.0, "y": 2.0, "heal_amount": 30.0 }
    ]
}

Behavior types: stationary, berserker, hit_and_run, orbit_melee, orbit_ranged,
                opportunist, adaptive, backstab,
                blood_pack_disruptor_melee, blood_pack_disruptor_ranged
"""

import json
import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class LevelConfig:
    """
    Level configuration loaded from JSON.

    Attributes:
        id: Unique level identifier
        name: Human-readable name
        room_size: World dimensions (square, units)
        player_x: Player spawn X coordinate
        player_y: Player spawn Y coordinate
        player_angle: Player initial facing angle (radians)
        monsters: List of monster specs, each with x, y, health, behavior dict
        blood_packs: List of blood pack specs, each with x, y, heal_amount
    """
    id: str = "default"
    name: str = ""
    room_size: float = 10.0
    player_x: float = 8.0
    player_y: float = 5.0
    player_angle: float = math.pi
    monsters: List[Dict[str, Any]] = field(default_factory=list)
    blood_packs: List[Dict[str, Any]] = field(default_factory=list)


class LevelLoader:
    """Loads LevelConfig from JSON files or dictionaries."""

    @staticmethod
    def from_json(path: str) -> LevelConfig:
        """
        Load a LevelConfig from a JSON file.

        Args:
            path: Path to the JSON file

        Returns:
            LevelConfig instance

        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the JSON is missing required fields
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return LevelLoader.from_dict(data)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> LevelConfig:
        """
        Build a LevelConfig from a dictionary.

        Args:
            data: Dictionary with level configuration

        Returns:
            LevelConfig instance
        """
        room = data.get("room", {})
        player = data.get("player", {})

        config = LevelConfig(
            id=data.get("id", "default"),
            name=data.get("name", ""),
            room_size=float(room.get("size", 10.0)),
            player_x=float(player.get("x", 8.0)),
            player_y=float(player.get("y", 5.0)),
            player_angle=float(player.get("angle", math.pi)),
            monsters=data.get("monsters", []),
            blood_packs=data.get("blood_packs", []),
        )

        return config
