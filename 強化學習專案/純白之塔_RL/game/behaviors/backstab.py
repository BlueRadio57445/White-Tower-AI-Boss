"""
Backstab behavior - always tries to position behind the Boss and attack.
繞背型：永遠試圖移動到 Boss 背後，只在背後才攻擊
"""

from typing import Dict, Any
import numpy as np

from game.behaviors.base import (
    MonsterBehavior,
    MonsterAction,
    MovementType,
    TurningType,
    AttackType,
    BehaviorRegistry,
)


@BehaviorRegistry.register
class BackstabBehavior(MonsterBehavior):
    """
    繞背型行為

    策略：
    - 永遠試圖移動到 Boss 的後方
    - 只有位於 Boss 後方扇形範圍內才攻擊
    - 純反應式，不需要複雜狀態機

    核心邏輯：
    - 計算 Boss 背後的目標位置
    - 判斷自己是否已在後方（Boss 朝向角度 ± 容差）
    - 在後方且在攻擊範圍時攻擊，否則繞到背後
    """

    behavior_type = "backstab"

    def __init__(
        self,
        backstab_distance: float = 2.0,          # 在 Boss 背後的目標停留距離
        backstab_tolerance: float = 0.7,          # 判斷「在後方」的角度容差（弧度）
        approach_sprint_threshold: float = 5.0,   # 超過此距離才奔跑
        **kwargs
    ):
        super().__init__(**kwargs)
        self.backstab_distance = backstab_distance
        self.backstab_tolerance = backstab_tolerance
        self.approach_sprint_threshold = approach_sprint_threshold

    def _is_behind_player(
        self, monster_pos: np.ndarray, player_pos: np.ndarray, player_angle: float
    ) -> bool:
        """判斷怪物是否在玩家背後"""
        vec_to_monster = monster_pos - player_pos
        dist = np.linalg.norm(vec_to_monster)
        if dist < 0.01:
            return False

        # 玩家背後方向角度
        back_angle = player_angle + np.pi
        vec_angle = np.arctan2(vec_to_monster[1], vec_to_monster[0])

        angle_diff = vec_angle - back_angle
        while angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        while angle_diff < -np.pi:
            angle_diff += 2 * np.pi

        return abs(angle_diff) < self.backstab_tolerance

    def _get_target_behind_player(
        self, player_pos: np.ndarray, player_angle: float
    ) -> np.ndarray:
        """計算 Boss 背後的目標位置"""
        back_dir = np.array([-np.cos(player_angle), -np.sin(player_angle)])
        return player_pos + back_dir * self.backstab_distance

    def decide_action(self, entity, world) -> MonsterAction:
        if world.player is None:
            return MonsterAction()

        player_pos = world.player.position.as_array()
        player_angle = world.player.entity.angle
        monster_pos = entity.position.as_array()
        monster_angle = entity.angle

        distance_to_player = self._get_distance_to_target(monster_pos, player_pos)
        is_behind = self._is_behind_player(monster_pos, player_pos, player_angle)

        if is_behind and distance_to_player <= self.attack_range:
            # 已在背後且在攻擊範圍內：面向 Boss 並攻擊
            angle_to_player = self._get_angle_to_target(monster_pos, monster_angle, player_pos)
            turning = self._decide_turning(angle_to_player)
            attack = AttackType.ATTACK if self._is_facing_target(angle_to_player) else AttackType.NONE
            return MonsterAction(movement=MovementType.IDLE, turning=turning, attack=attack)
        else:
            # 移向 Boss 背後的目標位置
            target_pos = self._get_target_behind_player(player_pos, player_angle)
            angle_to_target = self._get_angle_to_target(monster_pos, monster_angle, target_pos)
            distance_to_target = self._get_distance_to_target(monster_pos, target_pos)

            turning = self._decide_turning(angle_to_target)

            movement = MovementType.IDLE
            if self._is_facing_target(angle_to_target, tolerance=0.5):
                if distance_to_target > self.approach_sprint_threshold:
                    movement = MovementType.SPRINT_FORWARD
                elif distance_to_target > 0.5:
                    movement = MovementType.FORWARD

            return MonsterAction(movement=movement, turning=turning, attack=AttackType.NONE)

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["backstab_distance"] = self.backstab_distance
        data["backstab_tolerance"] = self.backstab_tolerance
        data["approach_sprint_threshold"] = self.approach_sprint_threshold
        return data
