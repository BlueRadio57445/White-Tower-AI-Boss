"""
Orbit behaviors - circling around the Agent while attacking.
繞圈圈打法：在 Agent 周圍做圓周運動
"""

from typing import Dict, Any
import numpy as np

from game.behaviors.base import (
    MonsterBehavior,
    MonsterAction,
    MovementType,
    TurningType,
    AttackType,
    BehaviorRegistry
)


@BehaviorRegistry.register
class OrbitMeleeBehavior(MonsterBehavior):
    """
    近戰繞圈行為

    策略：
    - 維持在 Agent 周圍的短距離
    - 持續橫移（strafe）做圓周運動
    - 同時面向 Agent 並攻擊

    Minecraft 玩家常用的近戰 PvP 技巧
    """

    behavior_type = "orbit_melee"

    def __init__(
        self,
        target_radius: float = 2.5,     # 目標繞行半徑
        radius_tolerance: float = 1.0,   # 半徑容許誤差
        clockwise: bool = True,          # 順時針或逆時針
        strafe_while_approaching: bool = True,  # 接近時也橫移
        **kwargs
    ):
        # 近戰設定
        kwargs.setdefault("attack_range", 3.0)
        kwargs.setdefault("attack_damage", 12.0)
        kwargs.setdefault("attack_cooldown_ticks", 20)

        super().__init__(**kwargs)
        self.target_radius = target_radius
        self.radius_tolerance = radius_tolerance
        self.clockwise = clockwise
        self.strafe_while_approaching = strafe_while_approaching

    def decide_action(self, entity, world) -> MonsterAction:
        if world.player is None:
            return MonsterAction()

        agent_pos = world.player.position.as_array()
        monster_pos = entity.position.as_array()
        monster_angle = entity.angle

        # 計算角度差和距離
        angle_to_agent = self._get_angle_to_target(monster_pos, monster_angle, agent_pos)
        distance = self._get_distance_to_target(monster_pos, agent_pos)

        # 決定轉向：永遠面向 Agent
        turning = self._decide_turning(angle_to_agent)

        # 決定移動
        movement = self._decide_movement(angle_to_agent, distance)

        # 決定攻擊
        attack = AttackType.NONE
        if distance <= self.attack_range and self._is_facing_target(angle_to_agent, tolerance=0.4):
            attack = AttackType.ATTACK

        return MonsterAction(
            movement=movement,
            turning=turning,
            attack=attack
        )

    def _decide_movement(self, angle_to_agent: float, distance: float) -> MovementType:
        """決定移動方式"""
        too_far = distance > self.target_radius + self.radius_tolerance
        too_close = distance < self.target_radius - self.radius_tolerance
        facing = self._is_facing_target(angle_to_agent, tolerance=0.5)

        if too_far:
            # 太遠：前進接近
            if facing:
                if self.strafe_while_approaching:
                    # 斜向接近（前進 + 橫移的效果）
                    return MovementType.FORWARD
                else:
                    return MovementType.SPRINT_FORWARD
            return MovementType.IDLE

        elif too_close:
            # 太近：後退
            if facing:
                return MovementType.BACKWARD
            return MovementType.IDLE

        else:
            # 適當距離：橫移繞圈
            if self.clockwise:
                return MovementType.RIGHT
            else:
                return MovementType.LEFT

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["target_radius"] = self.target_radius
        data["radius_tolerance"] = self.radius_tolerance
        data["clockwise"] = self.clockwise
        data["strafe_while_approaching"] = self.strafe_while_approaching
        return data


@BehaviorRegistry.register
class OrbitRangedBehavior(MonsterBehavior):
    """
    遠程繞圈行為（Kiting）

    策略：
    - 維持在 Agent 周圍的長距離
    - 持續橫移做圓周運動
    - 面向 Agent 時射擊
    - Agent 靠近時後退

    弓箭手/法師的標準打法
    """

    behavior_type = "orbit_ranged"

    # 武器類型
    WEAPON_BOW = "bow"
    WEAPON_STAFF = "staff"

    def __init__(
        self,
        target_radius: float = 6.0,      # 目標繞行半徑（較遠）
        radius_tolerance: float = 1.5,
        danger_radius: float = 4.0,      # 低於此距離要緊急後退
        clockwise: bool = True,
        weapon_type: str = "bow",
        **kwargs
    ):
        # 遠程設定
        if weapon_type == self.WEAPON_BOW:
            kwargs.setdefault("attack_range", 8.0)
            kwargs.setdefault("attack_damage", 15.0)
            kwargs.setdefault("attack_cooldown_ticks", 30)
        else:  # staff
            kwargs.setdefault("attack_range", 10.0)
            kwargs.setdefault("attack_damage", 20.0)
            kwargs.setdefault("attack_cooldown_ticks", 40)

        super().__init__(**kwargs)
        self.target_radius = target_radius
        self.radius_tolerance = radius_tolerance
        self.danger_radius = danger_radius
        self.clockwise = clockwise
        self.weapon_type = weapon_type

    def decide_action(self, entity, world) -> MonsterAction:
        if world.player is None:
            return MonsterAction()

        agent_pos = world.player.position.as_array()
        monster_pos = entity.position.as_array()
        monster_angle = entity.angle

        # 計算角度差和距離
        angle_to_agent = self._get_angle_to_target(monster_pos, monster_angle, agent_pos)
        distance = self._get_distance_to_target(monster_pos, agent_pos)

        # 決定轉向：永遠面向 Agent
        turning = self._decide_turning(angle_to_agent)

        # 決定移動
        movement = self._decide_movement(angle_to_agent, distance)

        # 決定攻擊（遠程射擊）
        attack = AttackType.NONE
        if (distance <= self.attack_range and
            self._is_facing_target(angle_to_agent, tolerance=0.3)):
            attack = AttackType.ATTACK

        return MonsterAction(
            movement=movement,
            turning=turning,
            attack=attack
        )

    def _decide_movement(self, angle_to_agent: float, distance: float) -> MovementType:
        """決定移動方式"""
        in_danger = distance < self.danger_radius
        too_far = distance > self.target_radius + self.radius_tolerance
        too_close = distance < self.target_radius - self.radius_tolerance
        facing = self._is_facing_target(angle_to_agent, tolerance=0.5)

        if in_danger:
            # 危險！Agent 太近，緊急後退
            if facing:
                return MovementType.BACKWARD
            else:
                # 沒面向 Agent，先轉身
                return MovementType.IDLE

        elif too_close:
            # 有點近：後退 + 橫移
            if facing:
                return MovementType.BACKWARD
            return MovementType.IDLE

        elif too_far:
            # 太遠：接近
            if facing:
                return MovementType.FORWARD
            return MovementType.IDLE

        else:
            # 適當距離：橫移繞圈
            if self.clockwise:
                return MovementType.RIGHT
            else:
                return MovementType.LEFT

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["target_radius"] = self.target_radius
        data["radius_tolerance"] = self.radius_tolerance
        data["danger_radius"] = self.danger_radius
        data["clockwise"] = self.clockwise
        data["weapon_type"] = self.weapon_type
        return data
