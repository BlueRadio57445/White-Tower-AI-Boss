"""
Opportunist behavior - waits for Boss to start casting, then rushes in.
機會主義者：維持安全距離等待，Boss 蓄力時立刻衝進去攻擊，打完撤退
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
class OpportunistBehavior(MonsterBehavior):
    """
    機會主義者行為

    策略：
    - 維持安全距離繞圈等待
    - 偵測到 Boss 開始蓄力（casting progress 上升），立刻衝進去
    - 抵達攻擊範圍後攻擊
    - 攻擊後立即撤退，回到繞圈等待

    狀態機：
    CIRCLING → RUSHING（Boss 蓄力中）→ ATTACKING（進入攻擊範圍）→ RETREATING → CIRCLING
    """

    behavior_type = "opportunist"

    STATE_CIRCLING = "circling"
    STATE_RUSHING = "rushing"
    STATE_ATTACKING = "attacking"
    STATE_RETREATING = "retreating"

    def __init__(
        self,
        safe_radius: float = 5.0,         # 繞圈等待的安全距離
        radius_tolerance: float = 1.0,    # 距離容許誤差
        clockwise: bool = True,           # 繞圈方向
        casting_trigger: float = 0.1,     # casting progress 超過此值就衝進去
        retreat_distance: float = 5.0,    # 撤退到此距離後切換回繞圈
        **kwargs
    ):
        super().__init__(**kwargs)
        self.safe_radius = safe_radius
        self.radius_tolerance = radius_tolerance
        self.clockwise = clockwise
        self.casting_trigger = casting_trigger
        self.retreat_distance = retreat_distance

        self._internal_state["state"] = self.STATE_CIRCLING

    def decide_action(self, entity, world) -> MonsterAction:
        if world.player is None:
            return MonsterAction()

        agent_pos = world.player.position.as_array()
        monster_pos = entity.position.as_array()
        monster_angle = entity.angle

        angle_to_agent = self._get_angle_to_target(monster_pos, monster_angle, agent_pos)
        distance = self._get_distance_to_target(monster_pos, agent_pos)

        state = self._internal_state.get("state", self.STATE_CIRCLING)
        casting_progress = world.get_casting_progress()

        # 狀態轉換
        if state == self.STATE_CIRCLING:
            if casting_progress > self.casting_trigger:
                state = self.STATE_RUSHING
                self._internal_state["state"] = state

        elif state == self.STATE_RUSHING:
            if distance <= self.attack_range:
                state = self.STATE_ATTACKING
                self._internal_state["state"] = state
            elif casting_progress <= 0.0:
                # Boss 蓄力取消（技能打出去了），退回繞圈
                state = self.STATE_CIRCLING
                self._internal_state["state"] = state

        elif state == self.STATE_ATTACKING:
            if not self.can_attack():
                # 攻擊剛用掉，撤退
                state = self.STATE_RETREATING
                self._internal_state["state"] = state

        elif state == self.STATE_RETREATING:
            if distance >= self.retreat_distance:
                state = self.STATE_CIRCLING
                self._internal_state["state"] = state

        # 轉向永遠面向 Agent
        turning = self._decide_turning(angle_to_agent)

        if state == self.STATE_CIRCLING:
            return self._circling_action(angle_to_agent, distance, turning)
        elif state == self.STATE_RUSHING:
            return self._rushing_action(angle_to_agent, turning)
        elif state == self.STATE_ATTACKING:
            return self._attacking_action(angle_to_agent, distance, turning)
        else:  # RETREATING
            return self._retreating_action(angle_to_agent)

    def _circling_action(
        self, angle_to_agent: float, distance: float, turning: TurningType
    ) -> MonsterAction:
        """繞圈等待：維持安全距離橫移"""
        too_far = distance > self.safe_radius + self.radius_tolerance
        too_close = distance < self.safe_radius - self.radius_tolerance
        facing = self._is_facing_target(angle_to_agent, tolerance=0.5)

        if too_far and facing:
            movement = MovementType.FORWARD
        elif too_close and facing:
            movement = MovementType.BACKWARD
        else:
            movement = MovementType.RIGHT if self.clockwise else MovementType.LEFT

        return MonsterAction(movement=movement, turning=turning, attack=AttackType.NONE)

    def _rushing_action(self, angle_to_agent: float, turning: TurningType) -> MonsterAction:
        """衝鋒：全速衝向 Agent"""
        facing = self._is_facing_target(angle_to_agent, tolerance=0.5)
        movement = MovementType.SPRINT_FORWARD if facing else MovementType.IDLE
        return MonsterAction(movement=movement, turning=turning, attack=AttackType.NONE)

    def _attacking_action(
        self, angle_to_agent: float, distance: float, turning: TurningType
    ) -> MonsterAction:
        """攻擊：停下來，面向並攻擊"""
        attack = AttackType.NONE
        if distance <= self.attack_range and self._is_facing_target(angle_to_agent):
            attack = AttackType.ATTACK
        return MonsterAction(movement=MovementType.IDLE, turning=turning, attack=attack)

    def _retreating_action(self, angle_to_agent: float) -> MonsterAction:
        """撤退：背對 Agent 奔跑"""
        # 計算背對方向的轉向
        if abs(angle_to_agent) < 1.5:  # 接近面向 Agent，用 180 度掉頭
            turning = TurningType.TURN_180
        else:
            angle_away = angle_to_agent - np.pi if angle_to_agent > 0 else angle_to_agent + np.pi
            turning = self._decide_turning(angle_away)

        # 已背對 Agent 時才奔跑
        movement = MovementType.SPRINT_FORWARD if abs(angle_to_agent) > 2.0 else MovementType.IDLE
        return MonsterAction(movement=movement, turning=turning, attack=AttackType.NONE)

    def reset(self):
        super().reset()
        self._internal_state["state"] = self.STATE_CIRCLING

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["safe_radius"] = self.safe_radius
        data["radius_tolerance"] = self.radius_tolerance
        data["clockwise"] = self.clockwise
        data["casting_trigger"] = self.casting_trigger
        data["retreat_distance"] = self.retreat_distance
        return data
