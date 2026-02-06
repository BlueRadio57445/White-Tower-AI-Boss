"""
Hit-and-run behavior - attack once then flee.
偷傷害打法：打到 Agent 一下就逃跑
"""

from typing import Dict, Any

from game.behaviors.base import (
    MonsterBehavior,
    MonsterAction,
    MovementType,
    TurningType,
    AttackType,
    BehaviorRegistry
)


@BehaviorRegistry.register
class HitAndRunBehavior(MonsterBehavior):
    """
    偷傷害行為

    策略：
    - 接近 Agent
    - 攻擊一次
    - 立即 180 度掉頭 + 奔跑逃離
    - 逃離一段距離後再次接近

    狀態機：
    - APPROACHING: 接近 Agent
    - ATTACKING: 準備攻擊
    - FLEEING: 逃跑中
    """

    behavior_type = "hit_and_run"

    # 狀態常數
    STATE_APPROACHING = "approaching"
    STATE_ATTACKING = "attacking"
    STATE_FLEEING = "fleeing"

    def __init__(
        self,
        flee_distance: float = 6.0,     # 逃到這個距離才停
        safe_distance: float = 7.0,     # 超過這個距離開始接近
        flee_duration: int = 30,        # 逃跑最少持續的 ticks
        **kwargs
    ):
        super().__init__(**kwargs)
        self.flee_distance = flee_distance
        self.safe_distance = safe_distance
        self.flee_duration = flee_duration

        # 初始化狀態
        self._internal_state["state"] = self.STATE_APPROACHING
        self._internal_state["flee_ticks"] = 0

    def decide_action(self, entity, world) -> MonsterAction:
        if world.player is None:
            return MonsterAction()

        agent_pos = world.player.position.as_array()
        monster_pos = entity.position.as_array()
        monster_angle = entity.angle

        # 計算角度差和距離
        angle_to_agent = self._get_angle_to_target(monster_pos, monster_angle, agent_pos)
        distance = self._get_distance_to_target(monster_pos, agent_pos)

        # 取得當前狀態
        state = self._internal_state.get("state", self.STATE_APPROACHING)
        flee_ticks = self._internal_state.get("flee_ticks", 0)

        # 狀態轉換邏輯
        if state == self.STATE_FLEEING:
            flee_ticks += 1
            self._internal_state["flee_ticks"] = flee_ticks

            # 逃夠遠且夠久了，開始接近
            if distance >= self.safe_distance and flee_ticks >= self.flee_duration:
                state = self.STATE_APPROACHING
                self._internal_state["state"] = state

        elif state == self.STATE_APPROACHING:
            # 進入攻擊範圍，準備攻擊
            if distance <= self.attack_range:
                state = self.STATE_ATTACKING
                self._internal_state["state"] = state

        elif state == self.STATE_ATTACKING:
            # 攻擊後立即進入逃跑狀態
            if not self.can_attack():  # 攻擊剛用掉
                state = self.STATE_FLEEING
                self._internal_state["state"] = state
                self._internal_state["flee_ticks"] = 0

        # 根據狀態決定行動
        if state == self.STATE_APPROACHING:
            return self._approaching_action(angle_to_agent, distance)
        elif state == self.STATE_ATTACKING:
            return self._attacking_action(angle_to_agent, distance)
        else:  # FLEEING
            return self._fleeing_action(angle_to_agent, distance)

    def _approaching_action(self, angle_to_agent: float, distance: float) -> MonsterAction:
        """接近模式：面向 Agent，奔跑接近"""
        turning = self._decide_turning(angle_to_agent)

        movement = MovementType.IDLE
        if self._is_facing_target(angle_to_agent, tolerance=0.5):
            if distance > self.attack_range + 2:
                movement = MovementType.SPRINT_FORWARD
            else:
                movement = MovementType.FORWARD

        return MonsterAction(
            movement=movement,
            turning=turning,
            attack=AttackType.NONE
        )

    def _attacking_action(self, angle_to_agent: float, distance: float) -> MonsterAction:
        """攻擊模式：面向 Agent，攻擊"""
        turning = self._decide_turning(angle_to_agent)

        attack = AttackType.NONE
        if distance <= self.attack_range and self._is_facing_target(angle_to_agent):
            attack = AttackType.ATTACK

        return MonsterAction(
            movement=MovementType.FORWARD,  # 持續靠近
            turning=turning,
            attack=attack
        )

    def _fleeing_action(self, angle_to_agent: float, distance: float) -> MonsterAction:
        """逃跑模式：背對 Agent，奔跑逃離"""
        # 計算背對 Agent 需要轉多少
        # 如果 angle_to_agent 接近 0，要轉 180 度
        # 如果 angle_to_agent 接近 ±π，已經背對了

        angle_away = angle_to_agent
        if angle_away > 0:
            angle_away = angle_away - 3.14159
        else:
            angle_away = angle_away + 3.14159

        # 如果幾乎面向 Agent，用 180 度掉頭
        if abs(angle_to_agent) < 1.5:  # 接近面向
            turning = TurningType.TURN_180
        else:
            turning = self._decide_turning(angle_away)

        # 如果已經背對 Agent，奔跑
        movement = MovementType.IDLE
        if abs(angle_to_agent) > 2.0:  # 大致背對
            movement = MovementType.SPRINT_FORWARD
        elif abs(angle_to_agent) > 1.0:
            movement = MovementType.FORWARD

        return MonsterAction(
            movement=movement,
            turning=turning,
            attack=AttackType.NONE
        )

    def reset(self):
        super().reset()
        self._internal_state["state"] = self.STATE_APPROACHING
        self._internal_state["flee_ticks"] = 0

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["flee_distance"] = self.flee_distance
        data["safe_distance"] = self.safe_distance
        data["flee_duration"] = self.flee_duration
        return data
