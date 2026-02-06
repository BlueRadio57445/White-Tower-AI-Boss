"""
Berserker behavior - rushes at Agent and fights to the death.
狂戰士打法：衝上去跟 Agent 拚個你死我活
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
class BerserkerBehavior(MonsterBehavior):
    """
    狂戰士行為

    策略：
    - 永遠朝 Agent 衝鋒
    - 距離遠時奔跑接近
    - 近距離時走路 + 持續攻擊
    - 永不後退
    """

    behavior_type = "berserker"

    def __init__(
        self,
        sprint_threshold: float = 5.0,  # 超過此距離就奔跑
        **kwargs
    ):
        super().__init__(**kwargs)
        self.sprint_threshold = sprint_threshold

    def decide_action(self, entity, world) -> MonsterAction:
        if world.player is None:
            return MonsterAction()

        agent_pos = world.player.position.as_array()
        monster_pos = entity.position.as_array()
        monster_angle = entity.angle

        # 計算角度差和距離
        angle_diff = self._get_angle_to_target(monster_pos, monster_angle, agent_pos)
        distance = self._get_distance_to_target(monster_pos, agent_pos)

        # 決定轉向：永遠面向 Agent
        turning = self._decide_turning(angle_diff)

        # 決定移動：面向 Agent 時才前進
        movement = MovementType.IDLE
        if self._is_facing_target(angle_diff, tolerance=0.5):
            if distance > self.sprint_threshold:
                movement = MovementType.SPRINT_FORWARD
            else:
                movement = MovementType.FORWARD

        # 決定攻擊：近距離且面向時攻擊
        attack = AttackType.NONE
        if distance <= self.attack_range and self._is_facing_target(angle_diff):
            attack = AttackType.ATTACK

        return MonsterAction(
            movement=movement,
            turning=turning,
            attack=attack
        )

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["sprint_threshold"] = self.sprint_threshold
        return data
