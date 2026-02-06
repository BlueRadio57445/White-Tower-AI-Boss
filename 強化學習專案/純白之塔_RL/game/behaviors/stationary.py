"""
Stationary behavior - stands still but faces and attacks the Agent.
"""

from game.behaviors.base import (
    MonsterBehavior,
    MonsterAction,
    MovementType,
    TurningType,
    AttackType,
    BehaviorRegistry
)


@BehaviorRegistry.register
class StationaryBehavior(MonsterBehavior):
    """
    站樁行為

    - 不移動
    - 持續面向 Agent
    - 在攻擊範圍內時攻擊
    """

    behavior_type = "stationary"

    def decide_action(self, entity, world) -> MonsterAction:
        if world.player is None:
            return MonsterAction()

        agent_pos = world.player.position.as_array()
        monster_pos = entity.position.as_array()
        monster_angle = entity.angle

        # 計算與 Agent 的角度差和距離
        angle_diff = self._get_angle_to_target(monster_pos, monster_angle, agent_pos)
        distance = self._get_distance_to_target(monster_pos, agent_pos)

        # 決定轉向
        turning = self._decide_turning(angle_diff)

        # 決定攻擊（在範圍內且面向 Agent）
        attack = AttackType.NONE
        if distance <= self.attack_range and self._is_facing_target(angle_diff):
            attack = AttackType.ATTACK

        return MonsterAction(
            movement=MovementType.IDLE,
            turning=turning,
            attack=attack
        )
