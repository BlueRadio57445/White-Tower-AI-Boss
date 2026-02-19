"""
Blood pack disruptor behaviors - guard blood packs to deny Boss healing.
血包干擾型：守住血包阻止 Boss 回血，同時伺機攻擊

Classes:
- BloodPackDisruptorMeleeBehavior: 近戰版，FIGHTING 委派給 OrbitMeleeBehavior
- BloodPackDisruptorRangedBehavior: 遠程版，FIGHTING 委派給 OrbitRangedBehavior，
  帶有 weapon_type 屬性，world.py 依此生成投射物

注意：world.get_blood_pack_position() 無血包時回傳 (0,0) 而非 None，
因此必須用 world.get_blood_pack_count() > 0 判斷血包是否存在。
"""

from typing import Dict, Any, Optional
import numpy as np

from game.behaviors.base import (
    MonsterBehavior,
    MonsterAction,
    MovementType,
    TurningType,
    AttackType,
    BehaviorRegistry,
)
from game.behaviors.orbit import OrbitMeleeBehavior, OrbitRangedBehavior


# =============================================================================
# 共用狀態機基類（不對外暴露，不需要 behavior_type）
# =============================================================================

class _BloodPackDisruptorBase(MonsterBehavior):
    """
    血包干擾型行為的共用狀態機。

    狀態機：
    FIGHTING（無血包）→ RACING（血包生成，衝過去）→ GUARDING（守在血包旁攻擊）→ FIGHTING（血包消失）
    """

    STATE_FIGHTING = "fighting"
    STATE_RACING = "racing"
    STATE_GUARDING = "guarding"

    def __init__(
        self,
        guard_radius: float = 1.5,          # 守衛血包的有效半徑
        attack_while_guarding: bool = True,  # 守血包時是否攻擊 Boss
        **kwargs
    ):
        super().__init__(**kwargs)
        self.guard_radius = guard_radius
        self.attack_while_guarding = attack_while_guarding
        self._internal_state["state"] = self.STATE_FIGHTING

    def decide_action(self, entity, world) -> MonsterAction:
        if world.player is None:
            return MonsterAction()

        agent_pos = world.player.position.as_array()
        monster_pos = entity.position.as_array()
        monster_angle = entity.angle

        has_blood_pack = world.get_blood_pack_count() > 0
        blood_pack_pos = world.get_blood_pack_position() if has_blood_pack else None
        state = self._internal_state.get("state", self.STATE_FIGHTING)

        # 狀態轉換
        if state == self.STATE_FIGHTING:
            if has_blood_pack:
                state = self.STATE_RACING
                self._internal_state["state"] = state

        elif state == self.STATE_RACING:
            if not has_blood_pack:
                state = self.STATE_FIGHTING
                self._internal_state["state"] = state
            else:
                if self._get_distance_to_target(monster_pos, blood_pack_pos) <= self.guard_radius:
                    state = self.STATE_GUARDING
                    self._internal_state["state"] = state

        elif state == self.STATE_GUARDING:
            if not has_blood_pack:
                state = self.STATE_FIGHTING
                self._internal_state["state"] = state

        # 根據狀態決定行動
        if state == self.STATE_FIGHTING:
            return self._fighting_action(monster_pos, monster_angle, agent_pos)
        elif state == self.STATE_RACING:
            return self._racing_action(monster_pos, monster_angle, blood_pack_pos, agent_pos)
        else:  # GUARDING
            return self._guarding_action(monster_pos, monster_angle, blood_pack_pos, agent_pos)

    def _fighting_action(
        self, monster_pos: np.ndarray, monster_angle: float, agent_pos: np.ndarray
    ) -> MonsterAction:
        raise NotImplementedError

    def _racing_action(
        self,
        monster_pos: np.ndarray,
        monster_angle: float,
        blood_pack_pos: np.ndarray,
        agent_pos: np.ndarray
    ) -> MonsterAction:
        """衝向血包（近戰/遠程共用）"""
        angle_to_pack = self._get_angle_to_target(monster_pos, monster_angle, blood_pack_pos)
        turning = self._decide_turning(angle_to_pack)
        movement = MovementType.SPRINT_FORWARD if self._is_facing_target(angle_to_pack, tolerance=0.5) else MovementType.IDLE

        # 衝途中若 Boss 進入攻擊範圍也順手打一下
        attack = AttackType.NONE
        if self.attack_while_guarding:
            angle_to_agent = self._get_angle_to_target(monster_pos, monster_angle, agent_pos)
            dist_to_agent = self._get_distance_to_target(monster_pos, agent_pos)
            if dist_to_agent <= self.attack_range and self._is_facing_target(angle_to_agent):
                attack = AttackType.ATTACK

        return MonsterAction(movement=movement, turning=turning, attack=attack)

    def _guarding_action(
        self,
        monster_pos: np.ndarray,
        monster_angle: float,
        blood_pack_pos: Optional[np.ndarray],
        agent_pos: np.ndarray
    ) -> MonsterAction:
        raise NotImplementedError

    def reset(self):
        super().reset()
        self._internal_state["state"] = self.STATE_FIGHTING

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["guard_radius"] = self.guard_radius
        data["attack_while_guarding"] = self.attack_while_guarding
        return data


# =============================================================================
# 近戰版
# =============================================================================

@BehaviorRegistry.register
class BloodPackDisruptorMeleeBehavior(_BloodPackDisruptorBase):
    """
    血包干擾型（近戰版）

    FIGHTING 模式：委派給 OrbitMeleeBehavior
    GUARDING 模式：守在血包旁，等 Boss 進入近戰範圍時攻擊
    """

    behavior_type = "blood_pack_disruptor_melee"

    def __init__(
        self,
        guard_radius: float = 1.5,
        attack_while_guarding: bool = True,
        # OrbitMeleeBehavior 參數（名稱與原類別完全一致）
        target_radius: float = 2.5,
        radius_tolerance: float = 1.0,
        clockwise: bool = True,
        strafe_while_approaching: bool = True,
        **kwargs
    ):
        kwargs.setdefault("attack_range", 3.0)
        kwargs.setdefault("attack_damage", 10.0)
        kwargs.setdefault("attack_cooldown_ticks", 20)
        super().__init__(guard_radius=guard_radius, attack_while_guarding=attack_while_guarding, **kwargs)

        self._orbit = OrbitMeleeBehavior(
            target_radius=target_radius,
            radius_tolerance=radius_tolerance,
            clockwise=clockwise,
            strafe_while_approaching=strafe_while_approaching,
            attack_range=self.attack_range,
            attack_damage=self.attack_damage,
            attack_cooldown_ticks=self.attack_cooldown_ticks,
        )

    def _fighting_action(self, monster_pos, monster_angle, agent_pos) -> MonsterAction:
        """戰鬥模式：委派給 OrbitMeleeBehavior 的移動邏輯"""
        angle_to_agent = self._get_angle_to_target(monster_pos, monster_angle, agent_pos)
        distance = self._get_distance_to_target(monster_pos, agent_pos)
        turning = self._decide_turning(angle_to_agent)
        movement = self._orbit._decide_movement(angle_to_agent, distance)

        attack = AttackType.NONE
        if distance <= self.attack_range and self._is_facing_target(angle_to_agent, tolerance=0.4):
            attack = AttackType.ATTACK

        return MonsterAction(movement=movement, turning=turning, attack=attack)

    def _guarding_action(self, monster_pos, monster_angle, blood_pack_pos, agent_pos) -> MonsterAction:
        """守衛血包（近戰版）：守在血包旁，Boss 靠近時攻擊"""
        if blood_pack_pos is None:
            return MonsterAction()

        dist_to_pack = self._get_distance_to_target(monster_pos, blood_pack_pos)
        angle_to_agent = self._get_angle_to_target(monster_pos, monster_angle, agent_pos)
        dist_to_agent = self._get_distance_to_target(monster_pos, agent_pos)

        turning = self._decide_turning(angle_to_agent)
        movement = MovementType.IDLE

        if dist_to_pack > self.guard_radius:
            # 跑太遠了，優先回到血包旁
            angle_to_pack = self._get_angle_to_target(monster_pos, monster_angle, blood_pack_pos)
            turning = self._decide_turning(angle_to_pack)
            if self._is_facing_target(angle_to_pack, tolerance=0.5):
                movement = MovementType.SPRINT_FORWARD
        elif (dist_to_agent > self.attack_range
              and dist_to_pack < self.guard_radius * 0.5
              and self._is_facing_target(angle_to_agent, tolerance=0.5)):
            # 仍在血包旁，可以稍微靠近攻擊
            movement = MovementType.FORWARD

        attack = AttackType.NONE
        if (self.attack_while_guarding
                and dist_to_agent <= self.attack_range
                and self._is_facing_target(angle_to_agent, tolerance=0.4)):
            attack = AttackType.ATTACK

        return MonsterAction(movement=movement, turning=turning, attack=attack)

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["target_radius"] = self._orbit.target_radius
        data["radius_tolerance"] = self._orbit.radius_tolerance
        data["clockwise"] = self._orbit.clockwise
        data["strafe_while_approaching"] = self._orbit.strafe_while_approaching
        return data


# =============================================================================
# 遠程版
# =============================================================================

@BehaviorRegistry.register
class BloodPackDisruptorRangedBehavior(_BloodPackDisruptorBase):
    """
    血包干擾型（遠程版）

    FIGHTING 模式：委派給 OrbitRangedBehavior（kiting 走位）
    GUARDING 模式：守在血包旁，持續向 Boss 射擊

    帶有 weapon_type 屬性，world.py 以 hasattr(behavior, 'weapon_type') 偵測後
    生成對應投射物（ARROW / MAGIC_BOLT），與 OrbitRangedBehavior 行為一致。
    """

    behavior_type = "blood_pack_disruptor_ranged"

    def __init__(
        self,
        guard_radius: float = 1.5,
        attack_while_guarding: bool = True,
        # OrbitRangedBehavior 參數（名稱與原類別完全一致）
        target_radius: float = 6.0,
        radius_tolerance: float = 1.5,
        danger_radius: float = 4.0,
        clockwise: bool = True,
        weapon_type: str = OrbitRangedBehavior.WEAPON_BOW,
        **kwargs
    ):
        # weapon_type 決定攻擊範圍和傷害，與 OrbitRangedBehavior 一致
        if weapon_type == OrbitRangedBehavior.WEAPON_BOW:
            kwargs.setdefault("attack_range", 8.0)
            kwargs.setdefault("attack_damage", 15.0)
            kwargs.setdefault("attack_cooldown_ticks", 30)
        else:  # staff
            kwargs.setdefault("attack_range", 10.0)
            kwargs.setdefault("attack_damage", 20.0)
            kwargs.setdefault("attack_cooldown_ticks", 40)

        super().__init__(guard_radius=guard_radius, attack_while_guarding=attack_while_guarding, **kwargs)

        # world.py 以 hasattr(behavior, 'weapon_type') 判斷是否生成投射物
        self.weapon_type = weapon_type

        self._orbit = OrbitRangedBehavior(
            target_radius=target_radius,
            radius_tolerance=radius_tolerance,
            danger_radius=danger_radius,
            clockwise=clockwise,
            weapon_type=weapon_type,
            attack_range=self.attack_range,
            attack_damage=self.attack_damage,
            attack_cooldown_ticks=self.attack_cooldown_ticks,
        )

    def _fighting_action(self, monster_pos, monster_angle, agent_pos) -> MonsterAction:
        """戰鬥模式：委派給 OrbitRangedBehavior 的移動邏輯"""
        angle_to_agent = self._get_angle_to_target(monster_pos, monster_angle, agent_pos)
        distance = self._get_distance_to_target(monster_pos, agent_pos)
        turning = self._decide_turning(angle_to_agent)
        movement = self._orbit._decide_movement(angle_to_agent, distance)

        attack = AttackType.NONE
        if distance <= self.attack_range and self._is_facing_target(angle_to_agent, tolerance=0.3):
            attack = AttackType.ATTACK

        return MonsterAction(movement=movement, turning=turning, attack=attack)

    def _guarding_action(self, monster_pos, monster_angle, blood_pack_pos, agent_pos) -> MonsterAction:
        """守衛血包（遠程版）：守在原地，持續向 Boss 射擊"""
        if blood_pack_pos is None:
            return MonsterAction()

        dist_to_pack = self._get_distance_to_target(monster_pos, blood_pack_pos)
        angle_to_agent = self._get_angle_to_target(monster_pos, monster_angle, agent_pos)
        dist_to_agent = self._get_distance_to_target(monster_pos, agent_pos)

        turning = self._decide_turning(angle_to_agent)
        movement = MovementType.IDLE

        if dist_to_pack > self.guard_radius:
            # 跑太遠了，優先回到血包旁
            angle_to_pack = self._get_angle_to_target(monster_pos, monster_angle, blood_pack_pos)
            turning = self._decide_turning(angle_to_pack)
            if self._is_facing_target(angle_to_pack, tolerance=0.5):
                movement = MovementType.SPRINT_FORWARD

        attack = AttackType.NONE
        if (self.attack_while_guarding
                and dist_to_agent <= self.attack_range
                and self._is_facing_target(angle_to_agent, tolerance=0.3)):
            attack = AttackType.ATTACK

        return MonsterAction(movement=movement, turning=turning, attack=attack)

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["target_radius"] = self._orbit.target_radius
        data["radius_tolerance"] = self._orbit.radius_tolerance
        data["danger_radius"] = self._orbit.danger_radius
        data["clockwise"] = self._orbit.clockwise
        data["weapon_type"] = self.weapon_type
        return data
