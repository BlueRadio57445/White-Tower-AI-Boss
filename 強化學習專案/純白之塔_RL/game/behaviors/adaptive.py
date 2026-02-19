"""
Adaptive behavior - changes strategy based on own health percentage.
適應型：根據自身血量切換策略，委派給對應的子行為
"""

from typing import Dict, Any, Optional

from game.behaviors.base import (
    MonsterBehavior,
    MonsterAction,
    BehaviorRegistry,
)
from game.behaviors.berserker import BerserkerBehavior
from game.behaviors.orbit import OrbitMeleeBehavior
from game.behaviors.hit_and_run import HitAndRunBehavior


@BehaviorRegistry.register
class AdaptiveBehavior(MonsterBehavior):
    """
    適應型行為

    策略：
    - 血量高時（> aggressive_threshold）：激進，委派給 BerserkerBehavior
    - 血量中時（cautious_threshold ~ aggressive_threshold）：謹慎，委派給 OrbitMeleeBehavior
    - 血量極低時（< cautious_threshold）：絕境，委派給 HitAndRunBehavior

    相位：
    PHASE_AGGRESSIVE（血量 > 60%）
        ↕
    PHASE_CAUTIOUS（血量 30%-60%）
        ↕
    PHASE_DESPERATE（血量 < 30%）

    注意：can_attack / use_attack / update_cooldowns 會轉發給當前 _active 子行為，
    確保 HitAndRunBehavior 的狀態機（依賴 can_attack 偵測攻擊發生）能正確運作。
    """

    behavior_type = "adaptive"

    PHASE_AGGRESSIVE = "aggressive"
    PHASE_CAUTIOUS = "cautious"
    PHASE_DESPERATE = "desperate"

    def __init__(
        self,
        aggressive_threshold: float = 0.7,     # 超過此血量比例為激進模式
        cautious_threshold: float = 0.4,        # 低於此血量比例為絕境模式
        # BerserkerBehavior 參數（名稱與原類別完全一致）
        sprint_threshold: float = 5.0,
        # OrbitMeleeBehavior 參數（名稱與原類別完全一致）
        target_radius: float = 2.5,
        radius_tolerance: float = 1.0,
        clockwise: bool = True,
        strafe_while_approaching: bool = True,
        # HitAndRunBehavior 參數（名稱與原類別完全一致）
        flee_distance: float = 6.0,
        safe_distance: float = 7.0,
        flee_duration: int = 30,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.aggressive_threshold = aggressive_threshold
        self.cautious_threshold = cautious_threshold

        # 共用的攻擊參數傳入各子行為
        shared = dict(
            attack_range=self.attack_range,
            attack_damage=self.attack_damage,
            attack_cooldown_ticks=self.attack_cooldown_ticks,
        )

        self._berserker = BerserkerBehavior(sprint_threshold=sprint_threshold, **shared)
        self._orbit = OrbitMeleeBehavior(
            target_radius=target_radius,
            radius_tolerance=radius_tolerance,
            clockwise=clockwise,
            strafe_while_approaching=strafe_while_approaching,
            **shared
        )
        self._hit_and_run = HitAndRunBehavior(
            flee_distance=flee_distance,
            safe_distance=safe_distance,
            flee_duration=flee_duration,
            **shared
        )

        self._active: Optional[MonsterBehavior] = None

    def _get_phase(self, entity) -> str:
        """根據自身血量決定當前相位"""
        if entity.has_health():
            hp_ratio = entity.health.percentage
            if hp_ratio >= self.aggressive_threshold:
                return self.PHASE_AGGRESSIVE
            elif hp_ratio >= self.cautious_threshold:
                return self.PHASE_CAUTIOUS
            else:
                return self.PHASE_DESPERATE
        return self.PHASE_AGGRESSIVE  # 沒有血量資訊時預設激進

    def decide_action(self, entity, world) -> MonsterAction:
        if world.player is None:
            return MonsterAction()

        phase = self._get_phase(entity)

        if phase == self.PHASE_AGGRESSIVE:
            self._active = self._berserker
        elif phase == self.PHASE_CAUTIOUS:
            self._active = self._orbit
        else:  # DESPERATE
            self._active = self._hit_and_run

        return self._active.decide_action(entity, world)

    # =========================================================================
    # 轉發冷卻方法給當前子行為
    # HitAndRunBehavior 的狀態機依賴 can_attack() 偵測攻擊發動（在 MonsterActionExecutor
    # 呼叫 use_attack / update_cooldowns 後，下一 tick 的 can_attack 才會回傳 False）。
    # 若不轉發，子行為的冷卻永遠不會被更新，狀態機無法轉換到 FLEEING。
    # =========================================================================

    def can_attack(self) -> bool:
        return self._active.can_attack() if self._active else super().can_attack()

    def use_attack(self):
        if self._active:
            self._active.use_attack()
        else:
            super().use_attack()

    def update_cooldowns(self):
        if self._active:
            self._active.update_cooldowns()
        else:
            super().update_cooldowns()

    def reset(self):
        super().reset()
        self._berserker.reset()
        self._orbit.reset()
        self._hit_and_run.reset()
        self._active = None

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["aggressive_threshold"] = self.aggressive_threshold
        data["cautious_threshold"] = self.cautious_threshold
        data["sprint_threshold"] = self._berserker.sprint_threshold
        data["target_radius"] = self._orbit.target_radius
        data["radius_tolerance"] = self._orbit.radius_tolerance
        data["clockwise"] = self._orbit.clockwise
        data["strafe_while_approaching"] = self._orbit.strafe_while_approaching
        data["flee_distance"] = self._hit_and_run.flee_distance
        data["safe_distance"] = self._hit_and_run.safe_distance
        data["flee_duration"] = self._hit_and_run.flee_duration
        return data
