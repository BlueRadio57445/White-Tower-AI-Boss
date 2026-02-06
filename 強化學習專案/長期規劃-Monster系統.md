# Monster 系統設計文檔

## 概述

Monster 代表真人 Minecraft 玩家。為了讓訓練出的 Boss AI 能有效對抗真人，Monster 的行為必須模擬真實玩家的操作方式。

### 設計原則

**笛卡兒積動作空間**：真人玩家可以同時操作 WASD（移動）+ 滑鼠（轉頭）+ 攻擊鍵，所以 Monster 的動作是這三者的組合：

```
動作 = 移動 × 轉向 × 攻擊
     = {FORWARD, BACKWARD, LEFT, RIGHT, SPRINT_FORWARD, IDLE}
     × {TURN_LEFT, TURN_RIGHT, TURN_180, NONE}
     × {ATTACK, NONE}
```

**與 Agent 的差異**：
- Agent（Boss）每 tick 只能做一件事（移動或轉向或放技能）
- Monster（玩家）每 tick 可以同時移動 + 轉向 + 攻擊

**Minecraft 限制**：
- 只有正前方可以奔跑（SPRINT_FORWARD），側移和後退無法加速
- 最有效的逃跑方式是 180 度掉頭 + 奔跑

---

## 程式碼架構

### 檔案結構

```
game/behaviors/
├── __init__.py              # 導出所有類別
├── base.py                  # 基礎類別和枚舉
│   ├── MovementType         # 移動枚舉
│   ├── TurningType          # 轉向枚舉
│   ├── AttackType           # 攻擊枚舉
│   ├── MonsterAction        # 動作資料類別
│   ├── MonsterBehavior      # 行為基類
│   ├── BehaviorRegistry     # 行為註冊表
│   └── MonsterActionExecutor # 動作執行器
├── stationary.py            # 站樁行為
├── berserker.py             # 狂戰士行為
├── hit_and_run.py           # 偷傷害行為
└── orbit.py                 # 繞圈行為（近戰/遠程）
```

---

## 動作原語

### MovementType（WASD）

```python
class MovementType(Enum):
    IDLE = auto()           # 不移動
    FORWARD = auto()        # W - 向前走
    BACKWARD = auto()       # S - 向後走（較慢）
    LEFT = auto()           # A - 左側移
    RIGHT = auto()          # D - 右側移
    SPRINT_FORWARD = auto() # W + Sprint - 向前跑（最快，只有前進可用）
```

### TurningType（滑鼠移動）

```python
class TurningType(Enum):
    NONE = auto()           # 不轉頭
    LEFT = auto()           # 向左轉
    RIGHT = auto()          # 向右轉
    TURN_180 = auto()       # 180度掉頭（緊急逃跑用）
```

### AttackType（滑鼠點擊）

```python
class AttackType(Enum):
    NONE = auto()           # 不攻擊
    ATTACK = auto()         # 攻擊（近戰或發射投射物，取決於武器）
```

### MonsterAction（組合動作）

```python
@dataclass
class MonsterAction:
    """一個 tick 內的完整動作"""
    movement: MovementType = MovementType.IDLE
    turning: TurningType = TurningType.NONE
    attack: AttackType = AttackType.NONE
```

---

## 速度常數

```python
class MovementSpeed:
    WALK_FORWARD = 0.25     # 向前走
    WALK_BACKWARD = 0.15    # 向後走（較慢）
    STRAFE = 0.20           # 側移
    SPRINT = 0.45           # 奔跑（最快）

class TurningSpeed:
    NORMAL = 0.3            # 弧度/tick
    TURN_180 = π            # 瞬間 180 度
```

---

## 行為基類

```python
class MonsterBehavior(ABC):
    """Monster AI 行為基類"""

    behavior_type: str = "base"

    def __init__(
        self,
        walk_speed: float = 0.25,
        sprint_speed: float = 0.45,
        turn_speed: float = 0.3,
        attack_cooldown_ticks: int = 20,  # 攻擊 CD（約 1 秒）
        attack_range: float = 3.0,
        attack_damage: float = 10.0
    ):
        ...

    @abstractmethod
    def decide_action(self, entity, world) -> MonsterAction:
        """決定這個 tick 的動作"""
        pass

    def can_attack(self) -> bool:
        """攻擊是否已冷卻"""
        return self._attack_cooldown_remaining <= 0

    def use_attack(self):
        """消耗攻擊 CD"""
        self._attack_cooldown_remaining = self.attack_cooldown_ticks

    # 輔助方法
    def _get_angle_to_target(self, pos, angle, target) -> float: ...
    def _get_distance_to_target(self, pos, target) -> float: ...
    def _decide_turning(self, angle_diff) -> TurningType: ...
    def _is_facing_target(self, angle_diff, tolerance) -> bool: ...
```

---

## 已實作的行為

### 1. StationaryBehavior（站樁）

```python
behavior_type = "stationary"
```

**策略**：
- 不移動
- 持續面向 Agent
- 在攻擊範圍內時攻擊

**適用場景**：測試、弱小敵人

---

### 2. BerserkerBehavior（狂戰士）

```python
behavior_type = "berserker"
```

**策略**：
- 永遠朝 Agent 衝鋒
- 距離遠時奔跑接近
- 近距離時走路 + 持續攻擊
- 永不後退

**參數**：
- `sprint_threshold`: 超過此距離就奔跑（預設 5.0）

**適用場景**：近戰劍士、激進型玩家

---

### 3. HitAndRunBehavior（偷傷害）

```python
behavior_type = "hit_and_run"
```

**策略**：
- 接近 Agent
- 攻擊一次
- 立即 180 度掉頭 + 奔跑逃離
- 逃離一段距離後再次接近

**狀態機**：
```
APPROACHING → ATTACKING → FLEEING → APPROACHING
```

**參數**：
- `flee_distance`: 逃到這個距離才停（預設 6.0）
- `safe_distance`: 超過這個距離開始接近（預設 7.0）
- `flee_duration`: 逃跑最少持續的 ticks（預設 30）

**適用場景**：技巧型近戰玩家、謹慎型玩家

---

### 4. OrbitMeleeBehavior（近戰繞圈）

```python
behavior_type = "orbit_melee"
```

**策略**：
- 維持在 Agent 周圍的短距離
- 持續橫移（strafe）做圓周運動
- 同時面向 Agent 並攻擊

**參數**：
- `target_radius`: 目標繞行半徑（預設 2.5）
- `radius_tolerance`: 半徑容許誤差（預設 1.0）
- `clockwise`: 順時針或逆時針（預設 True）

**適用場景**：PvP 高手、近戰玩家

---

### 5. OrbitRangedBehavior（遠程繞圈 / Kiting）

```python
behavior_type = "orbit_ranged"
```

**策略**：
- 維持在 Agent 周圍的長距離
- 持續橫移做圓周運動
- 面向 Agent 時射擊
- Agent 靠近時後退

**參數**：
- `target_radius`: 目標繞行半徑（預設 6.0，較遠）
- `danger_radius`: 低於此距離要緊急後退（預設 4.0）
- `weapon_type`: "bow" 或 "staff"

**武器差異**：
| 武器 | 射程 | 傷害 | CD |
|-----|------|------|-----|
| bow | 8.0 | 15.0 | 30 ticks |
| staff | 10.0 | 20.0 | 40 ticks |

**適用場景**：弓箭手、法師

---

## 動作執行器

```python
class MonsterActionExecutor:
    """執行 MonsterAction 並更新實體狀態"""

    def execute(self, entity, action, behavior) -> Dict[str, Any]:
        """
        執行動作

        Returns:
            {
                "hit_wall": bool,
                "attacked": bool,
                "attack_damage": float
            }
        """
        # 1. 執行轉向（瞬間）
        self._execute_turning(entity, action.turning, behavior.turn_speed)

        # 2. 執行移動（檢查牆壁碰撞）
        hit_wall = self._execute_movement(entity, action.movement, behavior)

        # 3. 執行攻擊（如果 CD 好了）
        if action.attack == AttackType.ATTACK and behavior.can_attack():
            behavior.use_attack()
            return {"attacked": True, "attack_damage": behavior.attack_damage}

        # 4. 更新 CD
        behavior.update_cooldowns()

        return {"hit_wall": hit_wall, "attacked": False}
```

---

## 投射物整合

當 `OrbitRangedBehavior` 攻擊時：

```python
# 在 GameWorld 中
def _process_monster_attack(self, monster, behavior, result):
    if not result["attacked"]:
        return

    if behavior.weapon_type in ["bow", "staff"]:
        # 發射投射物（使用 Monster 當前朝向）
        self.projectile_manager.spawn_projectile(
            position=monster.position.copy(),
            direction=monster.angle,  # 就是 Monster 面向的方向
            projectile_type=behavior.weapon_type,
            owner_id=monster.id
        )
    else:
        # 近戰攻擊，直接造成傷害（如果 Agent 在範圍內）
        ...
```

---

## 使用範例

### 創建不同類型的 Monster

```python
from game.behaviors import (
    BerserkerBehavior,
    HitAndRunBehavior,
    OrbitMeleeBehavior,
    OrbitRangedBehavior
)

# 狂戰士劍士
sword_monster = EntityFactory.create_monster(
    x=8.0, y=8.0,
    behavior=BerserkerBehavior(
        attack_range=3.0,
        attack_damage=15.0,
        attack_cooldown_ticks=20
    )
)

# 技巧型近戰
skilled_melee = EntityFactory.create_monster(
    x=2.0, y=8.0,
    behavior=HitAndRunBehavior(
        flee_distance=5.0,
        attack_damage=12.0
    )
)

# 近戰繞圈
orbiter = EntityFactory.create_monster(
    x=5.0, y=2.0,
    behavior=OrbitMeleeBehavior(
        target_radius=2.0,
        clockwise=False  # 逆時針
    )
)

# 弓箭手
archer = EntityFactory.create_monster(
    x=8.0, y=2.0,
    behavior=OrbitRangedBehavior(
        target_radius=6.0,
        weapon_type="bow"
    )
)

# 法師
mage = EntityFactory.create_monster(
    x=2.0, y=2.0,
    behavior=OrbitRangedBehavior(
        target_radius=7.0,
        weapon_type="staff"
    )
)
```

### 在 GameWorld 中更新 Monster

```python
class GameWorld:
    def __init__(self):
        self.action_executor = MonsterActionExecutor(room_size=self.room_size)

    def _update_monsters(self):
        for monster in self.monsters:
            if not monster.is_alive:
                continue

            behavior = monster.behavior

            # 1. 決定動作
            action = behavior.decide_action(monster, self)

            # 2. 執行動作
            result = self.action_executor.execute(monster, action, behavior)

            # 3. 處理攻擊結果
            if result["attacked"]:
                self._process_monster_attack(monster, behavior, result)
```

---

## 訓練多樣性

為了讓 Agent 學會應對各種玩家風格，每個 episode 應該隨機生成不同的 Monster 組合：

```python
def generate_random_monsters(num_monsters: int = 4) -> List[MonsterConfig]:
    """隨機生成 Monster 配置"""
    behavior_pool = [
        ("berserker", BerserkerBehavior),
        ("hit_and_run", HitAndRunBehavior),
        ("orbit_melee", OrbitMeleeBehavior),
        ("orbit_ranged_bow", lambda: OrbitRangedBehavior(weapon_type="bow")),
        ("orbit_ranged_staff", lambda: OrbitRangedBehavior(weapon_type="staff")),
    ]

    configs = []
    for i in range(num_monsters):
        behavior_name, behavior_factory = random.choice(behavior_pool)

        # 隨機位置（避開中心）
        angle = random.uniform(0, 2 * np.pi)
        radius = random.uniform(3.0, 4.5)
        x = 5.0 + radius * np.cos(angle)
        y = 5.0 + radius * np.sin(angle)

        configs.append({
            "position": [x, y],
            "behavior": behavior_factory(),
            "health": random.uniform(80, 120)
        })

    return configs
```

---

## 序列化

行為支援序列化，用於地圖存檔和讀取：

```python
# 序列化
behavior_data = monster.behavior.to_dict()
# {"type": "orbit_melee", "target_radius": 2.5, "clockwise": true, ...}

# 反序列化
behavior = BehaviorRegistry.from_dict(behavior_data)
```

---

## 測試方法

```python
def test_berserker_chases_agent():
    world = GameWorld()
    monster = create_monster_with_behavior(
        BerserkerBehavior(),
        position=(8, 8)
    )
    world.player.position = np.array([2.0, 2.0])

    initial_dist = distance(monster, world.player)

    for _ in range(50):
        world.tick()

    final_dist = distance(monster, world.player)
    assert final_dist < initial_dist, "Berserker should chase Agent"


def test_hit_and_run_flees_after_attack():
    world = GameWorld()
    behavior = HitAndRunBehavior()
    monster = create_monster_with_behavior(behavior, position=(3, 3))
    world.player.position = np.array([3.0, 3.0])

    # 模擬直到攻擊發生
    for _ in range(100):
        world.tick()
        if behavior._internal_state.get("state") == "fleeing":
            break

    assert behavior._internal_state["state"] == "fleeing"


def test_orbit_maintains_distance():
    world = GameWorld()
    behavior = OrbitMeleeBehavior(target_radius=2.5)
    monster = create_monster_with_behavior(behavior, position=(5, 5))
    world.player.position = np.array([5.0, 5.0])

    distances = []
    for _ in range(100):
        world.tick()
        distances.append(distance(monster, world.player))

    # 距離應該穩定在 target_radius 附近
    avg_distance = np.mean(distances[-50:])
    assert abs(avg_distance - 2.5) < 1.0
```
