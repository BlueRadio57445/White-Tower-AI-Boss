# Monster 系統設計文檔

## ✅ 實作狀態：已完成

Monster 代表真人 Minecraft 玩家。為了讓訓練出的 Boss AI 能有效對抗真人，Monster 的行為必須模擬真實玩家的操作方式。

### 實際實作摘要

| 檔案 | 說明 |
|-----|------|
| `game/behaviors/__init__.py` | 導出所有類別 |
| `game/behaviors/base.py` | MovementType, TurningType, AttackType, MonsterAction, MonsterBehavior, BehaviorRegistry, MonsterActionExecutor |
| `game/behaviors/stationary.py` | StationaryBehavior |
| `game/behaviors/berserker.py` | BerserkerBehavior |
| `game/behaviors/hit_and_run.py` | HitAndRunBehavior |
| `game/behaviors/orbit.py` | OrbitMeleeBehavior, OrbitRangedBehavior |

### 已實作行為

| 行為 | 類型 | 策略 |
|-----|------|------|
| `stationary` | 站樁 | 不動，面向攻擊 |
| `berserker` | 近戰 | 正面衝鋒，永不後退 |
| `hit_and_run` | 近戰 | 打一下就 180° 掉頭逃跑 |
| `orbit_melee` | 近戰 | 橫移繞圈，近距離攻擊 |
| `orbit_ranged` | 遠程 | 保持距離，發射投射物（bow/staff）|

---

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

## 已實作的行為

### 1. StationaryBehavior（站樁）

**策略**：
- 不移動
- 持續面向 Agent
- 在攻擊範圍內時攻擊

**適用場景**：測試、弱小敵人

---

### 2. BerserkerBehavior（狂戰士）

**策略**：
- 永遠朝 Agent 衝鋒
- 距離遠時奔跑接近
- 近距離時走路 + 持續攻擊
- 永不後退

**適用場景**：近戰劍士、激進型玩家

---

### 3. HitAndRunBehavior（偷傷害）


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

## 待實作的行為

### 6. OpportunistBehavior（機會主義者）

**策略**：
- 維持安全距離繞圈等待
- 偵測到 Boss 開始蓄力（casting progress 上升），立刻衝進去
- 抵達攻擊範圍後攻擊
- 攻擊後立即撤退，回到繞圈等待

**狀態機**：
```
CIRCLING → RUSHING（Boss 蓄力中）→ ATTACKING（進入攻擊範圍）→ RETREATING → CIRCLING
```

**適用場景**：會讀招式的技巧型玩家

---

### 7. AdaptiveBehavior（適應型）

**策略**：
- 血量高時：激進，類似 berserker 直接衝
- 血量中時：中等，類似 orbit 與其周旋
- 血量極低時：保守，類似 hit_and_run 打完就跑

**狀態機**：
```
PHASE_AGGRESSIVE（血量 > 60%）
    ↕
PHASE_CAUTIOUS（血量 30%-60%）
    ↕
PHASE_DESPERATE（血量 < 30%）
```

**關鍵觀察值**：自身血量（需在 MonsterBehavior 中存取 entity.health）

**適用場景**：所有真實玩家都會考量自身血量

---

### 8. BackstabBehavior（繞背型）

**策略**：
- 永遠試圖移動到 Boss 的後方
- 只有位於 Boss 後方扇形範圍內才攻擊
- 不需要複雜狀態機，像 berserker/orbit 一樣純反應式

**核心邏輯**：
- 計算 Boss 背後的目標位置
- 判斷自己是否已在後方（Boss 朝向角度 ± 容差）
- 在後方時攻擊，不在後方時繞過去

**適用場景**：習慣偷背的玩家

---

### 9. BloodPackDisruptorBehavior（血包干擾型）

**策略**：
- 附近沒有血包時：正常移動攻擊 Boss
- 血包生成後：移動到血包位置守住
- 守住血包的同時，在不離開血包太遠的情況下攻擊 Boss

**狀態機**：
```
FIGHTING（無血包）→ RACING（血包生成，衝過去）→ GUARDING（守在血包旁攻擊）→ FIGHTING（血包消失）
```

**近戰版**：守住血包，等 Boss 接近時攻擊（或者以血包為圓心繞行，想辦法在不離開血包的前提下攻擊Boss）
**遠程版**：守住血包，在守衛位置向 Boss 射擊，不需要靠近

**關鍵觀察值**：血包位置 、血包距離 

**適用場景**：策略型玩家（防止Boss回血的輔助角色）

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

