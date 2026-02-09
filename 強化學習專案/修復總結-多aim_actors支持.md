# 修復總結 - 多 Aim Actors 支持

## 問題描述

訓練啟動時出現錯誤：
```
TypeError: '>=' not supported between instances of 'list' and 'int'
```

**錯誤位置**: `ai/agent.py` 的 `get_aim_value_for_action()` 方法

**根本原因**:
- 閃現技能需要 2 個 aim actors，在 `SKILL_TO_AIM_ACTOR` 中使用列表 `[2, 3]` 表示
- `get_aim_value_for_action()` 方法假設 `actor_idx` 是整數，無法處理列表

## 修復方案

修改 `ai/agent.py` 中的 `get_aim_value_for_action()` 方法，添加對列表的支持：

```python
def get_aim_value_for_action(self, action_discrete: int, aim_values: List[float]) -> float:
    """
    Get the relevant aim value for a given discrete action.
    For skills with multiple aim actors, returns the first one.
    """
    actor_idx = self.SKILL_TO_AIM_ACTOR.get(action_discrete, -1)

    # Handle skills with multiple aim actors (e.g., dash uses [2, 3])
    if isinstance(actor_idx, list):
        if len(actor_idx) > 0 and actor_idx[0] < len(aim_values):
            return aim_values[actor_idx[0]]
        return 0.0

    # Handle skills with single aim actor
    if actor_idx >= 0 and actor_idx < len(aim_values):
        return aim_values[actor_idx]
    return 0.0
```

### 設計決策

對於有多個 aim actors 的技能（如閃現），`get_aim_value_for_action()` 返回**第一個 aim 值**。

**原因**:
- 這個方法主要用於渲染時顯示瞄準偏移
- 對於多 aim actors 的技能，只顯示主要方向（第一個 actor）即可
- 實際執行時，`world.execute_action()` 會使用完整的 `aim_values` 列表

## 驗證結果

### 測試 1: Agent 動作空間配置
```
離散動作數量: 10 ✓
連續 Actor 數量: 6 ✓
w_actor_discrete shape: (10, 27) ✓
w_aim_actors count: 6 ✓
```

### 測試 2: 技能到 Actor 映射
```
SKILL_TO_AIM_ACTOR 映射:
  Action 4: Actor -1        (外圈刮 - 無瞄準)
  Action 5: Actor 0         (飛彈 - aim_missile)
  Action 6: Actor 1         (鐵錘 - aim_hammer)
  Action 7: Actor [2, 3]    (閃現 - aim_dash_direction, aim_dash_facing) ✓
  Action 8: Actor 4         (靈魂爪 - aim_claw)
  Action 9: Actor 5         (靈魂掌 - aim_palm)
```

### 測試 3: get_aim_value_for_action 方法
```
Action 0 (MOVE_FORWARD): 0.0 ✓
Action 5 (MISSILE): 0.1 ✓ (返回 aim_values[0])
Action 7 (DASH): 0.3 ✓ (返回 aim_values[2]，即第一個 dash actor)
```

### 測試 4: Agent.get_action() 方法
```
Action mask shape: (10,) ✓
Discrete action: 5
Aim values count: 6 ✓
Aim values: [0.157, 0.664, -0.199, -0.334, -0.425, 0.881]
```

### 測試 5: 訓練運行
```
訓練成功完成 5 個 epochs
無錯誤輸出 ✓
權重正確導出 ✓
```

### 測試 6: 權重導出格式
```json
{
  "parameters": {
    "n_discrete_actions": 10,  ✓
    "n_aim_actors": 6           ✓
  },
  "weights": {
    "actor_discrete": [[...]], // 10 x 27 ✓
    "aim_actors": [[...], [...], [...], [...], [...], [...]],  // 6 個 ✓
    "critic": [...]             // 27 ✓
  }
}
```

## 技能完整性檢查

| 技能 | Action | Aim Actors | 狀態 |
|------|--------|-----------|------|
| 外圈刮 | 4 | 無 | ✓ 正常 |
| 飛彈 | 5 | [0] | ✓ 正常 |
| 鐵錘 | 6 | [1] | ✓ 正常 |
| 閃現 | 7 | [2, 3] | ✓ 修復完成 |
| 靈魂爪 | 8 | [4] | ✓ 正常 |
| 靈魂掌 | 9 | [5] | ✓ 正常 |

## 後續使用

### 正常訓練
```bash
python 純白之塔_RL/main.py
```

### 開發者模式測試
```bash
python 純白之塔_RL/main.py --dev
```
- 按鍵 4: 閃現（需要 2 個 aim actors）
- 按鍵 5: 靈魂爪
- 按鍵 6: 靈魂掌

### 驗證測試
```bash
python verify_fix.py
```

## 修改文件

- ✅ `ai/agent.py` - 修改 `get_aim_value_for_action()` 方法

## 總結

**問題**: 多 aim actors 技能導致類型錯誤
**修復**: 添加列表類型支持
**驗證**: 所有測試通過
**狀態**: ✓ 完全修復，可以正常訓練

---

**修復日期**: 2026-02-09
**測試狀態**: ✓ 全部通過
**訓練狀態**: ✓ 可正常運行
