"""
驗證修復腳本 - 確認新技能和訓練系統正常工作
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '純白之塔_RL'))

import json
import numpy as np
from ai.agent import HybridPPOAgent
from game.world import GameWorld, Room
from ai.features import FeatureExtractor

def test_agent_action_space():
    """測試 Agent 動作空間配置"""
    print("=" * 60)
    print("測試 1: Agent 動作空間配置")
    print("=" * 60)

    agent = HybridPPOAgent(n_features=27)

    print(f"離散動作數量: {agent.n_discrete_actions}")
    print(f"連續 Actor 數量: {agent.n_aim_actors}")
    print(f"w_actor_discrete shape: {agent.w_actor_discrete.shape}")
    print(f"w_aim_actors count: {len(agent.w_aim_actors)}")

    assert agent.n_discrete_actions == 10, "應該有 10 個離散動作"
    assert agent.n_aim_actors == 6, "應該有 6 個 aim actors"
    print("[通過]\n")

def test_skill_to_aim_actor_mapping():
    """測試技能到 Actor 的映射"""
    print("=" * 60)
    print("測試 2: 技能到 Actor 映射")
    print("=" * 60)

    agent = HybridPPOAgent(n_features=27)

    print("SKILL_TO_AIM_ACTOR 映射:")
    for action, actor in agent.SKILL_TO_AIM_ACTOR.items():
        print(f"  Action {action}: Actor {actor}")

    # 測試閃現的多 aim actors
    assert isinstance(agent.SKILL_TO_AIM_ACTOR[7], list), "閃現應該使用列表"
    assert agent.SKILL_TO_AIM_ACTOR[7] == [2, 3], "閃現應該使用 [2, 3]"
    print("[OK] 通過\n")

def test_get_aim_value_for_action():
    """測試 get_aim_value_for_action 方法"""
    print("=" * 60)
    print("測試 3: get_aim_value_for_action 方法")
    print("=" * 60)

    agent = HybridPPOAgent(n_features=27)
    aim_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    # 測試不需要瞄準的動作
    result = agent.get_aim_value_for_action(0, aim_values)  # MOVE_FORWARD
    print(f"Action 0 (MOVE_FORWARD): {result}")
    assert result == 0.0, "移動動作不應該返回 aim 值"

    # 測試單 aim actor 的技能
    result = agent.get_aim_value_for_action(5, aim_values)  # MISSILE
    print(f"Action 5 (MISSILE): {result}")
    assert result == 0.1, "飛彈應該返回 aim_values[0]"

    # 測試多 aim actors 的技能（閃現）
    result = agent.get_aim_value_for_action(7, aim_values)  # DASH
    print(f"Action 7 (DASH): {result}")
    assert result == 0.3, "閃現應該返回第一個 aim_values[2]"

    print("[OK] 通過\n")

def test_agent_get_action():
    """測試 Agent 的 get_action 方法"""
    print("=" * 60)
    print("測試 4: Agent.get_action() 方法")
    print("=" * 60)

    agent = HybridPPOAgent(n_features=27)
    world = GameWorld(Room(size=10.0))
    world.reset()

    extractor = FeatureExtractor(world_size=10.0)
    obs = extractor.extract(world)

    action_mask = world.get_action_mask()
    print(f"Action mask shape: {action_mask.shape}")
    assert action_mask.shape[0] == 10, "Action mask 應該有 10 個元素"

    a_d, aim_values, prob_d, mus, v, logits = agent.get_action(obs, action_mask)

    print(f"Discrete action: {a_d}")
    print(f"Aim values count: {len(aim_values)}")
    print(f"Aim values: {aim_values}")

    assert 0 <= a_d < 10, "離散動作應該在 0-9 範圍內"
    assert len(aim_values) == 6, "應該有 6 個 aim values"
    print("[OK] 通過\n")

def test_weights_export():
    """測試權重導出格式"""
    print("=" * 60)
    print("測試 5: 權重導出格式")
    print("=" * 60)

    weights_path = os.path.join(os.path.dirname(__file__), '純白之塔_RL', 'weights.json')

    if not os.path.exists(weights_path):
        print("[WARN]  weights.json 不存在，跳過測試")
        return

    with open(weights_path, 'r') as f:
        data = json.load(f)

    print(f"Parameters: {data['parameters']}")

    assert data['parameters']['n_discrete_actions'] == 10, "應該有 10 個離散動作"
    assert data['parameters']['n_aim_actors'] == 6, "應該有 6 個 aim actors"

    actor_discrete = data['weights']['actor_discrete']
    aim_actors = data['weights']['aim_actors']

    print(f"actor_discrete shape: {len(actor_discrete)} x {len(actor_discrete[0])}")
    print(f"aim_actors count: {len(aim_actors)}")

    assert len(actor_discrete) == 10, "actor_discrete 應該有 10 行"
    assert len(aim_actors) == 6, "aim_actors 應該有 6 個"

    print("[OK] 通過\n")

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("開始驗證修復")
    print("=" * 60 + "\n")

    try:
        test_agent_action_space()
        test_skill_to_aim_actor_mapping()
        test_get_aim_value_for_action()
        test_agent_get_action()
        test_weights_export()

        print("=" * 60)
        print("[OK] 所有測試通過！修復成功！")
        print("=" * 60)
        print("\n可以正常進行訓練：")
        print("  python 純白之塔_RL/main.py")
        print("\n可以使用開發者模式測試新技能：")
        print("  python 純白之塔_RL/main.py --dev")

    except AssertionError as e:
        print(f"\n[FAIL] 測試失敗: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"\n[FAIL] 錯誤: {e}")
        import traceback
        traceback.print_exc()
