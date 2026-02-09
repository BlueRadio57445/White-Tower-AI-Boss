"""
簡單測試腳本 - 驗證新技能實現

測試內容：
1. 閃現技能的位移
2. 靈魂爪的拉動效果
3. 靈魂掌的推動效果
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '純白之塔_RL'))

import numpy as np
from game.world import GameWorld, Room
from game.player import PlayerConfig
from game.entity import EntityFactory

def test_dash_skill():
    """測試閃現技能"""
    print("=" * 50)
    print("測試 1: 閃現技能")
    print("=" * 50)

    world = GameWorld(Room(size=10.0))
    world.reset()

    initial_pos = world.get_player_position().copy()
    initial_angle = world.get_player_angle()

    print(f"初始位置: {initial_pos}")
    print(f"初始角度: {initial_angle:.2f}")

    # Cast dash: action 7, aim_values[2]=0.0 (forward), aim_values[3]=0.0 (keep facing)
    aim_values = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    event = world.execute_action(7, aim_values)

    # Dash has wind_up=0, so it should complete immediately after first tick
    # But we need to call tick() to process the skill completion
    for _ in range(2):  # Extra tick to ensure completion
        world.tick()

    final_pos = world.get_player_position().copy()
    final_angle = world.get_player_angle()

    print(f"最終位置: {final_pos}")
    print(f"最終角度: {final_angle:.2f}")
    print(f"移動距離: {np.linalg.norm(final_pos - initial_pos):.2f}")
    print(f"事件: {event}")
    print()

def test_soul_claw():
    """測試靈魂爪 (拉動)"""
    print("=" * 50)
    print("測試 2: 靈魂爪")
    print("=" * 50)

    world = GameWorld(Room(size=10.0))
    world.reset()

    # Get initial monster positions
    if world.monsters:
        monster = world.monsters[0]
        initial_monster_pos = monster.position.as_array().copy()
        player_pos = world.get_player_position()

        print(f"玩家位置: {player_pos}")
        print(f"怪物初始位置: {initial_monster_pos}")
        print(f"初始距離: {np.linalg.norm(initial_monster_pos - player_pos):.2f}")

        # Cast soul claw: action 8
        aim_values = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # aim_claw (actor 4) will be set by angle
        event = world.execute_action(8, aim_values)

        # Wait for wind-up (8 ticks)
        for _ in range(10):
            world.tick()

        final_monster_pos = monster.position.as_array().copy()
        print(f"怪物最終位置: {final_monster_pos}")
        print(f"最終距離: {np.linalg.norm(final_monster_pos - player_pos):.2f}")
        print(f"移動向量: {final_monster_pos - initial_monster_pos}")
        print(f"事件: {event}")
    else:
        print("沒有怪物可測試")
    print()

def test_soul_palm():
    """測試靈魂掌 (推動)"""
    print("=" * 50)
    print("測試 3: 靈魂掌")
    print("=" * 50)

    world = GameWorld(Room(size=10.0))
    world.reset()

    # Get initial monster positions
    if world.monsters:
        monster = world.monsters[0]
        initial_monster_pos = monster.position.as_array().copy()
        player_pos = world.get_player_position()

        print(f"玩家位置: {player_pos}")
        print(f"怪物初始位置: {initial_monster_pos}")
        print(f"初始距離: {np.linalg.norm(initial_monster_pos - player_pos):.2f}")

        # Cast soul palm: action 9
        aim_values = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # aim_palm (actor 5) will be set by angle
        event = world.execute_action(9, aim_values)

        # Wait for wind-up (8 ticks)
        for _ in range(10):
            world.tick()

        final_monster_pos = monster.position.as_array().copy()
        print(f"怪物最終位置: {final_monster_pos}")
        print(f"最終距離: {np.linalg.norm(final_monster_pos - player_pos):.2f}")
        print(f"移動向量: {final_monster_pos - initial_monster_pos}")
        print(f"事件: {event}")
    else:
        print("沒有怪物可測試")
    print()

def test_player_config():
    """測試玩家配置包含所有新技能"""
    print("=" * 50)
    print("測試 4: 玩家配置檢查")
    print("=" * 50)

    config = PlayerConfig()
    print(f"技能總數: {len(config.skills)}")
    print("技能列表:")
    for skill_id, skill_config in config.skills.items():
        print(f"  - {skill_id}: {skill_config.name} (CD: {skill_config.cooldown_ticks}, "
              f"Wind-up: {skill_config.wind_up_ticks}, Shape: {skill_config.shape_type.value})")
    print()

if __name__ == "__main__":
    print("\n>>> 新技能實作驗證測試 <<<\n")

    try:
        test_player_config()
        test_dash_skill()
        test_soul_claw()
        test_soul_palm()

        print("=" * 50)
        print("[成功] 所有測試完成！")
        print("=" * 50)
        print("\n[提示] 使用 'python 純白之塔_RL/main.py --dev' 進入開發者模式手動測試")
        print("   按鍵 4: 閃現")
        print("   按鍵 5: 靈魂爪")
        print("   按鍵 6: 靈魂掌")

    except Exception as e:
        print(f"\n[錯誤] 測試失敗: {e}")
        import traceback
        traceback.print_exc()
