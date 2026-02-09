"""
測試最後兩個技能：血池和召喚血包
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '純白之塔_RL'))

import numpy as np
from game.world import GameWorld, Room
from game.player import PlayerConfig

def test_blood_pool():
    """測試血池技能"""
    print("=" * 60)
    print("測試 1: 血池技能")
    print("=" * 60)

    world = GameWorld(Room(size=10.0))
    world.reset()

    player_pos = world.get_player_position()
    print(f"玩家位置: {player_pos}")
    print(f"技能狀態 - in_blood_pool: {world.player.skills.in_blood_pool}")

    # Cast blood pool: action 10
    aim_values = [0.0] * 6
    event = world.execute_action(10, aim_values)
    print(f"施放事件: {event}")

    # Tick once to process the skill
    world.tick()

    print(f"施放後 - in_blood_pool: {world.player.skills.in_blood_pool}")
    print(f"血池剩餘時間: {world.player.skills.blood_pool_remaining}")

    # Simulate being in pool
    for tick in range(16):
        world.tick()
        if tick == 0:
            print(f"第 1 tick - in_blood_pool: {world.player.skills.in_blood_pool}")
        elif tick == 14:
            print(f"第 15 tick - in_blood_pool: {world.player.skills.in_blood_pool}")
        elif tick == 15:
            print(f"第 16 tick (emerge) - in_blood_pool: {world.player.skills.in_blood_pool}")

    print("[OK] 通過\n")

def test_summon_blood_pack():
    """測試召喚血包技能"""
    print("=" * 60)
    print("測試 2: 召喚血包技能")
    print("=" * 60)

    world = GameWorld(Room(size=10.0))
    world.reset()

    # Remove initial blood pack
    world.items = []

    initial_count = world.get_blood_pack_count()
    print(f"初始血包數量: {initial_count}")

    # Cast summon pack: action 11
    aim_values = [0.0] * 6
    event = world.execute_action(11, aim_values)
    print(f"施放事件: {event}")

    # Wait for wind-up (10 ticks)
    for _ in range(12):
        world.tick()

    final_count = world.get_blood_pack_count()
    print(f"召喚後血包數量: {final_count}")
    print(f"血包位置:")
    for item in world.items:
        if item.is_alive and item.has_tag("blood_pack"):
            print(f"  - {item.position.as_array()}")

    assert final_count == 3, f"應該有 3 個血包，但只有 {final_count} 個"
    print("[OK] 通過\n")

def test_summon_max_limit():
    """測試召喚血包的數量限制"""
    print("=" * 60)
    print("測試 3: 召喚血包數量限制")
    print("=" * 60)

    world = GameWorld(Room(size=10.0))
    world.reset()

    # Remove initial blood pack
    world.items = []

    # First summon - should create 3 packs
    aim_values = [0.0] * 6
    world.execute_action(11, aim_values)
    for _ in range(12):
        world.tick()

    count_after_first = world.get_blood_pack_count()
    print(f"第一次召喚後: {count_after_first} 個血包")

    # Second summon - should fail (max reached)
    # Reset cooldown for testing
    world.player.skills.skill_cooldowns["summon_pack"] = 0

    world.execute_action(11, aim_values)
    for _ in range(12):
        world.tick()

    count_after_second = world.get_blood_pack_count()
    print(f"第二次召喚後: {count_after_second} 個血包")

    assert count_after_second == 3, f"應該仍然只有 3 個血包，但有 {count_after_second} 個"
    print("[OK] 通過 - 成功阻止超過上限\n")

def test_player_config():
    """測試玩家配置包含所有技能"""
    print("=" * 60)
    print("測試 4: 玩家配置檢查")
    print("=" * 60)

    config = PlayerConfig()
    print(f"技能總數: {len(config.skills)}")
    print("技能列表:")
    for skill_id, skill_config in config.skills.items():
        print(f"  - {skill_id}: {skill_config.name} (CD: {skill_config.cooldown_ticks}, "
              f"Wind-up: {skill_config.wind_up_ticks}, Shape: {skill_config.shape_type.value})")

    assert len(config.skills) == 8, f"應該有 8 個技能，但只有 {len(config.skills)} 個"
    assert "blood_pool" in config.skills, "缺少血池技能"
    assert "summon_pack" in config.skills, "缺少召喚血包技能"
    print("[OK] 通過\n")

if __name__ == "__main__":
    print("\n>>> 最後技能實作驗證測試 <<<\n")

    try:
        test_player_config()
        test_blood_pool()
        test_summon_blood_pack()
        test_summon_max_limit()

        print("=" * 60)
        print("[成功] 所有測試完成！")
        print("=" * 60)
        print("\n[提示] 使用開發者模式測試新技能：")
        print("  python 純白之塔_RL/main.py --dev")
        print("  按鍵 7: 血池")
        print("  按鍵 8: 召喚血包")

    except Exception as e:
        print(f"\n[錯誤] 測試失敗: {e}")
        import traceback
        traceback.print_exc()
