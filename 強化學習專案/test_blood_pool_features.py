"""
Test script for blood pool enhancement features:
1. Blood pool invulnerability (melee and ranged)
2. Blood pool visual effects (manual verification)
3. Summon blood pack reward
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '純白之塔_RL'))

import numpy as np
from game.world import GameWorld, Room
from game.player import PlayerConfig
from ai.reward import RewardCalculator
from game.behaviors import StationaryBehavior


def test_blood_pool_invulnerability_melee():
    """Test that blood pool blocks melee damage."""
    print("=" * 60)
    print("Test 1: Blood Pool Invulnerability (Melee)")
    print("=" * 60)

    world = GameWorld(Room(size=10.0))
    world.reset()

    # Position player and monster close together
    world.player.position.x = 5.0
    world.player.position.y = 5.0

    # Place monster very close
    world.monsters[0].position.x = 5.5
    world.monsters[0].position.y = 5.0
    world.monsters[0].position.angle = np.pi  # Facing left toward player
    world.monsters[0].movement_behavior = StationaryBehavior(attack_damage=20.0)

    initial_health = world.player.current_health
    print(f"Initial player health: {initial_health}")

    # Tick without blood pool - should take damage
    world.tick()
    health_after_attack = world.player.current_health
    print(f"Health after melee attack (no pool): {health_after_attack}")
    damage_taken = initial_health - health_after_attack
    print(f"Damage taken: {damage_taken}")

    # Reset health
    world.player.health.heal(1000)

    # Enter blood pool
    aim_values = [0.0] * 6
    world.execute_action(10, aim_values)  # Cast blood pool
    world.tick()  # Process blood pool entry

    print(f"In blood pool: {world.player.skills.in_blood_pool}")
    health_in_pool = world.player.current_health

    # Monster should attack but deal no damage
    world.tick()
    health_after_attack_in_pool = world.player.current_health
    print(f"Health after melee attack (in pool): {health_after_attack_in_pool}")

    assert health_in_pool == health_after_attack_in_pool, \
        f"Blood pool should block melee damage! {health_in_pool} -> {health_after_attack_in_pool}"

    print("[OK] Blood pool blocks melee damage\n")


def test_blood_pool_invulnerability_ranged():
    """Test that blood pool blocks projectile damage."""
    print("=" * 60)
    print("Test 2: Blood Pool Invulnerability (Ranged)")
    print("=" * 60)

    world = GameWorld(Room(size=10.0))
    world.reset()

    # Enter blood pool
    aim_values = [0.0] * 6
    world.execute_action(10, aim_values)  # Cast blood pool
    world.tick()  # Process blood pool entry

    print(f"In blood pool: {world.player.skills.in_blood_pool}")
    initial_health = world.player.current_health
    print(f"Initial player health: {initial_health}")

    # Manually spawn a projectile heading toward player
    from game.projectile import ProjectileType
    player_pos = world.get_player_position()

    # Spawn arrow heading toward player
    world.projectile_manager.spawn_projectile(
        position=np.array([player_pos[0] - 2.0, player_pos[1]]),
        direction=np.array([1.0, 0.0]),  # Moving right toward player
        owner_id="test_archer",
        projectile_type=ProjectileType.ARROW,
        damage_override=15.0
    )

    # Tick several times to let projectile hit
    for _ in range(5):
        world.tick()

    final_health = world.player.current_health
    print(f"Health after projectile pass (in pool): {final_health}")

    assert initial_health == final_health, \
        f"Blood pool should block projectile damage! {initial_health} -> {final_health}"

    print("[OK] Blood pool blocks projectile damage\n")


def test_summon_blood_pack_reward():
    """Test that summoning blood packs gives correct reward."""
    print("=" * 60)
    print("Test 3: Summon Blood Pack Reward")
    print("=" * 60)

    world = GameWorld(Room(size=10.0))
    reward_calc = RewardCalculator(world.event_bus)
    world.reset()

    # Remove existing blood packs
    world.items = []

    # Summon blood packs: action 11
    aim_values = [0.0] * 6
    world.execute_action(11, aim_values)

    # Wait for wind-up (10 ticks)
    for _ in range(12):
        world.tick()

    # Check reward
    reward = reward_calc.get_reward()
    last_event = reward_calc.get_last_event()

    print(f"Last event: {last_event}")
    print(f"Reward received: {reward}")

    # Should get 3.0 * 3 = 9.0 for summoning 3 packs
    # Plus some tick penalties (-0.01 * 12 = -0.12)
    expected_reward = 3.0 * 3
    tick_penalty = -0.01 * 12

    print(f"Expected reward (3 packs * 3.0): {expected_reward}")
    print(f"Tick penalty ({12} ticks * -0.01): {tick_penalty}")
    print(f"Total expected: {expected_reward + tick_penalty}")

    # Allow for small floating point differences
    assert abs(reward - (expected_reward + tick_penalty)) < 0.01, \
        f"Wrong reward! Expected ~{expected_reward + tick_penalty}, got {reward}"

    print("[OK] Summon blood pack gives correct reward\n")


def test_summon_blood_pack_no_reward_at_limit():
    """Test that summoning at limit gives no reward."""
    print("=" * 60)
    print("Test 4: Summon Blood Pack No Reward at Limit")
    print("=" * 60)

    world = GameWorld(Room(size=10.0))
    reward_calc = RewardCalculator(world.event_bus)
    world.reset()

    # Remove existing blood packs
    world.items = []

    # First summon (should succeed)
    aim_values = [0.0] * 6
    world.execute_action(11, aim_values)
    for _ in range(12):
        world.tick()

    # Clear accumulated reward
    reward_calc.get_reward()

    # Reset cooldown
    world.player.skills.skill_cooldowns["summon_pack"] = 0

    # Second summon (should fail - already at limit)
    world.execute_action(11, aim_values)
    for _ in range(12):
        world.tick()

    # Check reward
    reward = reward_calc.get_reward()
    last_event = reward_calc.get_last_event()

    print(f"Last event: {last_event}")
    print(f"Reward received: {reward}")

    # Should only get tick penalties, no summon reward
    tick_penalty = -0.01 * 12
    print(f"Expected reward (tick penalty only): {tick_penalty}")

    # Allow for small floating point differences
    assert abs(reward - tick_penalty) < 0.01, \
        f"Should get no summon reward at limit! Expected {tick_penalty}, got {reward}"

    print("[OK] No reward when summoning at limit\n")


def test_blood_pool_visual_effects():
    """Test that blood pool visual effects don't crash (manual visual verification needed)."""
    print("=" * 60)
    print("Test 5: Blood Pool Visual Effects (No Crash)")
    print("=" * 60)

    try:
        from rendering.pygame_renderer import PygameRenderer

        world = GameWorld(Room(size=10.0))
        world.reset()
        renderer = PygameRenderer(world_size=10.0)

        # Enter blood pool
        aim_values = [0.0] * 6
        world.execute_action(10, aim_values)
        world.tick()

        # Try to render (should not crash)
        renderer.render(world, epoch=0, reward=0.0, sigma=0.1, action=10,
                       continuous=0.0, event="TEST", action_names=None)

        renderer.close()
        print("[OK] Blood pool visual effects render without crash")
        print("Note: Visual appearance should be verified manually in dev mode\n")

    except Exception as e:
        print(f"[FAIL] Visual rendering crashed: {e}\n")
        raise


if __name__ == "__main__":
    print("\n>>> Blood Pool Enhancement Features Test <<<\n")

    try:
        test_blood_pool_invulnerability_melee()
        test_blood_pool_invulnerability_ranged()
        test_summon_blood_pack_reward()
        test_summon_blood_pack_no_reward_at_limit()
        test_blood_pool_visual_effects()

        print("=" * 60)
        print("[SUCCESS] All blood pool features working correctly!")
        print("=" * 60)
        print("\nFeatures implemented:")
        print("1. Blood pool invulnerability (melee + ranged)")
        print("2. Blood pool visual effects (pulsing red pool + emerge warning)")
        print("3. Summon blood pack reward (+3.0 per pack)")
        print("\nTo verify visual effects:")
        print("  python 純白之塔_RL/main.py --dev")
        print("  Press 7 to enter blood pool")
        print("  Observe pulsing red effect and emerge warning ring")

    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
