"""
Developer mode for debugging game logic.

In this mode:
- Keyboard controls the player instead of the AI agent
- The game world is frozen unless there's keyboard input
- Mouse controls the aim offset

Controls:
- W: Move forward
- S: Move backward
- A: Turn left
- D: Turn right
- P: Pass (advance one tick without action)
- 1: Cast skill 1 (basic attack)
- 2: Cast skill 2 (reserved)
- ESC: Quit

Mouse:
- Mouse position determines aim offset during casting
"""

import math
import numpy as np

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

from core.events import EventBus, EventType
from game.world import GameWorld, Room
from rendering.pygame_renderer import PygameRenderer, COLORS


class DevModeRenderer(PygameRenderer):
    """
    Extended renderer for developer mode.
    Adds mouse aim indicator and developer mode UI.
    """

    def __init__(self, width: int = 600, height: int = 700, world_size: float = 10.0):
        super().__init__(width, height, world_size)
        pygame.display.set_caption("純白之塔 RL - Developer Mode")

    def draw_aim_cursor(self, world, mouse_pos: tuple) -> float:
        """
        Draw aim cursor and return the aim offset.

        Args:
            world: GameWorld instance
            mouse_pos: Current mouse position (screen coordinates)

        Returns:
            Aim offset in radians (-0.5 to 0.5)
        """
        if not world.player or not world.player.has_position():
            return 0.0

        player_pos = self.world_to_screen(world.get_player_position())
        player_angle = world.get_player_angle()

        # Calculate angle from player to mouse
        dx = mouse_pos[0] - player_pos[0]
        dy = -(mouse_pos[1] - player_pos[1])  # Y is inverted on screen
        mouse_angle = math.atan2(dy, dx)

        # Calculate offset from player's facing direction
        aim_offset = mouse_angle - player_angle

        # Normalize to [-π, π]
        while aim_offset > math.pi:
            aim_offset -= 2 * math.pi
        while aim_offset < -math.pi:
            aim_offset += 2 * math.pi

        # Clamp to ±0.5 radians
        aim_offset = max(-0.5, min(0.5, aim_offset))

        # Draw mouse cursor line
        line_length = 30
        cursor_end = (
            mouse_pos[0],
            mouse_pos[1]
        )

        # Draw crosshair at mouse position
        crosshair_size = 10
        pygame.draw.line(
            self.screen, (255, 255, 0),
            (mouse_pos[0] - crosshair_size, mouse_pos[1]),
            (mouse_pos[0] + crosshair_size, mouse_pos[1]), 2
        )
        pygame.draw.line(
            self.screen, (255, 255, 0),
            (mouse_pos[0], mouse_pos[1] - crosshair_size),
            (mouse_pos[0], mouse_pos[1] + crosshair_size), 2
        )

        # Draw line from player to mouse
        pygame.draw.line(
            self.screen, (255, 255, 0, 128),
            player_pos, mouse_pos, 1
        )

        return aim_offset

    def draw_dev_info_panel(
        self,
        tick: int,
        action_name: str,
        aim_offset: float,
        event: str,
        player_health: float,
        player_max_health: float,
        is_casting: bool,
        casting_progress: float
    ) -> None:
        """Draw developer mode information panel."""
        # Top info bar
        info_y = 10
        top_text = f"DEV MODE | Tick: {tick} | Health: {player_health:.0f}/{player_max_health:.0f}"
        text_surface = self.font_large.render(top_text, True, (255, 200, 100))
        self.screen.blit(text_surface, (15, info_y))

        # Second line - controls hint
        hint_y = 30
        hint_text = "W:Forward S:Back A:Left D:Right P:Pass 1:外圈刮 2:飛彈 3:鐵錘 ESC:Quit"
        hint_surface = self.font_small.render(hint_text, True, (150, 150, 150))
        self.screen.blit(hint_surface, (15, hint_y))

        # Bottom info - action details
        bottom_y = self.height - 35
        casting_str = f" [Casting: {int(casting_progress*100)}%]" if is_casting else ""
        action_text = f"Action: {action_name} | Aim: {aim_offset:+.2f} | Event: {event}{casting_str}"
        text_surface = self.font_small.render(action_text, True, COLORS['text'])
        self.screen.blit(text_surface, (15, bottom_y))

    def render_dev_mode(
        self,
        world,
        tick: int,
        action_name: str,
        aim_offset: float,
        event: str,
        mouse_pos: tuple
    ) -> bool:
        """
        Render a full frame in developer mode.

        Returns:
            False if window was closed, True otherwise
        """
        # Draw everything
        self.draw_background()
        self.draw_grid()
        self.draw_boundary()

        # Draw entities
        self.draw_blood_pack(world)
        self.draw_monster(world)

        # Draw projectiles
        self.draw_projectiles(world)

        # Draw skill indicator
        self.draw_skill_indicator(world)

        # Draw player
        self.draw_player(world)

        # Draw aim cursor
        current_aim_offset = self.draw_aim_cursor(world, mouse_pos)

        # Draw casting bar and cooldowns
        self.draw_casting_bar(world)
        self.draw_skill_cooldowns(world)

        # Get player info
        player_health = world.get_player_current_health()
        player_max_health = world.get_player_max_health()
        is_casting = world.player and world.player.has_skills() and world.player.skills.is_casting
        casting_progress = 1.0 - world.get_casting_progress() if is_casting else 0.0

        # Draw dev info panel
        self.draw_dev_info_panel(
            tick, action_name, aim_offset, event,
            player_health, player_max_health,
            is_casting, casting_progress
        )

        # Update display
        pygame.display.flip()

        return True


class DevMode:
    """
    Developer mode controller.

    Allows manual control of the player for debugging game logic.
    The game world only advances when the player provides input.
    """

    # Action mappings
    # 0=FORWARD, 1=BACKWARD, 2=LEFT, 3=RIGHT, 4=OUTER_SLASH, 5=MISSILE, 6=HAMMER, 7=PASS
    ACTION_NAMES = ["FORWARD", "BACKWARD", "LEFT", "RIGHT", "外圈刮", "飛彈", "鐵錘", "PASS"]

    def __init__(self, world_size: float = 10.0):
        """
        Initialize developer mode.

        Args:
            world_size: Size of the game world
        """
        if not PYGAME_AVAILABLE:
            raise ImportError("pygame is required for developer mode")

        self.world_size = world_size
        self.world = GameWorld(Room(size=world_size))
        self.renderer = DevModeRenderer(world_size=world_size)

        # State tracking
        self.tick_count = 0
        self.last_action = "NONE"
        self.last_event = ""
        self.last_aim_offset = 0.0
        self.running = True

    def run(self) -> None:
        """Run the developer mode main loop."""
        print("=" * 50)
        print("Developer Mode Started")
        print("=" * 50)
        print("Controls:")
        print("  W - Move forward")
        print("  S - Move backward")
        print("  A - Turn left")
        print("  D - Turn right")
        print("  P - Pass (advance tick without action)")
        print("  1 - 外圈刮 (Outer Slash) - Ring AOE, no aim needed")
        print("  2 - 飛彈 (Missile) - Projectile, uses mouse aim")
        print("  3 - 鐵錘 (Hammer) - Rectangle, uses mouse aim")
        print("  R - Reset world")
        print("  ESC - Quit")
        print()
        print("Mouse: Controls aim direction during casting (飛彈/鐵錘)")
        print("=" * 50)

        # Reset the world
        self.world.reset()

        while self.running:
            # Get current mouse position for aim calculation
            mouse_pos = pygame.mouse.get_pos()

            # Calculate aim offset from mouse
            aim_offset = self._calculate_aim_offset(mouse_pos)

            # Handle events and wait for input
            action = self._wait_for_input()

            if action is None:
                # Window closed or ESC pressed
                break

            # Execute action if one was provided
            if action is not None:
                self._execute_action(action, aim_offset)

            # Render current state
            if not self.renderer.render_dev_mode(
                self.world,
                self.tick_count,
                self.last_action,
                self.last_aim_offset,
                self.last_event,
                mouse_pos
            ):
                break

        self.renderer.close()
        print("Developer mode ended.")

    def _calculate_aim_offset(self, mouse_pos: tuple) -> float:
        """Calculate aim offset from mouse position."""
        if not self.world.player or not self.world.player.has_position():
            return 0.0

        player_screen_pos = self.renderer.world_to_screen(
            self.world.get_player_position()
        )
        player_angle = self.world.get_player_angle()

        # Calculate angle from player to mouse
        dx = mouse_pos[0] - player_screen_pos[0]
        dy = -(mouse_pos[1] - player_screen_pos[1])  # Y is inverted
        mouse_angle = math.atan2(dy, dx)

        # Calculate offset from facing direction
        aim_offset = mouse_angle - player_angle

        # Normalize to [-π, π]
        while aim_offset > math.pi:
            aim_offset -= 2 * math.pi
        while aim_offset < -math.pi:
            aim_offset += 2 * math.pi

        # Clamp to ±0.5 radians
        return max(-0.5, min(0.5, aim_offset))

    def _wait_for_input(self) -> int:
        """
        Wait for keyboard input and return action code.

        Returns:
            Action code (0-7) or None if quit
            0 = Forward, 1 = Backward, 2 = Left, 3 = Right, 4 = Outer Slash, 5 = Missile, 6 = Hammer, 7 = Pass
        """
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    return None

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                        return None
                    elif event.key == pygame.K_w:
                        return 0  # Forward
                    elif event.key == pygame.K_s:
                        return 1  # Backward
                    elif event.key == pygame.K_a:
                        return 2  # Left
                    elif event.key == pygame.K_d:
                        return 3  # Right
                    elif event.key == pygame.K_1:
                        return 4  # Cast 外圈刮 (Outer Slash)
                    elif event.key == pygame.K_2:
                        return 5  # Cast 飛彈 (Missile)
                    elif event.key == pygame.K_3:
                        return 6  # Cast 鐵錘 (Hammer)
                    elif event.key == pygame.K_p:
                        return 7  # Pass
                    elif event.key == pygame.K_r:
                        # Reset world
                        self.world.reset()
                        self.tick_count = 0
                        self.last_event = "WORLD RESET"
                        return 7  # Pass after reset

            # Update display while waiting
            mouse_pos = pygame.mouse.get_pos()
            self.renderer.render_dev_mode(
                self.world,
                self.tick_count,
                "WAITING...",
                self._calculate_aim_offset(mouse_pos),
                self.last_event,
                mouse_pos
            )

            # Small delay to prevent CPU spinning
            pygame.time.wait(16)  # ~60 FPS

    def _execute_action(self, action: int, aim_offset: float) -> None:
        """
        Execute the given action and advance the game tick.

        Args:
            action: Action code (0-6)
            aim_offset: Aim offset for casting
        """
        if action == 7:  # Pass
            self.last_action = "PASS"
            self.last_event = ""
            self.last_aim_offset = 0.0
        else:
            # Map action to discrete action
            action_discrete = action  # 0=forward, 1=backward, 2=left, 3=right, 4-6=skills

            # Build aim_values list based on action
            # Action 4 (outer_slash): no aim needed
            # Action 5 (missile): uses aim_actor 0
            # Action 6 (hammer): uses aim_actor 1
            aim_values = [0.0, 0.0]  # [aim_missile, aim_hammer]
            if action == 5:
                aim_values[0] = aim_offset  # missile uses actor 0
            elif action == 6:
                aim_values[1] = aim_offset  # hammer uses actor 1

            self.last_action = self.ACTION_NAMES[action]
            self.last_aim_offset = aim_offset if action in (5, 6) else 0.0

            # Execute the action
            self.last_event = self.world.execute_action(action_discrete, aim_values)

        # Advance the game tick
        self.world.tick()
        self.tick_count += 1

        # Check for game-ending conditions
        if self.world.player and not self.world.player.is_alive:
            self.last_event = "PLAYER DIED!"
        elif len(self.world.monsters) == 0:
            self.last_event = "ALL MONSTERS DEAD - VICTORY!"


def main():
    """Entry point for developer mode."""
    dev_mode = DevMode(world_size=10.0)
    dev_mode.run()


if __name__ == "__main__":
    main()
