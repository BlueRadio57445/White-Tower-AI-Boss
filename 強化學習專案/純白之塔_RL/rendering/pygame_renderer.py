"""
Pygame-based renderer for visualizing the game world.
Displays entities with direction indicators, skill ranges, and casting progress.
"""

import math
from typing import Tuple, Optional

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False


# Color definitions (RGB)
COLORS = {
    'background': (30, 30, 30),
    'boundary': (200, 200, 200),
    'grid': (50, 50, 50),
    'player': (100, 150, 255),
    'monster': (255, 100, 100),
    'blood_pack': (100, 255, 100),
    'skill_range': (255, 255, 0, 80),  # Yellow with alpha
    'aim_line': (255, 165, 0),
    'progress_bar': (0, 255, 255),
    'progress_bg': (60, 60, 60),
    'text': (220, 220, 220),
    'info_bg': (40, 40, 40),
}


class PygameRenderer:
    """
    Pygame-based renderer for the game world.

    Features:
    - Direction-indicating triangles for entities
    - Skill range visualization (fan-shaped area)
    - Casting progress bar
    - Information panel
    """

    def __init__(self, width: int = 600, height: int = 700, world_size: float = 10.0):
        """
        Initialize the Pygame renderer.

        Args:
            width: Window width in pixels
            height: Window height in pixels
            world_size: World coordinate size (default 10x10)
        """
        if not PYGAME_AVAILABLE:
            raise ImportError("pygame is not installed. Install with: pip install pygame")

        self.width = width
        self.height = height
        self.world_size = world_size

        # Calculate game area dimensions
        self.margin = 60
        self.info_panel_height = 100
        self.game_area_size = min(width - 2 * self.margin,
                                   height - self.info_panel_height - 2 * self.margin)

        # Initialize Pygame
        pygame.init()
        pygame.display.set_caption("純白之塔 RL - Training Visualization")
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()

        # Font for text
        pygame.font.init()
        self.font_large = pygame.font.SysFont('consolas', 18)
        self.font_small = pygame.font.SysFont('consolas', 14)

        # Create a surface for skill range (with alpha)
        self.skill_surface = pygame.Surface((width, height), pygame.SRCALPHA)

    def world_to_screen(self, world_pos) -> Tuple[int, int]:
        """
        Convert world coordinates to screen pixels.

        Args:
            world_pos: World position as [x, y] array or tuple

        Returns:
            Screen position as (x, y) tuple
        """
        scale = self.game_area_size / self.world_size

        screen_x = self.margin + world_pos[0] * scale
        # Y is flipped (screen Y increases downward, world Y increases upward)
        screen_y = self.margin + (self.world_size - world_pos[1]) * scale

        return (int(screen_x), int(screen_y))

    def world_to_screen_length(self, length: float) -> int:
        """
        Convert a world-space length to screen pixels.

        Args:
            length: Length in world units

        Returns:
            Length in pixels
        """
        scale = self.game_area_size / self.world_size
        return int(length * scale)

    def draw_background(self) -> None:
        """Draw the background color."""
        self.screen.fill(COLORS['background'])

    def draw_boundary(self) -> None:
        """Draw the game area boundary."""
        rect = pygame.Rect(
            self.margin,
            self.margin,
            self.game_area_size,
            self.game_area_size
        )
        pygame.draw.rect(self.screen, COLORS['boundary'], rect, 2)

    def draw_grid(self) -> None:
        """Draw optional grid lines."""
        scale = self.game_area_size / self.world_size

        for i in range(int(self.world_size) + 1):
            # Vertical lines
            x = self.margin + i * scale
            pygame.draw.line(
                self.screen, COLORS['grid'],
                (x, self.margin), (x, self.margin + self.game_area_size)
            )
            # Horizontal lines
            y = self.margin + i * scale
            pygame.draw.line(
                self.screen, COLORS['grid'],
                (self.margin, y), (self.margin + self.game_area_size, y)
            )

    def draw_entity_with_direction(
        self,
        pos: Tuple[int, int],
        angle: float,
        color: Tuple[int, int, int],
        size: int = 15
    ) -> None:
        """
        Draw a directional triangle for an entity.

        Args:
            pos: Screen position (x, y)
            angle: Facing angle in radians (0 = right, pi/2 = up)
            color: RGB color tuple
            size: Triangle size in pixels
        """
        # Calculate triangle vertices
        # Tip points in the direction of angle
        # Note: screen Y is inverted, so we negate sin for Y
        tip = (
            pos[0] + size * math.cos(angle),
            pos[1] - size * math.sin(angle)
        )

        # Back-left vertex (at angle + 140 degrees)
        back_angle_left = angle + 2.44  # ~140 degrees
        left = (
            pos[0] + size * 0.6 * math.cos(back_angle_left),
            pos[1] - size * 0.6 * math.sin(back_angle_left)
        )

        # Back-right vertex (at angle - 140 degrees)
        back_angle_right = angle - 2.44
        right = (
            pos[0] + size * 0.6 * math.cos(back_angle_right),
            pos[1] - size * 0.6 * math.sin(back_angle_right)
        )

        # Draw filled triangle
        pygame.draw.polygon(self.screen, color, [tip, left, right])

        # Draw outline for better visibility
        darker = tuple(max(0, c - 50) for c in color)
        pygame.draw.polygon(self.screen, darker, [tip, left, right], 2)

    def draw_player(self, world) -> None:
        """
        Draw the player entity.

        Args:
            world: GameWorld instance
        """
        pos = world.get_player_position()
        angle = world.get_player_angle()
        screen_pos = self.world_to_screen(pos)

        self.draw_entity_with_direction(
            screen_pos, angle, COLORS['player'], size=18
        )

    def draw_monster(self, world) -> None:
        """
        Draw all monster entities.

        Args:
            world: GameWorld instance
        """
        if not world.monsters:
            return

        for monster in world.monsters:
            if not monster.is_alive or not monster.has_position():
                continue

            pos = monster.position.as_array()
            angle = monster.position.angle
            screen_pos = self.world_to_screen(pos)

            # Draw monster as triangle
            self.draw_entity_with_direction(
                screen_pos, angle, COLORS['monster'], size=16
            )

            # Draw health bar above monster
            if monster.has_health():
                bar_width = 30
                bar_height = 4
                health_pct = monster.health.percentage

                bar_x = screen_pos[0] - bar_width // 2
                bar_y = screen_pos[1] - 25

                # Background
                pygame.draw.rect(
                    self.screen, (60, 60, 60),
                    (bar_x, bar_y, bar_width, bar_height)
                )
                # Health fill
                pygame.draw.rect(
                    self.screen, COLORS['monster'],
                    (bar_x, bar_y, int(bar_width * health_pct), bar_height)
                )

    def draw_blood_pack(self, world) -> None:
        """
        Draw the blood pack item.

        Args:
            world: GameWorld instance
        """
        pos = world.get_blood_pack_position()
        screen_pos = self.world_to_screen(pos)

        # Draw as diamond shape
        size = 10
        points = [
            (screen_pos[0], screen_pos[1] - size),      # top
            (screen_pos[0] + size, screen_pos[1]),      # right
            (screen_pos[0], screen_pos[1] + size),      # bottom
            (screen_pos[0] - size, screen_pos[1]),      # left
        ]
        pygame.draw.polygon(self.screen, COLORS['blood_pack'], points)
        pygame.draw.polygon(self.screen, (50, 180, 50), points, 2)

    def draw_skill_indicator(self, world) -> None:
        """
        Draw skill range indicator (fan shape) and aim line.

        Args:
            world: GameWorld instance
        """
        if not world.player or not world.player.has_skills():
            return

        skills = world.player.skills

        if not skills.is_casting:
            return

        player_pos = self.world_to_screen(world.get_player_position())
        aim_angle = skills.aim_angle

        # Skill parameters (from basic_attack)
        range_px = self.world_to_screen_length(6.0)
        tolerance = 0.4  # radians (~23 degrees each side)

        # Clear skill surface
        self.skill_surface.fill((0, 0, 0, 0))

        # Draw fan shape using polygon approximation
        start_angle = aim_angle - tolerance
        end_angle = aim_angle + tolerance

        # Create fan polygon points
        fan_points = [player_pos]
        num_segments = 20
        for i in range(num_segments + 1):
            angle = start_angle + (end_angle - start_angle) * i / num_segments
            x = player_pos[0] + range_px * math.cos(angle)
            y = player_pos[1] - range_px * math.sin(angle)  # Y inverted
            fan_points.append((x, y))

        # Draw semi-transparent fan
        pygame.draw.polygon(
            self.skill_surface,
            COLORS['skill_range'],
            fan_points
        )

        # Blit skill surface onto main screen
        self.screen.blit(self.skill_surface, (0, 0))

        # Draw aim line (center of fan)
        end_x = player_pos[0] + range_px * math.cos(aim_angle)
        end_y = player_pos[1] - range_px * math.sin(aim_angle)
        pygame.draw.line(
            self.screen, COLORS['aim_line'],
            player_pos, (end_x, end_y), 2
        )

        # Draw arc outline for visibility
        pygame.draw.lines(
            self.screen, COLORS['aim_line'],
            False, fan_points[1:], 1
        )

    def draw_casting_bar(self, world) -> None:
        """
        Draw casting progress bar at bottom of game area.

        Args:
            world: GameWorld instance
        """
        if not world.player or not world.player.has_skills():
            return

        skills = world.player.skills

        if not skills.is_casting:
            return

        # Calculate progress (wind_up_remaining decreases from 4 to 0)
        # Progress is (4 - remaining) / 4
        total_ticks = 4  # wind_up_ticks for basic_attack
        progress = (total_ticks - skills.wind_up_remaining) / total_ticks

        # Bar dimensions
        bar_width = 200
        bar_height = 20
        bar_x = self.width // 2 - bar_width // 2
        bar_y = self.margin + self.game_area_size + 15

        # Background
        pygame.draw.rect(
            self.screen, COLORS['progress_bg'],
            (bar_x, bar_y, bar_width, bar_height)
        )

        # Progress fill
        fill_width = int(bar_width * progress)
        pygame.draw.rect(
            self.screen, COLORS['progress_bar'],
            (bar_x, bar_y, fill_width, bar_height)
        )

        # Border
        pygame.draw.rect(
            self.screen, COLORS['boundary'],
            (bar_x, bar_y, bar_width, bar_height), 2
        )

        # Progress text
        pct_text = f"{int(progress * 100)}% Wind-up"
        text_surface = self.font_small.render(pct_text, True, COLORS['text'])
        text_rect = text_surface.get_rect(center=(self.width // 2, bar_y + bar_height // 2))
        self.screen.blit(text_surface, text_rect)

    def draw_info_panel(
        self,
        epoch: int,
        reward: float,
        sigma: float,
        action: int,
        continuous: float,
        event: str,
        action_names: Optional[list] = None
    ) -> None:
        """
        Draw information panel at top and bottom.

        Args:
            epoch: Current training epoch
            reward: Current episode reward
            sigma: Exploration sigma
            action: Discrete action index
            continuous: Continuous action value
            event: Event string
            action_names: List of action names
        """
        if action_names is None:
            action_names = ["MOVE", "LEFT", "RIGHT", "CAST"]

        # Top info bar
        info_y = 10
        top_text = f"Ep: {epoch} | Reward: {reward:.1f} | Sigma: {sigma:.3f}"
        text_surface = self.font_large.render(top_text, True, COLORS['text'])
        self.screen.blit(text_surface, (15, info_y))

        # Bottom info - action details
        bottom_y = self.height - 35
        action_name = action_names[action] if 0 <= action < len(action_names) else "?"
        action_text = f"Action: {action_name} | Aim: {continuous:+.2f} | Event: {event}"
        text_surface = self.font_small.render(action_text, True, COLORS['text'])
        self.screen.blit(text_surface, (15, bottom_y))

    def render(
        self,
        world,
        epoch: int,
        reward: float,
        sigma: float,
        action: int,
        continuous: float,
        event: str,
        action_names: Optional[list] = None
    ) -> bool:
        """
        Render a full frame.

        Args:
            world: GameWorld instance
            epoch: Current training epoch
            reward: Current episode reward
            sigma: Exploration sigma
            action: Discrete action index
            continuous: Continuous action value
            event: Event string
            action_names: List of action names

        Returns:
            False if window was closed, True otherwise
        """
        # Handle Pygame events
        for event_obj in pygame.event.get():
            if event_obj.type == pygame.QUIT:
                return False

        # Draw everything
        self.draw_background()
        self.draw_grid()
        self.draw_boundary()

        # Draw entities
        self.draw_blood_pack(world)
        self.draw_monster(world)

        # Draw skill indicator (before player so it's behind)
        self.draw_skill_indicator(world)

        # Draw player
        self.draw_player(world)

        # Draw UI
        self.draw_casting_bar(world)
        self.draw_info_panel(epoch, reward, sigma, action, continuous, event, action_names)

        # Update display
        pygame.display.flip()

        # Limit frame rate
        self.clock.tick(15)  # 15 FPS for clearer observation

        return True

    def close(self) -> None:
        """Clean up Pygame resources."""
        pygame.quit()
