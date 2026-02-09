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
    'arrow': (139, 90, 43),           # Brown for arrows
    'magic_bolt': (180, 100, 255),    # Purple for magic bolts
    'magic_glow': (200, 150, 255, 100),  # Light purple glow
    # New skill colors
    'ring_outer': (255, 100, 100, 60),    # Red outer ring
    'ring_inner': (255, 200, 100, 40),    # Orange inner boundary
    'rectangle': (100, 200, 255, 60),     # Blue rectangle
    'rectangle_tip': (255, 100, 100, 80), # Red tip zone
    'missile_aim': (180, 100, 255),       # Purple aim line
    'skill_missile': (100, 200, 255),     # Cyan for skill missile
    'skill_missile_glow': (150, 220, 255, 100),  # Cyan glow
    # Skill cooldown bar colors (one per skill slot)
    'cd_outer_slash': (255, 200, 100),    # Yellow for 外圈刮
    'cd_missile': (180, 100, 255),        # Purple for 飛彈
    'cd_hammer': (100, 200, 255),         # Cyan for 鐵錘
    'cd_dash': (255, 100, 200),           # Pink for 閃現
    'cd_soul_claw': (100, 255, 200),      # Teal for 靈魂爪
    'cd_soul_palm': (255, 150, 100),      # Orange for 靈魂掌
    'cd_ready': (100, 255, 100),          # Green when ready
    'cd_bg': (35, 35, 35),               # Dark background
    # New skill colors
    'dash_target': (255, 100, 200, 80),   # Pink for dash target position
    'dash_arrow': (255, 100, 200),        # Pink for dash direction arrow
    'soul_claw': (100, 255, 200, 60),     # Teal for soul claw rectangle
    'soul_palm': (255, 150, 100, 60),     # Orange for soul palm rectangle
    # Blood pool colors
    'blood_pool': (150, 0, 0, 120),       # Dark red pool
    'blood_pool_emerge': (255, 50, 50, 100),  # Bright red emerge warning
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
        Draw the player entity with health bar.

        Args:
            world: GameWorld instance
        """
        pos = world.get_player_position()
        angle = world.get_player_angle()
        screen_pos = self.world_to_screen(pos)

        self.draw_entity_with_direction(
            screen_pos, angle, COLORS['player'], size=18
        )

        # Draw health bar above player
        health_pct = world.get_player_health_percentage()
        bar_width = 40
        bar_height = 5
        bar_x = screen_pos[0] - bar_width // 2
        bar_y = screen_pos[1] - 30

        # Background
        pygame.draw.rect(
            self.screen, (60, 60, 60),
            (bar_x, bar_y, bar_width, bar_height)
        )
        # Health fill (green -> yellow -> red based on health)
        if health_pct > 0.5:
            health_color = (100, 255, 100)  # Green
        elif health_pct > 0.25:
            health_color = (255, 255, 100)  # Yellow
        else:
            health_color = (255, 100, 100)  # Red
        pygame.draw.rect(
            self.screen, health_color,
            (bar_x, bar_y, int(bar_width * health_pct), bar_height)
        )
        # Border
        pygame.draw.rect(
            self.screen, COLORS['player'],
            (bar_x, bar_y, bar_width, bar_height), 1
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
        Draw skill range indicator based on skill shape type.

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
        shape_type = skills.current_skill_shape_type or "cone"
        extra_params = skills.current_skill_extra_params or {}
        skill_id = skills.current_skill or ""

        # Clear skill surface
        self.skill_surface.fill((0, 0, 0, 0))

        if shape_type == "cone":
            self._draw_cone_indicator(player_pos, aim_angle, skills)
        elif shape_type == "ring":
            self._draw_ring_indicator(player_pos, extra_params)
        elif shape_type == "rectangle":
            # Determine color based on skill type
            if "soul_claw" in skill_id:
                self._draw_rectangle_indicator(player_pos, aim_angle, extra_params, skill_type="soul_claw")
            elif "soul_palm" in skill_id:
                self._draw_rectangle_indicator(player_pos, aim_angle, extra_params, skill_type="soul_palm")
            else:
                self._draw_rectangle_indicator(player_pos, aim_angle, extra_params, skill_type="hammer")
        elif shape_type == "projectile":
            self._draw_projectile_aim_line(player_pos, aim_angle, extra_params)
        elif shape_type == "dash":
            self._draw_dash_indicator(player_pos, aim_angle, extra_params, world.get_player_angle())

        # Blit skill surface onto main screen
        self.screen.blit(self.skill_surface, (0, 0))

    def _draw_cone_indicator(self, player_pos, aim_angle, skills) -> None:
        """Draw cone (fan) shaped skill indicator."""
        range_px = self.world_to_screen_length(skills.current_skill_range)
        tolerance = skills.current_skill_angle_tolerance

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

    def _draw_ring_indicator(self, player_pos, extra_params) -> None:
        """Draw ring (annulus) shaped skill indicator."""
        inner_radius = extra_params.get("inner_radius", 3.0)
        outer_radius = extra_params.get("outer_radius", 4.5)

        inner_px = self.world_to_screen_length(inner_radius)
        outer_px = self.world_to_screen_length(outer_radius)

        # Draw outer circle (filled)
        pygame.draw.circle(
            self.skill_surface,
            COLORS['ring_outer'],
            player_pos,
            outer_px
        )

        # Draw inner circle (cut out by filling with transparent)
        # Since we can't truly cut out, we'll draw the inner circle in a different color
        pygame.draw.circle(
            self.skill_surface,
            (0, 0, 0, 0),  # Transparent
            player_pos,
            inner_px
        )

        # Draw circle outlines on main screen
        pygame.draw.circle(
            self.screen, (255, 100, 100),
            player_pos, outer_px, 2
        )
        pygame.draw.circle(
            self.screen, (255, 200, 100),
            player_pos, inner_px, 2
        )

    def _draw_rectangle_indicator(self, player_pos, aim_angle, extra_params, skill_type="hammer") -> None:
        """Draw rectangle shaped skill indicator with highlighted tip zone or pull/push indication."""
        length = extra_params.get("length", 5.0)
        width = extra_params.get("width", 0.8)
        tip_start = extra_params.get("tip_range_start", 4.0)
        pull_distance = extra_params.get("pull_distance", 0.0)
        push_distance = extra_params.get("push_distance", 0.0)

        length_px = self.world_to_screen_length(length)
        width_px = self.world_to_screen_length(width)
        tip_start_px = self.world_to_screen_length(tip_start)

        # Calculate rectangle corners
        forward = (math.cos(aim_angle), -math.sin(aim_angle))  # Y inverted
        right = (math.sin(aim_angle), math.cos(aim_angle))     # Y inverted

        # Base rectangle corners (from player to length)
        corners = [
            (player_pos[0] - right[0] * width_px / 2,
             player_pos[1] - right[1] * width_px / 2),
            (player_pos[0] + right[0] * width_px / 2,
             player_pos[1] + right[1] * width_px / 2),
            (player_pos[0] + forward[0] * length_px + right[0] * width_px / 2,
             player_pos[1] + forward[1] * length_px + right[1] * width_px / 2),
            (player_pos[0] + forward[0] * length_px - right[0] * width_px / 2,
             player_pos[1] + forward[1] * length_px - right[1] * width_px / 2),
        ]

        # Select color based on skill type
        if skill_type == "soul_claw":
            base_color = COLORS['soul_claw']
            outline_color = (100, 255, 200)
        elif skill_type == "soul_palm":
            base_color = COLORS['soul_palm']
            outline_color = (255, 150, 100)
        else:  # hammer
            base_color = COLORS['rectangle']
            outline_color = (100, 200, 255)

        # Draw base rectangle
        pygame.draw.polygon(
            self.skill_surface,
            base_color,
            corners
        )

        # Draw tip zone (highlighted area at end) - only for hammer
        if skill_type == "hammer" and tip_start > 0:
            tip_corners = [
                (player_pos[0] + forward[0] * tip_start_px - right[0] * width_px / 2,
                 player_pos[1] + forward[1] * tip_start_px - right[1] * width_px / 2),
                (player_pos[0] + forward[0] * tip_start_px + right[0] * width_px / 2,
                 player_pos[1] + forward[1] * tip_start_px + right[1] * width_px / 2),
                (player_pos[0] + forward[0] * length_px + right[0] * width_px / 2,
                 player_pos[1] + forward[1] * length_px + right[1] * width_px / 2),
                (player_pos[0] + forward[0] * length_px - right[0] * width_px / 2,
                 player_pos[1] + forward[1] * length_px - right[1] * width_px / 2),
            ]

            pygame.draw.polygon(
                self.skill_surface,
                COLORS['rectangle_tip'],
                tip_corners
            )

        # Draw outline on main screen
        pygame.draw.polygon(
            self.screen, outline_color,
            corners, 2
        )

        # Draw aim line
        end_x = player_pos[0] + forward[0] * length_px
        end_y = player_pos[1] + forward[1] * length_px
        pygame.draw.line(
            self.screen, COLORS['aim_line'],
            player_pos, (end_x, end_y), 2
        )

    def _draw_dash_indicator(self, player_pos, aim_angle, extra_params, current_angle) -> None:
        """Draw dash skill indicator showing target position and facing."""
        dash_distance = extra_params.get("dash_distance", 3.0)
        dash_facing_offset = extra_params.get("dash_facing_offset", 0.0)

        distance_px = self.world_to_screen_length(dash_distance)

        # Calculate target position
        target_x = player_pos[0] + distance_px * math.cos(aim_angle)
        target_y = player_pos[1] - distance_px * math.sin(aim_angle)  # Y inverted
        target_pos = (int(target_x), int(target_y))

        # Draw path line
        pygame.draw.line(
            self.screen, COLORS['dash_arrow'],
            player_pos, target_pos, 3
        )

        # Draw target position circle
        pygame.draw.circle(
            self.skill_surface,
            COLORS['dash_target'],
            target_pos,
            20
        )
        pygame.draw.circle(
            self.screen, COLORS['dash_arrow'],
            target_pos, 20, 2
        )

        # Draw facing direction arrow at target position
        final_facing = current_angle + dash_facing_offset
        arrow_length = 25
        arrow_end_x = target_x + arrow_length * math.cos(final_facing)
        arrow_end_y = target_y - arrow_length * math.sin(final_facing)  # Y inverted

        # Draw facing arrow
        pygame.draw.line(
            self.screen, COLORS['dash_arrow'],
            target_pos, (arrow_end_x, arrow_end_y), 3
        )

        # Draw arrowhead
        arrow_size = 8
        head_angle1 = final_facing + 2.5
        head_angle2 = final_facing - 2.5
        head1 = (
            arrow_end_x + arrow_size * math.cos(head_angle1),
            arrow_end_y - arrow_size * math.sin(head_angle1)
        )
        head2 = (
            arrow_end_x + arrow_size * math.cos(head_angle2),
            arrow_end_y - arrow_size * math.sin(head_angle2)
        )
        pygame.draw.polygon(
            self.screen, COLORS['dash_arrow'],
            [(arrow_end_x, arrow_end_y), head1, head2]
        )

    def _draw_projectile_aim_line(self, player_pos, aim_angle, extra_params) -> None:
        """Draw projectile aim line."""
        max_range = extra_params.get("max_range", 15.0)
        range_px = self.world_to_screen_length(max_range)

        # Draw aim line
        end_x = player_pos[0] + range_px * math.cos(aim_angle)
        end_y = player_pos[1] - range_px * math.sin(aim_angle)  # Y inverted

        pygame.draw.line(
            self.screen, COLORS['missile_aim'],
            player_pos, (end_x, end_y), 2
        )

        # Draw dashed line effect (small circles along the path)
        num_dots = 10
        for i in range(num_dots):
            t = (i + 1) / num_dots
            dot_x = player_pos[0] + range_px * t * math.cos(aim_angle)
            dot_y = player_pos[1] - range_px * t * math.sin(aim_angle)
            pygame.draw.circle(
                self.screen, COLORS['missile_aim'],
                (int(dot_x), int(dot_y)), 3
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

        # Calculate progress using stored total wind_up ticks
        total_ticks = skills.current_skill_wind_up_total or 4
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

    def draw_skill_cooldowns(self, world) -> None:
        """
        Draw per-skill cooldown bars below the casting bar.

        Shows skill name, a fill bar (dark=on cooldown, bright=ready),
        and remaining tick count.
        """
        cooldown_info = world.get_skill_cooldown_info()
        if not cooldown_info:
            return

        skill_order = ["outer_slash", "missile", "hammer", "dash", "soul_claw", "soul_palm"]
        skill_colors = [
            COLORS['cd_outer_slash'],
            COLORS['cd_missile'],
            COLORS['cd_hammer'],
            COLORS['cd_dash'],
            COLORS['cd_soul_claw'],
            COLORS['cd_soul_palm'],
        ]

        # Position: directly below casting bar area
        section_y = self.margin + self.game_area_size + 42
        bar_height = 8
        label_height = 12
        col_width = self.width // 3  # 3 columns, 2 rows

        for i, (skill_id, color) in enumerate(zip(skill_order, skill_colors)):
            info = cooldown_info.get(skill_id)
            if info is None:
                continue
            remaining, max_cd, name = info

            # Layout: 3 columns × 2 rows
            col = i % 3
            row = i // 3
            bar_w = col_width - 20
            bar_x = col_width * col + 10
            label_y = section_y + row * (bar_height + label_height + 5)
            bar_y = label_y + label_height

            # Determine display color
            if remaining == 0:
                display_color = COLORS['cd_ready']
                status = "READY"
            else:
                display_color = color
                status = str(remaining)

            # Label
            label_text = f"{name}  {status}"
            lbl_surface = self.font_small.render(label_text, True, display_color)
            self.screen.blit(lbl_surface, (bar_x, label_y))

            # Background bar
            pygame.draw.rect(
                self.screen, COLORS['cd_bg'],
                (bar_x, bar_y, bar_w, bar_height)
            )

            # Fill bar (cooldown remaining as dark fill, ready portion as bright)
            if max_cd > 0:
                ready_frac = 1.0 - remaining / max_cd
                fill_w = int(bar_w * ready_frac)
                if remaining == 0:
                    pygame.draw.rect(
                        self.screen, COLORS['cd_ready'],
                        (bar_x, bar_y, bar_w, bar_height)
                    )
                elif fill_w > 0:
                    # Show progress filling from left
                    dim = tuple(max(0, c - 80) for c in color)
                    pygame.draw.rect(
                        self.screen, dim,
                        (bar_x, bar_y, fill_w, bar_height)
                    )

            # Border
            pygame.draw.rect(
                self.screen, display_color,
                (bar_x, bar_y, bar_w, bar_height), 1
            )

    def draw_blood_pool_effect(self, world) -> None:
        """
        Draw blood pool effect when player is in blood pool state.

        Args:
            world: GameWorld instance
        """
        if not world.player or not world.player.has_skills():
            return

        if not world.player.skills.in_blood_pool:
            return

        # Get player position
        pos = world.get_player_position()
        screen_pos = self.world_to_screen(pos)

        # Calculate pool radius (scales with remaining time for pulsing effect)
        remaining = world.player.skills.blood_pool_remaining
        base_radius = 40
        pulse = 1.0 + 0.2 * math.sin(remaining * 0.5)  # Pulsing effect
        radius = int(base_radius * pulse)

        # Draw multiple layers for depth
        # Outer dark red glow
        temp_surface = pygame.Surface((radius * 2 + 20, radius * 2 + 20), pygame.SRCALPHA)
        pygame.draw.circle(temp_surface, COLORS['blood_pool'], (radius + 10, radius + 10), radius + 10)
        self.screen.blit(temp_surface, (screen_pos[0] - radius - 10, screen_pos[1] - radius - 10))

        # Inner brighter circle
        inner_radius = int(radius * 0.7)
        temp_surface2 = pygame.Surface((inner_radius * 2, inner_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(temp_surface2, (180, 0, 0, 150), (inner_radius, inner_radius), inner_radius)
        self.screen.blit(temp_surface2, (screen_pos[0] - inner_radius, screen_pos[1] - inner_radius))

        # Warning ring when about to emerge (last 3 ticks)
        if remaining <= 3:
            emerge_radius = world.player.skills.blood_pool_emerge_radius
            emerge_radius_px = int(emerge_radius * self.game_area_size / self.world_size)

            # Draw emerge warning ring
            temp_surface3 = pygame.Surface((emerge_radius_px * 2 + 10, emerge_radius_px * 2 + 10), pygame.SRCALPHA)
            pygame.draw.circle(temp_surface3, COLORS['blood_pool_emerge'],
                             (emerge_radius_px + 5, emerge_radius_px + 5), emerge_radius_px + 5, 3)
            self.screen.blit(temp_surface3,
                           (screen_pos[0] - emerge_radius_px - 5, screen_pos[1] - emerge_radius_px - 5))

        # Draw remaining time text
        time_text = f"Pool: {remaining}"
        text_surface = self.font_small.render(time_text, True, (255, 200, 200))
        text_rect = text_surface.get_rect(center=(screen_pos[0], screen_pos[1] + 50))
        self.screen.blit(text_surface, text_rect)

    def draw_projectiles(self, world) -> None:
        """
        Draw all active projectiles.

        Args:
            world: GameWorld instance
        """
        projectiles = world.get_active_projectiles()

        for proj in projectiles:
            pos = proj.position
            screen_pos = self.world_to_screen(pos)

            # Get projectile direction for drawing
            direction = proj.direction

            # Import here to avoid circular import
            from game.projectile import ProjectileType

            if proj.projectile_type == ProjectileType.ARROW:
                self._draw_arrow(screen_pos, direction)
            elif proj.projectile_type == ProjectileType.MAGIC_BOLT:
                self._draw_magic_bolt(screen_pos)
            elif proj.projectile_type == ProjectileType.SKILL_MISSILE:
                self._draw_skill_missile(screen_pos, direction)

    def _draw_arrow(self, pos: Tuple[int, int], direction) -> None:
        """Draw an arrow projectile."""
        # Arrow length in pixels
        length = 12
        head_size = 5

        # Calculate end point based on direction
        angle = math.atan2(direction[1], direction[0])
        end_x = pos[0] + length * math.cos(angle)
        end_y = pos[1] - length * math.sin(angle)  # Y inverted

        # Draw arrow shaft
        pygame.draw.line(
            self.screen, COLORS['arrow'],
            pos, (end_x, end_y), 2
        )

        # Draw arrowhead
        head_angle1 = angle + 2.5  # ~143 degrees
        head_angle2 = angle - 2.5
        head1 = (
            end_x + head_size * math.cos(head_angle1),
            end_y - head_size * math.sin(head_angle1)
        )
        head2 = (
            end_x + head_size * math.cos(head_angle2),
            end_y - head_size * math.sin(head_angle2)
        )
        pygame.draw.polygon(
            self.screen, COLORS['arrow'],
            [(end_x, end_y), head1, head2]
        )

    def _draw_magic_bolt(self, pos: Tuple[int, int]) -> None:
        """Draw a magic bolt projectile with glow effect."""
        # Draw glow (larger circle, semi-transparent)
        glow_radius = 10
        glow_surface = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(
            glow_surface,
            COLORS['magic_glow'],
            (glow_radius, glow_radius),
            glow_radius
        )
        self.screen.blit(
            glow_surface,
            (pos[0] - glow_radius, pos[1] - glow_radius)
        )

        # Draw core (smaller solid circle)
        core_radius = 5
        pygame.draw.circle(
            self.screen, COLORS['magic_bolt'],
            pos, core_radius
        )

        # Draw bright center
        pygame.draw.circle(
            self.screen, (255, 220, 255),
            pos, 2
        )

    def _draw_skill_missile(self, pos: Tuple[int, int], direction) -> None:
        """Draw a skill missile projectile with glow effect."""
        # Draw glow (larger circle, semi-transparent)
        glow_radius = 12
        glow_surface = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(
            glow_surface,
            COLORS['skill_missile_glow'],
            (glow_radius, glow_radius),
            glow_radius
        )
        self.screen.blit(
            glow_surface,
            (pos[0] - glow_radius, pos[1] - glow_radius)
        )

        # Draw core (elongated in direction of travel)
        angle = math.atan2(direction[1], direction[0])
        core_length = 8
        core_width = 4

        # Calculate tail position
        tail_x = pos[0] - core_length * math.cos(angle)
        tail_y = pos[1] + core_length * math.sin(angle)  # Y inverted

        # Draw elongated core
        pygame.draw.line(
            self.screen, COLORS['skill_missile'],
            (tail_x, tail_y), pos, core_width
        )

        # Draw bright center
        pygame.draw.circle(
            self.screen, (200, 255, 255),
            pos, 4
        )

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
            action_names = ["MOVE", "LEFT", "RIGHT", "外圈刮", "飛彈", "鐵錘"]

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

        # Draw projectiles
        self.draw_projectiles(world)

        # Draw skill indicator (before player so it's behind)
        self.draw_skill_indicator(world)

        # Draw blood pool effect (if player is in blood pool)
        self.draw_blood_pool_effect(world)

        # Draw player
        self.draw_player(world)

        # Draw UI
        self.draw_casting_bar(world)
        self.draw_skill_cooldowns(world)
        self.draw_info_panel(epoch, reward, sigma, action, continuous, event, action_names)

        # Update display
        pygame.display.flip()

        # Limit frame rate
        self.clock.tick(15)  # 15 FPS for clearer observation

        return True

    def close(self) -> None:
        """Clean up Pygame resources."""
        pygame.quit()
