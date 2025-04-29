import pygame
import os
import numpy as np
from config import GRID_SIZE, CELL_SIZE, PADDING, WIDTH, HEIGHT, OBSTACLES, SPECIAL_JUMP, TERMINAL_STATE

class GridGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
        pygame.display.set_caption("GridWorld Q-Learning")
        self.font = pygame.font.SysFont(None, 30)

        # Expose for utils
        self.padding = PADDING
        self.cell_size = CELL_SIZE
        self.obstacles = OBSTACLES

        # Load visual assets
        self.background = pygame.image.load(os.path.join("assets", "background.png"))
        self.background = pygame.transform.scale(self.background, (WIDTH, HEIGHT))

        self.agent_sprite = pygame.image.load(os.path.join("assets", "agent.png"))
        self.agent_sprite = pygame.transform.scale(self.agent_sprite, (CELL_SIZE, CELL_SIZE))

        self.goal_texture = pygame.image.load(os.path.join("assets", "goal.png"))
        self.goal_texture = pygame.transform.scale(self.goal_texture, (CELL_SIZE, CELL_SIZE))

        self.jump_texture = pygame.image.load(os.path.join("assets", "jump.png"))
        self.jump_texture = pygame.transform.scale(self.jump_texture, (CELL_SIZE, CELL_SIZE))

        self.obstacle_texture = pygame.image.load(os.path.join("assets", "obstacle.png"))
        self.obstacle_texture = pygame.transform.scale(self.obstacle_texture, (CELL_SIZE, CELL_SIZE))

        self.jump_sound = pygame.mixer.Sound(os.path.join("assets", "jump.wav"))
        self.goal_sound = pygame.mixer.Sound(os.path.join("assets", "goal.wav"))

    def draw_grid(self, agent_pos, reward=None, visited=None, replay_path=None, stats=None):
        self.screen.blit(self.background, (0, 0))

        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                x = self.padding + col * self.cell_size
                y = self.padding + row * self.cell_size
                pos = (row + 1, col + 1)

                if pos in OBSTACLES:
                    self.screen.blit(self.obstacle_texture, (x, y))
                elif pos == TERMINAL_STATE:
                    self.screen.blit(self.goal_texture, (x, y))
                elif pos in SPECIAL_JUMP:
                    self.screen.blit(self.jump_texture, (x, y))
                else:
                    tile = pygame.Surface((self.cell_size, self.cell_size))
                    tile.set_alpha(60)
                    tile.fill((240, 240, 240))
                    self.screen.blit(tile, (x, y))

                pygame.draw.rect(self.screen, (0, 0, 255), pygame.Rect(x, y, self.cell_size, self.cell_size), 1)

        if visited:
            for pos in visited:
                if pos != agent_pos:
                    x = self.padding + (pos[1] - 1) * self.cell_size
                    y = self.padding + (pos[0] - 1) * self.cell_size
                    trail = pygame.Surface((self.cell_size, self.cell_size))
                    trail.set_alpha(80)
                    trail.fill((100, 100, 100))
                    self.screen.blit(trail, (x, y))

        if replay_path:
            for pos in replay_path:
                if pos != agent_pos:
                    x = self.padding + (pos[1] - 1) * self.cell_size
                    y = self.padding + (pos[0] - 1) * self.cell_size
                    green = pygame.Surface((self.cell_size, self.cell_size))
                    green.set_alpha(120)
                    green.fill((0, 255, 0))
                    self.screen.blit(green, (x, y))

        agent_x = self.padding + (agent_pos[1] - 1) * self.cell_size
        agent_y = self.padding + (agent_pos[0] - 1) * self.cell_size
        self.screen.blit(self.agent_sprite, (agent_x, agent_y))

        for i in range(GRID_SIZE):
            label = self.font.render(str(i + 1), True, (0, 0, 0))
            self.screen.blit(label, (self.padding - 30, self.padding + i * self.cell_size + 40))
            self.screen.blit(label, (self.padding + i * self.cell_size + 40, self.padding - 30))

        if reward is not None:
            reward_text = self.font.render(f"Reward: {reward:.2f}", True, (0, 0, 0))
            self.screen.blit(reward_text, (self.padding, HEIGHT - 70))

        if stats:
            panel_x = self.padding
            panel_y = HEIGHT - 40
            pygame.draw.rect(self.screen, (30, 30, 30), (panel_x, panel_y, 750, 40), border_radius=8)
            pygame.draw.rect(self.screen, (0, 255, 128), (panel_x, panel_y, 750, 40), 2, border_radius=8)
            text = self.font.render(
                f"EP: {stats['episode']}   ε: {stats['epsilon']:.2f}   "
                f"Reward: {stats['reward']:.1f}   Steps: {stats['steps']}   "
                f"Avg(30): {stats['avg_reward']:.1f}   Time: {stats['avg_time']:.2f}s",
                True, (0, 255, 128)
            )
            self.screen.blit(text, (panel_x + 10, panel_y + 8))

        pygame.display.flip()