import os
import csv
import json
import cv2
import pygame
import numpy as np

def save_q_table(q_table, path="results/q_table.json"):
    os.makedirs("results", exist_ok=True)
    serialised_q = {str(k): v for k, v in q_table.items()}  # tuple keys to strings
    with open(path, "w") as f:
        json.dump(serialised_q, f, indent=4)
    print("💾 Q-table saved to", path)

def save_csv_log(log_data, path="results/training_log.csv"):
    os.makedirs("results", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "Reward", "Steps", "Time", "Epsilon"])
        writer.writerows(log_data)
    print("📊 Log saved to", path)

def save_video(frames, filename="results/final_episode_replay.mp4", fps=5):
    os.makedirs("results", exist_ok=True)
    if not frames:
        print("⚠️ No frames to save.")
        return
    height, width = frames[0].shape[:2]
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)
    out.release()
    print(f"📽️ Saved replay to {filename}")

def render_policy_map(q_table, game):
    pygame.init()
    screen = game.screen.copy()
    arrow_font = pygame.font.SysFont("arial", 20, bold=True)
    for (row, col), q_values in q_table.items():
        best_action = np.argmax(q_values)
        center = (
            game.padding + (col - 1) * game.cell_size + game.cell_size // 2,
            game.padding + (row - 1) * game.cell_size + game.cell_size // 2
        )
        arrows = ["↑", "↓", "→", "←"]
        arrow = arrows[best_action]
        text = arrow_font.render(arrow, True, (0, 0, 0))
        text_rect = text.get_rect(center=center)
        screen.blit(text, text_rect)

    pygame.image.save(screen, "results/policy_map.png")
    print("✅ Policy map saved to results/policy_map.png")
    pygame.display.set_caption("Policy Map")
    game.screen.blit(screen, (0, 0))
    pygame.display.flip()

def render_heatmap(q_table, game):
    pygame.init()
    screen = game.screen.copy()
    max_q = max([max(q) for q in q_table.values()]) if q_table else 1
    min_q = min([min(q) for q in q_table.values()]) if q_table else 0

    for (row, col), q_values in q_table.items():
        value = max(q_values)
        norm_val = (value - min_q) / (max_q - min_q + 1e-6)
        colour = pygame.Color(0)
        colour.hsva = (120 * norm_val, 100, 100, 100)
        rect = pygame.Rect(
            game.padding + (col - 1) * game.cell_size,
            game.padding + (row - 1) * game.cell_size,
            game.cell_size, game.cell_size
        )
        pygame.draw.rect(screen, colour, rect)
        text = pygame.font.SysFont("arial", 14).render(f"{value:.1f}", True, (0, 0, 0))
        screen.blit(text, text.get_rect(center=rect.center))

    pygame.image.save(screen, "results/value_heatmap.png")
    print("🌡️ Value heatmap saved to results/value_heatmap.png")
    pygame.display.set_caption("Value Heatmap")
    game.screen.blit(screen, (0, 0))
    pygame.display.flip()
