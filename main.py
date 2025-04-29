import time
import json
import pygame
import os
import csv
from config import START_POS
from environment import GridWorld
from q_learning import QLearningAgent
from game import GridGame
from utils import save_video, save_q_table, render_policy_map, render_heatmap

MAX_EPISODES = 100
EARLY_STOPPING_THRESHOLD = 10
EARLY_STOPPING_WINDOW = 30

env = GridWorld()
agent = QLearningAgent()
game = GridGame()

episode_rewards = []
episode_steps = []
episode_times = []
log_data = []
training_log = []  # for training_log.csv

# Save training log to CSV
def save_csv_log(log_data, path="results/training_log.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Episode", "TotalReward", "Steps", "Epsilon"])
        writer.writeheader()
        writer.writerows(log_data)

# Print Q-table values
def print_value_table(q_table):
    print("-" * 34)
    for row in range(1, 6):
        row_vals = []
        for col in range(1, 6):
            pos = (row, col)
            if pos in q_table:
                v = max(q_table[pos])
                row_vals.append(f"{v:>6.3f}")
            else:
                row_vals.append(" None ")
        print("| " + " | ".join(row_vals) + " |")
        print("-" * 34)

# === Training Loop ===
for episode in range(1, MAX_EPISODES + 1):
    state = env.reset()
    path = [state]
    total_reward = 0
    done = False
    steps = 0
    start_time = time.time()

    while not done:
        action = agent.act(state)

        # Trace
        action_name = ["up", "down", "right", "left"][action - 1]
        print(f"current position {state} action {action_name}")

        next_state, reward, done = env.step(action)

        # 🔊 Sound trigger
        if reward == 5:
            game.jump_sound.play()
        elif reward == 10:
            game.goal_sound.play()

        print(f"nxt state {next_state}")
        print("-" * 30)

        agent.learn(state, action, reward, next_state)
        state = next_state
        path.append(state)
        total_reward += reward
        steps += 1

        avg_reward = sum(episode_rewards[-29:] + [total_reward]) / min(len(episode_rewards)+1, 30)
        stats = {
            "episode": episode,
            "epsilon": agent.epsilon,
            "reward": total_reward,
            "steps": steps,
            "avg_reward": avg_reward,
            "avg_time": 0.0
        }

        game.draw_grid(state, reward=total_reward, visited=path, stats=stats)
        pygame.time.wait(80)

    elapsed = time.time() - start_time
    episode_rewards.append(total_reward)
    episode_steps.append(steps)
    episode_times.append(elapsed)

    log_data.append([episode, total_reward, steps, round(elapsed, 2), round(agent.epsilon, 4)])
    training_log.append({
        "Episode": episode,
        "TotalReward": total_reward,
        "Steps": steps,
        "Epsilon": round(agent.epsilon, 4)
    })

    print(f"Episode {episode}: Total Reward = {total_reward:.2f} | Steps: {steps} | Time: {elapsed:.2f}s")
    print(f"Game End Reward {total_reward}")

    if len(episode_rewards) >= EARLY_STOPPING_WINDOW:
        avg_recent = sum(episode_rewards[-EARLY_STOPPING_WINDOW:]) / EARLY_STOPPING_WINDOW
        if avg_recent > EARLY_STOPPING_THRESHOLD:
            print("✅ Early stopping: average reward > 10")
            break

    agent.decay_epsilon()

print("🎉 Training complete!")
save_csv_log(training_log)
save_q_table(agent.q_table)

# === Final Episode Replay ===
print("▶️ Replaying final training episode...")
env.reset()
frames = []
state = START_POS
replay_path = [state]
done = False
replay_reward = 0
step = 0

while not done:
    if state not in agent.q_table:
        print(f"⚠️ No Q-values for state {state}. Ending replay.")
        break

    action = int(agent.q_table[state].index(max(agent.q_table[state]))) + 1
    next_state, reward, done = env.step(action)

    # 🔊 Sound trigger in replay
    if reward == 5:
        game.jump_sound.play()
    elif reward == 10:
        game.goal_sound.play()

    state = next_state
    replay_path.append(state)
    replay_reward += reward
    step += 1

    game.draw_grid(state, reward=replay_reward, visited=replay_path)
    frames.append(pygame.surfarray.array3d(game.screen))
    pygame.time.wait(100)

save_video(frames, filename="results/final_episode_replay.mp4")
print(f"✅ Final replay in {step} steps with reward {replay_reward:.2f}")
print("📽️ Saved replay to results/final_episode_replay.mp4")

# === Visuals ===
render_policy_map(agent.q_table, game)
render_heatmap(agent.q_table, game)

# Final Value Table
print("\nFinal Value Table:")
print_value_table(agent.q_table)

# Close Window
print("🕹️ Close the window or press ESC to continue...")
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            running = False

pygame.quit()
