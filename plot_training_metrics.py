import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

# === Load training log ===
log_path = "results/training_log.csv"
if not os.path.exists(log_path):
    raise FileNotFoundError("Run main.py first to generate training_log.csv")

log = pd.read_csv(log_path)

# === Load Q-table (optional for stability plot) ===
q_table_path = "results/q_table.json"
if os.path.exists(q_table_path):
    with open(q_table_path) as f:
        raw_q = json.load(f)
    q_table = {eval(k): v for k, v in raw_q.items()}
    avg_q_value = np.mean([max(v) for v in q_table.values()])
else:
    q_table = None
    avg_q_value = None

# === Ensure plot output folder ===
plot_dir = "results/plots"
os.makedirs(plot_dir, exist_ok=True)

# === Extract log values ===
episodes = log['Episode']
rewards = log['TotalReward']
steps = log['Steps']
epsilons = log['Epsilon']

# === 1. Learning Curve ===
plt.figure(figsize=(8, 4))
plt.plot(episodes, rewards, color='blue')
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("🎯 Learning Curve: Reward per Episode")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{plot_dir}/learning_curve.png")
plt.close()

# === 2. Exploration vs Exploitation ===
plt.figure(figsize=(8, 4))
plt.plot(episodes, epsilons, color='purple')
plt.xlabel("Episode")
plt.ylabel("Epsilon")
plt.title("🧪 Exploration vs Exploitation (Epsilon Decay)")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{plot_dir}/epsilon_decay.png")
plt.close()

# === 3. Winning Percentage ===
success = rewards >= 10
rolling_win = success.rolling(window=10).mean()
plt.figure(figsize=(8, 4))
plt.plot(episodes, rolling_win, color='green')
plt.xlabel("Episode")
plt.ylabel("Winning Rate")
plt.title("🏆 Winning Percentage Over Time")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{plot_dir}/winning_percentage.png")
plt.close()

# === 4. Reward vs Steps ===
plt.figure(figsize=(8, 4))
plt.scatter(steps, rewards, c='orange', alpha=0.7)
plt.xlabel("Steps Taken")
plt.ylabel("Total Reward")
plt.title("🕹️ Total Reward vs Steps per Episode")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{plot_dir}/reward_vs_steps.png")
plt.close()

# === 5. Average Q-value stability ===
if q_table:
    q_values = [max(v) for v in q_table.values()]
    plt.figure(figsize=(8, 4))
    plt.hist(q_values, bins=10, color='cyan', edgecolor='black')
    plt.xlabel("Max Q-value")
    plt.ylabel("Frequency")
    plt.title(f"🧠 Q-Value Distribution (Mean: {avg_q_value:.2f})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/q_value_distribution.png")
    plt.close()

# === 6. Average Steps per Episode ===
rolling_steps = steps.rolling(window=10).mean()
plt.figure(figsize=(8, 4))
plt.plot(episodes, rolling_steps, color='brown')
plt.xlabel("Episode")
plt.ylabel("Avg Steps (10-episode)")
plt.title("📉 Average Steps per Episode")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{plot_dir}/avg_steps_per_episode.png")
plt.close()

# === 7. Smoothed Reward Curve ===
rolling_reward = rewards.rolling(window=10).mean()
plt.figure(figsize=(8, 4))
plt.plot(episodes, rolling_reward, color='navy', label="Moving Avg Reward")
plt.plot(episodes, rewards, alpha=0.3, color='grey', label="Raw Reward")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("📈 Rolling Reward Curve (Window=10)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{plot_dir}/smoothed_reward_curve.png")
plt.close()

print("✅ All learning and diagnostic plots saved to results/plots/")
