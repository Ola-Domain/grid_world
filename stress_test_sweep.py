import itertools
import json
import random
import numpy as np
from environment import GridWorld
from q_learning import QLearningAgent

def run_experiment(alpha, gamma, epsilon_decay, remove_jump=False, custom_obstacles=None):
    env = GridWorld()
    if remove_jump:
        env.special_jump = {}
    if custom_obstacles:
        env.obstacles = custom_obstacles

    agent = QLearningAgent(alpha=alpha, gamma=gamma, epsilon=1.0)
    max_episodes = 100
    early_stop = 10
    window = 30

    rewards = []
    steps = []
    for episode in range(1, max_episodes + 1):
        state = env.reset()
        total_reward = 0
        done = False
        count_steps = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            count_steps += 1

        agent.decay_epsilon(min_epsilon=0.1, decay_rate=epsilon_decay)
        rewards.append(total_reward)
        steps.append(count_steps)

        if episode >= window:
            avg_recent = np.mean(rewards[-window:])
            if avg_recent > early_stop:
                break

    summary = {
        "episodes_run": episode,
        "avg_reward_last30": np.mean(rewards[-window:]),
        "final_epsilon": agent.epsilon,
        "avg_steps": np.mean(steps)
    }
    return summary

# Hyperparameter grids
alphas = [0.1, 0.5, 0.9]
gammas = [0.5, 0.7, 0.9, 0.99]
epsilon_decays = [0.99, 0.95, 0.9]

# Stress test configs
stress_tests = [
    {"name": "baseline", "remove_jump": False, "custom_obstacles": None},
    {"name": "no_jump", "remove_jump": True, "custom_obstacles": None},
    {"name": "random_obstacles", "remove_jump": False,
     "custom_obstacles": random.sample([(r, c) for r in range(1,6) for c in range(1,6)
     if (r,c) not in [(2,1),(5,5)]], 4)}
]

# Run everything
results = []
for test in stress_tests:
    for alpha, gamma, eps_decay in itertools.product(alphas, gammas, epsilon_decays):
        summary = run_experiment(alpha, gamma, eps_decay,
                                 remove_jump=test["remove_jump"],
                                 custom_obstacles=test["custom_obstacles"])
        summary.update({
            "test_name": test["name"],
            "alpha": alpha,
            "gamma": gamma,
            "epsilon_decay": eps_decay
        })
        results.append(summary)
        print(f"{test['name']} | alpha={alpha} gamma={gamma} eps_decay={eps_decay} → "
              f"episodes: {summary['episodes_run']}, avg_reward_last30: {summary['avg_reward_last30']:.2f}, "
              f"final_eps: {summary['final_epsilon']:.2f}, avg_steps: {summary['avg_steps']:.2f}")

# Save to JSON
with open("results/sweep_results.json", "w") as f:
    json.dump(results, f, indent=4)
