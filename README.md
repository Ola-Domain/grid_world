Here’s the **updated README** that includes the new stress testing, hyperparameter sweep, and evaluation graphs you ran — this integrates everything you’ve worked hard on:

---

# Deep Q-Learning GridWorld Agent

This project is a **visual, interactive, and analytics-rich implementation** of a **Q-Learning agent** solving a 5×5 **GridWorld** environment using **Reinforcement Learning (RL)**. Built with **Python**, **Pygame**, and **Seaborn/Matplotlib**, it features live visualizations, sound effects, and deep evaluation metrics to track the agent’s learning process.

---

## 🧠 What is Reinforcement Learning?

Reinforcement Learning (RL) is a branch of Machine Learning where an agent learns to make decisions by interacting with an environment to maximise cumulative rewards. In this project:

* The agent learns **which actions lead to the highest long-term gains**.
* A **Q-table** stores estimated rewards (Q-values) for each state-action pair.
* An **epsilon-greedy strategy** balances **exploration** (trying new things) and **exploitation** (using known best actions).

---

## ✨ Features

* ✅ 5×5 bounded GridWorld
* ✅ Special Jump (2,4 → 4,4) with +5 reward
* ✅ Terminal goal (5,5) with +10 reward
* ✅ Obstacles blocking movement
* ✅ Tabular Q-learning with dynamic epsilon decay
* ✅ Sound effects on jump and goal events
* ✅ Sprite-based grid with live HUD showing stats
* ✅ Automatic early stopping when avg reward >10 over 30 episodes
* ✅ Stress tests: no jump, random obstacles
* ✅ Hyperparameter sweep on α, γ, ε-decay
* ✅ Plots and heatmaps for learning curves, epsilon decay, convergence

---

## 📁 File Structure

```
DeepQ-GridWorld/
├── assets/
├── config.py
├── environment.py
├── q_learning.py
├── game.py
├── main.py
├── stress_test_sweep.py
├── plot_sweep_results.py
├── results/
│   ├── q_table.json
│   ├── policy_map.png
│   ├── value_heatmap.png
│   ├── sweep_results.json
│   ├── [training graphs].png
└── README.md
```

---

## 🚀 Setup & Run

```bash
git clone https://github.com/Ola-Domain/grid_world.git
cd grid_world
pip install pygame pandas matplotlib seaborn
python main.py
```

---

## 🎯 Environment Rules

* **Grid Size:** 5×5
* **Start:** (2,1)
* **Terminal:** (5,5), reward +10
* **Jump Tile:** (2,4 → 4,4), reward +5
* **Obstacles:** (3,3), (4,3), (3,4), (3,5)
* **Other moves:** reward -1
* **Actions:** North=1, South=2, East=3, West=4

---

## 🤖 Q-Learning Summary

* Update rule:
  *Q(s,a) ← Q(s,a) + α \[r + γ max(Q(s',a')) − Q(s,a)]*
* Parameters:

  * Learning rate (α): how fast Q-values update
  * Discount factor (γ): weight on future rewards
  * Epsilon (ε): exploration rate, decays over time
* Q-table initialized to zero, updated each step

---

## 📊 Evaluation and Results

* **Live Metrics in HUD:**

  * Episode number
  * Total reward
  * Epsilon value
  * Average reward over last 30 episodes
  * Average episode time

* **Generated Outputs:**

  * 🎉 `results/q_table.json`
  * 📽️ `results/final_episode_replay.mp4`
  * 🧭 `results/policy_map.png`
  * 🌡️ `results/value_heatmap.png`

* **Stress Testing & Hyperparameter Sweep:**

  * Baseline, no-jump, random-obstacle variants
  * Learning curves, epsilon decay, convergence plots
  * Alpha/gamma sweep with heatmaps

---

## 📈 Evaluation Plots

* Learning curve: total reward per episode
* Epsilon decay curve: exploration → exploitation
* Winning percentage over episodes
* Total reward vs. steps
* Average steps per episode
* Q-value distribution
* Final policy map (best action per state)
* Hyperparameter sweep: heatmaps and bar plots

---

## 🔧 Customization Tips

* Replace textures and sounds in `assets/`
* Modify hyperparameters in `q_learning.py`
* Change reward settings in `environment.py`
* Extend with Deep Q-Network (DQN) in future

---

## 💥 Future Work

* Save/load Q-table between runs
* Add stochastic rewards or transitions
* Implement ablation tests (disable jump, random obstacles)
* Integrate DQN with neural networks
* Record gameplay videos/GIFs
* Add manual mode for human-agent comparison

---

## 📌 References

* Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.)
* Watkins, C. J. C. H., & Dayan, P. (1992). *Q-learning*. Machine Learning, 8(3–4), 279–292
* Ulster University COM762 Lecture Slides (2024)
* GitHub: [https://github.com/Ola-Domain/grid\_world](https://github.com/Ola-Domain/grid_world)

---

## 📬 Contact

* **Sulaimon Oladotun Olasupo**
* 
  Computer Science and Technology
  Ulster University, Birmingham, UK
* 📧 [olasupooladotun6@gmail.com](mailto:olasupooladotun6@gmail.com)
* 
  💻 [GitHub Repository](https://github.com/Ola-Domain/grid_world)

---

Happy training, testing, and breaking your agents! 🧠🎯🔥

---

