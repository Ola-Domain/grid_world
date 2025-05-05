# Deep Q-Learning GridWorld Agent

This project is a visual and interactive implementation of a **Q-Learning agent** solving a 5x5 **GridWorld** environment using **Reinforcement Learning (RL)**. Built with **Python** and **Pygame**, it features custom textures, sound effects, and a responsive stats HUD to monitor the agent's learning process in real time.

---

## 🧠 What is Reinforcement Learning?
Reinforcement Learning is a type of Machine Learning where an agent learns to make decisions by interacting with an environment to maximize cumulative rewards. In this project:

- The agent learns **which actions lead to better long-term outcomes**.
- A **Q-table** is used to store expected rewards (Q-values) for each state-action pair.
- The agent uses an **epsilon-greedy strategy** to balance **exploration** (trying new actions) and **exploitation** (choosing the best known action).

---

## ✨ Features
- ✅ 5x5 bounded GridWorld
- ✅ Special Jump from (2,4) → (4,4) with +5 reward
- ✅ Terminal state at (5,5) with +10 reward
- ✅ Obstacles block movement
- ✅ Q-learning with dynamic epsilon decay
- ✅ Visual grid with sprite-based rendering and digital dashboard
- ✅ Sound effects for terminal and jump events
- ✅ Reward, average reward, epsilon, and time per episode displayed
- ✅ Trail of visited cells drawn during each episode
- ✅ Early stopping if average reward over 30 episodes > 10

---

## 📁 File Structure
```
DeepQ-GridWorld/
├── assets/
│   ├── agent.png
│   ├── background.png
│   ├── goal.png
│   ├── jump.png
│   ├── obstacle.png
│   ├── goal.wav
│   └── jump.wav
├── config.py              # Environment settings
├── environment.py         # Step function and grid logic
├── q_learning.py          # Q-table learning agent
├── game.py                # Pygame rendering and HUD
├── main.py                # Main training loop
└── README.md
```

---

## 🚀 Installation & Setup
### 🔹 Step 1: Clone the Repository
```bash
git clone https://github.com/Ola-Domain/QGridWorld.git
cd QGridWorld
```

### 🔹 Step 2: Install Dependencies
```bash
pip install pygame
```

### 🔹 Step 3: Run the Simulator
```bash
python main.py
```

You will see a live rendering of the GridWorld, the agent moving step by step, and training statistics at the bottom.

---

## 🎯 Environment Rules
- Grid Size: 5×5
- Start: (2, 1)
- Terminal: (5, 5) — reward +10
- Special Jump: (2, 4) → (4, 4) — reward +5
- Obstacles: (3,3), (4,3), (3,4), (3,5)
- All other movements: reward -1
- Actions:
  - North = 1
  - South = 2
  - East = 3
  - West = 4

---

## 🤖 Q-Learning Details
- Q(s, a) ← Q(s, a) + α [r + γ max(Q(s', a')) - Q(s, a)]
- **Learning Rate (α)** = how fast we update
- **Discount Factor (γ)** = how much future rewards matter
- **Epsilon (ε)** = exploration rate; decays over episodes
- Q-table is initialized empty and updated through interactions

---

## 📊 Live Training Stats
- 🔁 Episode number
- 📈 Total reward in current episode
- 📉 Epsilon value (exploration vs exploitation)
- 📊 Avg reward over last 30 episodes
- ⏱️ Avg time per episode
- 💡 Displayed in a styled digital HUD at the bottom of the screen

---

## 🔧 Customization Tips
- 🎨 Replace textures in `assets/` to change visuals
- 🎼 Add your own `.wav` sound files for jumps or goals
- ⚙️ Tune learning rate, gamma, and epsilon decay in `q_learning.py`
- 🧠 Modify reward scheme in `environment.py` for experimentation

---

## 💡 Future Work
- 🔄 Save and load the Q-table to persist learning progress
- 🧭 Visualize the learned policy with arrows or directional cues
- 📈 Export stats/logs to CSV for offline analysis
- 🌀 Add stochasticity to rewards or state transitions
- 🧠 Implement Deep Q-Network (DQN) with neural network function approximation
- 🎞️ Record agent movement into video/GIF for presentations
- 🕹️ Add keyboard-controlled manual mode for comparison

---

## 📌 References
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
- Watkins, C. J. C. H., & Dayan, P. (1992). *Q-learning*. Machine Learning, 8(3-4), 279–292.
- OpenAI Gym Documentation: https://www.gymlibrary.dev/
- Pygame Docs: https://www.pygame.org/docs/

---

## 🛡️ License
This project is licensed under the **Ulster University License**.

---

## 📬 Contact
- Your Name: [Sulaimon Olasupo](mailto:olasupooladotun6@gmail.com)
- Repository URL: [https://github.com/Ola-Domain/grid_world.git](https://github.com/Ola-Domain/grid_world.git)

---

Happy Training! 🧠🕹️🎯

