# Q-learning agent for a 5x5 Grid World (Grid Game Format)
import numpy as np
import random
import matplotlib.pyplot as plt
import time
import os

# Grid world setup
GRID_SIZE = 5
START = (1, 0)
GOAL = (4, 4)
JUMP_FROM = (1, 3)
JUMP_TO = (3, 3)
OBSTACLES = [(1, 1), (1, 2), (1, 3), (2, 1)]

ACTIONS = ['N', 'S', 'E', 'W']
action_map = {
    'N': (-1, 0),
    'S': (1, 0),
    'E': (0, 1),
    'W': (0, -1)
}

# Hyperparameters
ALPHA = 1.0
GAMMA = 0.9
EPSILON = 0.1
EPISODES = 100

# Q-table initialization
Q_table = {}
for i in range(GRID_SIZE):
    for j in range(GRID_SIZE):
        Q_table[(i, j)] = {a: 0 for a in ACTIONS}


def is_valid(state):
    x, y = state
    return 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE and (x, y) not in OBSTACLES


def take_action(state, action):
    if state == JUMP_FROM:
        return JUMP_TO, 5, False

    dx, dy = action_map[action]
    next_state = (state[0] + dx, state[1] + dy)

    if not is_valid(next_state):
        return state, -1, False

    if next_state == GOAL:
        return next_state, 10, True

    return next_state, -1, False


def choose_action(state):
    if random.uniform(0, 1) < EPSILON:
        return random.choice(ACTIONS)
    else:
        return max(Q_table[state], key=Q_table[state].get)


def print_grid(state):
    os.system('cls' if os.name == 'nt' else 'clear')
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if (i, j) == state:
                print('A', end=' ')
            elif (i, j) in OBSTACLES:
                print('X', end=' ')
            elif (i, j) == GOAL:
                print('G', end=' ')
            elif (i, j) == JUMP_FROM:
                print('J', end=' ')
            else:
                print('.', end=' ')
        print()
    print()


# Training the agent
rewards_per_episode = []

for episode in range(EPISODES):
    state = START
    total_reward = 0
    done = False

    while not done:
        print_grid(state)
        action = choose_action(state)
        next_state, reward, done = take_action(state, action)

        old_value = Q_table[state][action]
        next_max = max(Q_table[next_state].values())

        Q_table[state][action] = old_value + ALPHA * (reward + GAMMA * next_max - old_value)

        state = next_state
        total_reward += reward

        time.sleep(0.1)

    rewards_per_episode.append(total_reward)


# Visualize final value table
value_grid = np.zeros((GRID_SIZE, GRID_SIZE))
for i in range(GRID_SIZE):
    for j in range(GRID_SIZE):
        value_grid[i, j] = max(Q_table[(i, j)].values())

fig, ax = plt.subplots()
for i in range(GRID_SIZE):
    for j in range(GRID_SIZE):
        if (i, j) in OBSTACLES:
            ax.text(j, i, 'X', ha='center', va='center', color='black')
        elif (i, j) == GOAL:
            ax.text(j, i, 'G', ha='center', va='center', color='blue')
        elif (i, j) == JUMP_FROM:
            ax.text(j, i, 'J', ha='center', va='center', color='green')
        else:
            ax.text(j, i, str(round(value_grid[i, j], 2)), ha='center', va='center')

ax.set_xticks(np.arange(GRID_SIZE))
ax.set_yticks(np.arange(GRID_SIZE))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_title('State Value Table after Training')
plt.grid(True)
plt.show()