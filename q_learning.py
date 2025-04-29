import random

class QLearningAgent:
    def __init__(self, alpha=0.8, gamma=0.9, epsilon=1.0):
        self.q_table = {}  # {(row, col): [Q1, Q2, Q3, Q4]}
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate

    def act(self, state):
        """Returns an action based on epsilon-greedy strategy."""
        if state not in self.q_table:
            self.q_table[state] = [0.0] * 4  # Initialise Q-values for this state

        if random.random() < self.epsilon:
            return random.randint(1, 4)  # Explore: random action
        else:
            return int(self.q_table[state].index(max(self.q_table[state]))) + 1  # Exploit: best known action

    def learn(self, state, action, reward, next_state):
        """Update Q-table using TD learning rule."""
        if state not in self.q_table:
            self.q_table[state] = [0.0] * 4
        if next_state not in self.q_table:
            self.q_table[next_state] = [0.0] * 4

        a = action - 1  # adjust action index (1-indexed to 0-indexed)
        max_q = max(self.q_table[next_state])
        self.q_table[state][a] += self.alpha * (reward + self.gamma * max_q - self.q_table[state][a])

    def decay_epsilon(self, min_epsilon=0.1, decay_rate=0.99):
        """Reduce epsilon gradually each episode."""
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)
