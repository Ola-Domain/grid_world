from config import GRID_SIZE, OBSTACLES, SPECIAL_JUMP, START_POS, TERMINAL_STATE, ACTIONS

class GridWorld:
    def __init__(self):
        self.reset()

    def reset(self):
        """Reset the environment to the start position."""
        self.agent_pos = START_POS
        return self.agent_pos

    def step(self, action):
        """Apply action and return next_state, reward, done."""
        dx, dy = ACTIONS[action]
        x, y = self.agent_pos
        new_x, new_y = x + dx, y + dy

        # Check valid move (stay in bounds and not hitting obstacles)
        if (1 <= new_x <= GRID_SIZE and 1 <= new_y <= GRID_SIZE and (new_x, new_y) not in OBSTACLES):
            self.agent_pos = (new_x, new_y)

        # Handle jump tile (check before goal)
        if self.agent_pos in SPECIAL_JUMP:
            self.agent_pos = SPECIAL_JUMP[self.agent_pos]

            # If jump lands directly on goal
            if self.agent_pos == TERMINAL_STATE:
                return self.agent_pos, 10, True

            return self.agent_pos, 5, False

        # Goal reached
        if self.agent_pos == TERMINAL_STATE:
            return self.agent_pos, 10, True

        # Default step penalty
        return self.agent_pos, -1, False
