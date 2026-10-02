import random

# Cell names from the slides, row by row (5 rows x 4 columns)
NAMES = "ABCDEFGHIJKLMNOPQRST"
ACTIONS = ["up", "down", "left", "right"]


class GridWorld:
    def __init__(self):
        self.rows = 5
        self.cols = 4
        self.start = (1, 1)   # cell F (row 1, column 1)
        self.goal = (3, 2)    # cell O (row 3, column 2)

    def reset(self):
        """Put the agent back at the start. Return the starting state."""
        self.pos = self.start
        return self.to_state(self.pos)

    def to_state(self, pos):
        """Turn (row, col) into one number 0-19. A=0, F=5, O=14."""
        row, col = pos
        return row * self.cols + col

    def step(self, action):
        """Move the agent. Return (next_state, reward, done)."""
        row, col = self.pos
        if action == 0:      # up
            row -= 1
        elif action == 1:    # down
            row += 1
        elif action == 2:    # left
            col -= 1
        elif action == 3:    # right
            col += 1

        # Hitting a wall: stay inside the grid
        row = max(0, min(row, self.rows - 1))
        col = max(0, min(col, self.cols - 1))
        self.pos = (row, col)

        if self.pos == self.goal:
            return self.to_state(self.pos), 10, True   # reached O: +10, episode over
        return self.to_state(self.pos), -1, False      # every other step: -1


# ---- Random agent (slide 5): no learning, just wandering ----
env = GridWorld()
state = env.reset()
total_reward = 0
steps = 0
done = False

while not done:
    action = random.randint(0, 3)
    next_state, reward, done = env.step(action)
    print(f"{NAMES[state]} --{ACTIONS[action]}--> {NAMES[next_state]}   reward {reward}")
    state = next_state
    total_reward += reward
    steps += 1

print(f"\nReached the goal in {steps} steps, total reward {total_reward}")
            
