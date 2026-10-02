"""
Grid world + tabular Q-learning (from Prof. Cahill's RL slides).

Grid: 5 rows x 4 columns, cells A-T. Start at F, goal at O.
Reward: -1 for every step, +10 for reaching O.
"""
import random
import numpy as np
import matplotlib.pyplot as plt

NAMES = "ABCDEFGHIJKLMNOPQRST"          # cell names, row by row
ACTIONS = ["up", "down", "left", "right"]


# =====================================================================
# 1. THE ENVIRONMENT (the grid)
# =====================================================================
class GridWorld:
    def __init__(self):
        self.rows = 5
        self.cols = 4
        self.start = (1, 1)   # cell F
        self.goal = (3, 2)    # cell O

    def reset(self):
        """Start a new episode: put the agent back at F."""
        self.pos = self.start
        return self.to_state(self.pos)

    def to_state(self, pos):
        """Turn (row, col) into one number 0-19. A=0, F=5, O=14."""
        row, col = pos
        return row * self.cols + col

    def step(self, action):
        """Move the agent one cell. Return (next_state, reward, done)."""
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
            return self.to_state(self.pos), 10, True   # reached O: +10, episode ends
        return self.to_state(self.pos), -1, False      # any other step: -1


# =====================================================================
# 2. THE AGENT (Q-learning)
# =====================================================================
def choose_action(Q, state, epsilon):
    """Epsilon-greedy (slide 12): usually the best action, sometimes random."""
    if random.random() < epsilon:
        return random.randint(0, 3)          # explore: random move
    return int(np.argmax(Q[state]))          # exploit: move with highest Q


def train(env, episodes=500, alpha=0.1, gamma=0.95, epsilon=0.1, max_steps=100):
    """Run Q-learning. Return the Q-table and the total reward of each episode."""
    n_states = env.rows * env.cols
    Q = np.zeros((n_states, 4))   # 20 states x 4 actions = the 80 rows on slide 8
    rewards_per_episode = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0

        for step in range(max_steps):          # cap so an episode can't run forever
            action = choose_action(Q, state, epsilon)
            next_state, reward, done = env.step(action)

            # ---- The Q-learning update (slide 12) ----
            # target = what this move was actually worth:
            #          reward now + discounted best Q from the next cell
            if done:
                target = reward                # goal reached: no future
            else:
                target = reward + gamma * np.max(Q[next_state])
            # nudge the old guess a step of size alpha towards the target
            Q[state, action] = Q[state, action] + alpha * (target - Q[state, action])

            state = next_state
            total_reward += reward
            if done:
                break

        rewards_per_episode.append(total_reward)

    return Q, rewards_per_episode


# =====================================================================
# 3. RESULTS: print and plot
# =====================================================================
def print_best_path(env, Q):
    """Follow the highest Q from the start and print the path."""
    state = env.reset()
    path = [NAMES[state]]
    for _ in range(20):
        action = int(np.argmax(Q[state]))
        state, reward, done = env.step(action)
        path.append(NAMES[state])
        if done:
            break
    print("Learned path:", " -> ".join(path))


def plot_rewards(rewards, window=20):
    """Total reward per episode, like slide 14."""
    smooth = np.convolve(rewards, np.ones(window) / window, mode="valid")
    plt.figure(figsize=(8, 4))
    plt.plot(rewards, alpha=0.4, label="reward per episode")
    plt.plot(range(window - 1, len(rewards)), smooth, label=f"{window}-episode average")
    plt.axhline(8, color="grey", linestyle="--", label="best possible (8)")
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title("Q-learning on grid world")
    plt.legend()
    plt.tight_layout()
    plt.savefig("reward.png")


def plot_policy(env, Q):
    """Draw the best action in each cell as an arrow."""
    arrows = {0: (0, 0.3), 1: (0, -0.3), 2: (-0.3, 0), 3: (0.3, 0)}  # up, down, left, right
    plt.figure(figsize=(4, 5))
    for s in range(env.rows * env.cols):
        row, col = divmod(s, env.cols)
        plt.text(col - 0.4, -row + 0.35, NAMES[s], fontsize=8, color="grey")
        if (row, col) == env.goal:
            plt.text(col, -row, "GOAL", ha="center", va="center", fontsize=10, color="green")
            continue
        dx, dy = arrows[int(np.argmax(Q[s]))]
        plt.arrow(col - dx / 2, -row - dy / 2, dx, dy, head_width=0.12, color="tab:blue")
    plt.xlim(-0.5, env.cols - 0.5)
    plt.ylim(-env.rows + 0.5, 0.5)
    plt.xticks([])
    plt.yticks([])
    plt.title("Learned policy")
    plt.tight_layout()
    plt.savefig("policy.png")


# =====================================================================
# 4. RUN EVERYTHING
# =====================================================================
if __name__ == "__main__":
    env = GridWorld()

    # Check 1: with alpha = gamma = 1 the Q-values should match slides 9-11 exactly
    Q_check, _ = train(env, episodes=500, alpha=1, gamma=1)
    print("Slide check (should be 10, 9, 8):",
          Q_check[13, 3], Q_check[9, 1], Q_check[5, 1])   # Q(N,right), Q(J,down), Q(F,down)

    # Main run with normal settings
    Q, rewards = train(env, episodes=500, alpha=0.1, gamma=0.95, epsilon=0.1)
    print("Average reward, first 20 episodes:", np.mean(rewards[:20]))
    print("Average reward, last 20 episodes: ", np.mean(rewards[-20:]))
    print_best_path(env, Q)

    plot_rewards(rewards)
    plot_policy(env, Q)
    plt.show()
    
            
