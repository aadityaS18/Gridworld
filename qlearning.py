"""
Q-learning on a 6x6 grid world - 
Grid: x = 0..5, y = 0..5. Agent starts at (0, 2) (black dot), goal at (5, 5) (red X).
Reward: -1 for every step, +100 for reaching the goal.
An episode stops at the goal or after 200 steps (so the worst score is -200).
"""
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# ---------------- Settings (change these to experiment) ----------------
GRID_SIZE = 6
START = (0, 2)
GOAL = (5, 5)
STEP_REWARD = -1
GOAL_REWARD = 100
MAX_STEPS = 200

EPISODES = 8000
ALPHA = 0.1             # learning rate
GAMMA = 0.95            # discount
EPSILON_START = 1.0     # explore 100% at the start ...
EPSILON_MIN = 0.01      # ... down to 1% at the end
EPSILON_DECAY = 0.9995  # epsilon is multiplied by this after every episode

ACTIONS = ["up", "down", "left", "right"]
MOVES = [(0, 1), (0, -1), (-1, 0), (1, 0)]   # (dx, dy) for each action


# =====================================================================
# 1. THE ENVIRONMENT
# =====================================================================
class GridWorld:
    def __init__(self, size=GRID_SIZE, start=START, goal=GOAL):
        self.size, self.start, self.goal = size, start, goal

    def reset(self):
        self.pos = self.start
        return self.pos

    def step(self, action):
        """Move one cell. Return (next_position, reward, done)."""
        dx, dy = MOVES[action]
        x = min(max(self.pos[0] + dx, 0), self.size - 1)   # walls: stay inside the grid
        y = min(max(self.pos[1] + dy, 0), self.size - 1)
        self.pos = (x, y)
        if self.pos == self.goal:
            return self.pos, GOAL_REWARD, True
        return self.pos, STEP_REWARD, False


# =====================================================================
# 2. THE AGENT: epsilon-greedy + Q-learning update (slide 12)
# =====================================================================
def choose_action(Q, pos, epsilon):
    if random.random() < epsilon:
        return random.randint(0, 3)                  # explore
    x, y = pos
    return int(np.argmax(Q[x, y]))                   # exploit


def train(env):
    Q = np.zeros((GRID_SIZE, GRID_SIZE, 4))   # one Q-value per (x, y, action)
    rewards = []
    paths = {}                                # agent's path in some episodes, for the animation
    epsilon = EPSILON_START

    for episode in range(EPISODES):
        pos = env.reset()
        total, path = 0, [pos]

        for step in range(MAX_STEPS):
            action = choose_action(Q, pos, epsilon)
            next_pos, reward, done = env.step(action)

            # Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))
            x, y = pos
            nx, ny = next_pos
            target = reward if done else reward + GAMMA * np.max(Q[nx, ny])
            Q[x, y, action] += ALPHA * (target - Q[x, y, action])

            pos, total = next_pos, total + reward
            path.append(pos)
            if done:
                break

        rewards.append(total)
        if episode in (0, 100, 1000, EPISODES - 1):
            paths[episode] = path
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

    return Q, rewards, paths


# =====================================================================
# 3. PLOTS (slides 13 and 14)
# =====================================================================
def plot_rewards(rewards, window=100):
    """Slide 14 left: total reward per episode (blue) + moving average (orange)."""
    smooth = np.convolve(rewards, np.ones(window) / window, mode="valid")
    plt.figure(figsize=(8, 4.5))
    plt.plot(rewards, color="tab:blue")
    plt.plot(range(window - 1, len(rewards)), smooth, color="tab:orange")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.tight_layout()
    plt.savefig("reward_6x6.png")


def plot_policy(Q):
    """Slide 14 right: best action in each cell as an arrow."""
    arrow = {0: (0, 0.25), 1: (0, -0.25), 2: (-0.25, 0), 3: (0.25, 0)}
    plt.figure(figsize=(5, 5))
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            dx, dy = arrow[int(np.argmax(Q[x, y]))]
            plt.arrow(x, y, dx, dy, head_width=0.08, color="black")
    plt.xlim(-0.3, GRID_SIZE - 0.5)
    plt.ylim(-0.3, GRID_SIZE - 0.5)
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.tight_layout()
    plt.savefig("policy_6x6.png")


def animate(path, episode):
    """Slide 13: the agent (black dot) moving towards the goal (red X)."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(-0.3, GRID_SIZE - 0.7)
    ax.set_ylim(-0.3, GRID_SIZE - 0.7)
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.plot(*GOAL, "rX", markersize=20)
    dot, = ax.plot([], [], "ko", markersize=18)
    ax.text(2.5, 4.5, f"Episode {episode + 1}", ha="center", bbox=dict(fc="white"))
    path = path[:150]   # keep long random walks short

    def update(i):
        dot.set_data([path[i][0]], [path[i][1]])
        return dot,

    anim = FuncAnimation(fig, update, frames=len(path), interval=150)
    anim.save(f"episode_{episode + 1}.gif", writer=PillowWriter(fps=6))
    plt.close(fig)


# =====================================================================
# 4. RUN
# =====================================================================
if __name__ == "__main__":
    env = GridWorld()
    Q, rewards, paths = train(env)

    print("Average reward, first 100 episodes:", round(np.mean(rewards[:100]), 1))
    print("Average reward, last 100 episodes: ", round(np.mean(rewards[-100:]), 1))
    print("Best possible reward: 93 (8 steps: 7 x -1, then +100)")

    for episode, path in paths.items():
        animate(path, episode)                 # saves episode_1.gif, episode_101.gif, ...
    plot_rewards(rewards)
    plot_policy(Q)
    plt.show()