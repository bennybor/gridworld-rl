"""
GridWorld RL — Dog Agent Example
=================================
Run from the project root:
    python main.py

Grid legend:
    .  empty      #  wall      ~  slippery
    G  goal       X  trap      S  start
"""
import matplotlib.pyplot as plt

from config import GridConfig, AlgorithmConfig
from environment.gridworld import GridWorld
from algorithms.policy_iteration import policy_iteration
from algorithms.q_learning import q_learning
from algorithms.sarsa import sarsa
from algorithms.dqn import dqn
from visualization.renderer import render_summary, animate_episode


# ── 1. Define the grid ────────────────────────────────────────────────

LAYOUT = [
    "S....",
    ".###.",
    ".~...",
    "...X.",
    "....G",
]

grid_cfg = GridConfig(
    rows=5,
    cols=5,
    layout=LAYOUT,
    step_reward=-0.04,
    goal_reward=1.0,
    trap_reward=-1.0,
    slippery_slip_prob=0.3,   # 30 % chance of sliding sideways on '~' cells
    default_slip_prob=0.0,    # normal cells are deterministic
    gamma=0.95,
)

algo_cfg = AlgorithmConfig(
    # Shared TD params
    alpha=0.1,
    epsilon=1.0,
    epsilon_decay=0.998,
    epsilon_min=0.01,
    n_episodes=5_000,
    max_steps=200,
    # DQN extras
    dqn_n_episodes=2_000,
    dqn_hidden=[64, 64],
    dqn_target_update=100,
)

env = GridWorld(grid_cfg)


# ── 2. Solve ──────────────────────────────────────────────────────────

print("Running Policy Iteration …")
pi_policy, pi_values, pi_history = policy_iteration(env, algo_cfg)
print(f"  Converged in {len(pi_history)} policy sweeps.")

print("Running Q-Learning …")
ql_policy, ql_Q, ql_rewards, ql_lengths = q_learning(env, algo_cfg)
print(f"  Final 100-ep avg reward: {sum(ql_rewards[-100:]) / 100:.3f}")

print("Running SARSA …")
sa_policy, sa_Q, sa_rewards, sa_lengths = sarsa(env, algo_cfg)
print(f"  Final 100-ep avg reward: {sum(sa_rewards[-100:]) / 100:.3f}")

print("Running DQN …")
dqn_policy, dqn_Q, dqn_rewards, dqn_lengths = dqn(env, algo_cfg)
print(f"  Final 100-ep avg reward: {sum(dqn_rewards[-100:]) / 100:.3f}")


# ── 3. Visualise ──────────────────────────────────────────────────────

render_summary(env, pi_policy,  pi_values,          "Policy Iteration")
render_summary(env, ql_policy,  ql_Q.max(axis=1),   "Q-Learning",  ql_rewards,  ql_lengths)
render_summary(env, sa_policy,  sa_Q.max(axis=1),   "SARSA",       sa_rewards,  sa_lengths)
render_summary(env, dqn_policy, dqn_Q.max(axis=1),  "DQN",         dqn_rewards, dqn_lengths)

# Animate one episode following the Policy Iteration solution
ani, fig = animate_episode(env, pi_policy, "Policy Iteration (dog agent)")

plt.show()
