from __future__ import annotations
import numpy as np
import random

from environment.gridworld import GridWorld
from config import AlgorithmConfig


def sarsa(
    env: GridWorld, cfg: AlgorithmConfig
) -> tuple[np.ndarray, np.ndarray, list[float], list[int]]:
    """SARSA — on-policy TD control.

    Returns
    -------
    policy          : (n_states,) greedy policy derived from Q
    Q               : (n_states, n_actions) Q-table
    episode_rewards : total undiscounted reward per episode
    episode_lengths : number of steps per episode
    """
    Q       = np.zeros((env.n_states, env.n_actions), dtype=np.float64)
    epsilon = cfg.epsilon
    visits  = np.zeros(env.n_states, dtype=np.int64)

    episode_rewards: list[float] = []
    episode_lengths: list[int]   = []

    def eps_greedy(s: int) -> int:
        if np.random.random() < epsilon:
            return np.random.randint(env.n_actions)
        return int(np.argmax(Q[s]))

    for _ in range(cfg.n_episodes):
        if getattr(cfg, "exploring_starts", False):
            valid_states = [s for s in range(env.n_states) if not env.is_wall(s) and not env.is_terminal(s)]
            state = int(random.choice(valid_states))
            env.agent_pos = divmod(state, env.cols)
            env.done = False
        else:
            state        = env.reset()
        action       = eps_greedy(state)
        total_reward = 0.0
        steps        = 0

        for _ in range(cfg.max_steps):
            visits[state] += 1
            next_state, reward, done = env.step(action)
            # count arrival to next state
            visits[next_state] += 1
            next_action  = eps_greedy(next_state)
            total_reward += reward
            steps        += 1

            next_q           = 0.0 if done else Q[next_state, next_action]
            Q[state, action] += cfg.alpha * (
                reward + env.gamma * next_q - Q[state, action]
            )

            state  = next_state
            action = next_action
            if done:
                break

        epsilon = max(cfg.epsilon_min, epsilon * cfg.epsilon_decay)
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

    policy = np.argmax(Q, axis=1)
    return policy, Q, episode_rewards, episode_lengths, visits
