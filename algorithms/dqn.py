from __future__ import annotations
import random
from collections import deque
from typing import List

import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Suppress the NumPy 2.x / PyTorch 1.x ABI mismatch warning.
# Our code never uses the C-level numpy↔torch bridge, so the warning is harmless.
warnings.filterwarnings("ignore", message="Failed to initialize NumPy")

from environment.gridworld import GridWorld
from config import AlgorithmConfig

# Avoid the NumPy ↔ PyTorch C-bridge entirely by routing all data
# through Python lists.  This works regardless of NumPy version.


# ── Replay buffer (stores Python lists, not numpy arrays) ─────────────

class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buf: deque = deque(maxlen=capacity)

    def push(
        self,
        state: list,
        action: int,
        reward: float,
        next_state: list,
        done: float,
    ) -> None:
        self.buf.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        states, actions, rewards, nexts, dones = zip(*batch)
        return (
            list(states),
            list(actions),
            list(rewards),
            list(nexts),
            list(dones),
        )

    def __len__(self) -> int:
        return len(self.buf)


# ── Q-Network ─────────────────────────────────────────────────────────

class QNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden: List[int]) -> None:
        super().__init__()
        dims   = [input_dim, *hidden, output_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── DQN training loop ─────────────────────────────────────────────────

def dqn(
    env: GridWorld, cfg: AlgorithmConfig
) -> tuple[np.ndarray, np.ndarray, list[float], list[int]]:
    """Deep Q-Network with experience replay and target network.

    Returns
    -------
    policy          : (n_states,) greedy policy
    Q_table         : (n_states, n_actions) Q-values extracted from the network
    episode_rewards : total undiscounted reward per episode
    episode_lengths : number of steps per episode
    """
    device = torch.device("cpu")  # safe default; avoids CUDA/NumPy bridge issues

    q_net      = QNetwork(env.n_states, env.n_actions, cfg.dqn_hidden).to(device)
    target_net = QNetwork(env.n_states, env.n_actions, cfg.dqn_hidden).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=cfg.dqn_lr)
    buffer    = ReplayBuffer(cfg.dqn_buffer_size)
    epsilon   = cfg.dqn_epsilon
    visits    = np.zeros(env.n_states, dtype=np.int64)

    episode_rewards: list[float] = []
    episode_lengths: list[int]   = []
    total_steps = 0

    def state_vec(s: int) -> list:
        v = [0.0] * env.n_states
        v[s] = 1.0
        return v

    for _ in range(cfg.dqn_n_episodes):
        if getattr(cfg, "exploring_starts", False):
            valid_states = [s for s in range(env.n_states) if not env.is_wall(s) and not env.is_terminal(s)]
            state = int(random.choice(valid_states))
            env.agent_pos = divmod(state, env.cols)
            env.done = False
        else:
            state     = env.reset()
        sv        = state_vec(state)
        total_rew = 0.0
        steps     = 0

        for _ in range(cfg.max_steps):
            visits[state] += 1
            # Epsilon-greedy
            if random.random() < epsilon:
                action = random.randrange(env.n_actions)
            else:
                with torch.no_grad():
                    s_t    = torch.tensor([sv], dtype=torch.float32)
                    action = int(q_net(s_t).argmax(dim=1).item())

            next_state, reward, done = env.step(action)
            # count arrival to next state
            visits[next_state] += 1
            next_sv    = state_vec(next_state)
            total_rew += reward
            steps     += 1
            total_steps += 1

            buffer.push(sv, action, reward, next_sv, 1.0 if done else 0.0)
            state = next_state
            sv    = next_sv

            # Training step
            if len(buffer) >= cfg.dqn_batch_size:
                s_b, a_b, r_b, ns_b, d_b = buffer.sample(cfg.dqn_batch_size)

                s_t  = torch.tensor(s_b,  dtype=torch.float32)
                a_t  = torch.tensor(a_b,  dtype=torch.long)
                r_t  = torch.tensor(r_b,  dtype=torch.float32)
                ns_t = torch.tensor(ns_b, dtype=torch.float32)
                d_t  = torch.tensor(d_b,  dtype=torch.float32)

                current_q = q_net(s_t).gather(1, a_t.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    max_next_q = target_net(ns_t).max(dim=1)[0]
                    target_q   = r_t + env.gamma * max_next_q * (1.0 - d_t)

                loss = nn.functional.mse_loss(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if total_steps % cfg.dqn_target_update == 0:
                target_net.load_state_dict(q_net.state_dict())

            if done:
                break

        epsilon = max(cfg.dqn_epsilon_min, epsilon * cfg.dqn_epsilon_decay)
        episode_rewards.append(total_rew)
        episode_lengths.append(steps)

    # Extract Q-table via Python lists (no NumPy bridge)
    all_states = [[1.0 if i == s else 0.0 for i in range(env.n_states)]
                  for s in range(env.n_states)]
    with torch.no_grad():
        Q_list  = q_net(torch.tensor(all_states, dtype=torch.float32)).tolist()
    Q_table = np.array(Q_list, dtype=np.float64)

    policy = np.argmax(Q_table, axis=1)
    return policy, Q_table, episode_rewards, episode_lengths, visits
