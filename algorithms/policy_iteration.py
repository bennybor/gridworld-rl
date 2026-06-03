from __future__ import annotations
import numpy as np

from environment.gridworld import GridWorld
from config import AlgorithmConfig


def policy_iteration(
    env: GridWorld, cfg: AlgorithmConfig
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, float]]]:
    """Policy Iteration (DP, requires full model).

    Returns
    -------
    policy : (n_states,) int array — greedy action per state
    V      : (n_states,) float array — state values
        (expected returns; entering a goal gives its reward immediately)
    history: list of (sweep, max_delta_V) for convergence tracking
    """
    n     = env.n_states
    gamma = env.gamma

    V      = np.zeros(n, dtype=np.float64)
    policy = np.zeros(n, dtype=np.int64)
    history: list[tuple[int, float]] = []
    sweep = 0

    while True:
        # ── Policy Evaluation ─────────────────────────────────────────
        for _ in range(cfg.pi_max_iter):
            delta = 0.0
            for s in range(n):
                if env.is_wall(s) or env.is_terminal(s):
                    continue
                v = sum(
                    p * (r + gamma * V[ns])
                    for p, ns, r, _ in env.get_transitions(s, policy[s])
                )
                delta  = max(delta, abs(v - V[s]))
                V[s]   = v
            if delta < cfg.pi_threshold:
                break

        # ── Policy Improvement ────────────────────────────────────────
        policy_stable = True
        for s in range(n):
            if env.is_wall(s) or env.is_terminal(s):
                continue
            old = policy[s]
            q_vals = [
                sum(p * (r + gamma * V[ns]) for p, ns, r, _ in env.get_transitions(s, a))
                for a in env.ACTIONS
            ]
            policy[s] = int(np.argmax(q_vals))
            if policy[s] != old:
                policy_stable = False

        sweep += 1
        history.append((sweep, delta))

        if policy_stable:
            break

    return policy, V, history
