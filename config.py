from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List


@dataclass
class GridConfig:
    """Configuration for the GridWorld environment.

    Layout string chars:
        '.' = empty     '#' = wall     '~' = slippery
        'G' = goal      'X' = trap     'S' = start (also an empty cell)
    """
    rows: int = 4
    cols: int = 4

    # 2D list of strings, one string per row. If None, all-empty grid.
    layout: Optional[List[str]] = None

    # Fallback start position (row, col) when 'S' is absent from layout
    start: Tuple[int, int] = (0, 0)

    # Per-step reward applied when landing on a non-terminal cell
    step_reward: float = 0.0
    goal_reward: float = 1.0
    trap_reward: float = -1.0

    # Cell-specific reward overrides: {(row, col): reward}
    cell_rewards: Dict[Tuple[int, int], float] = field(default_factory=dict)

    # Probability of sliding 90° instead of the intended direction
    default_slip_prob: float = 0.0    # normal cells
    slippery_slip_prob: float = 0.3   # '~' cells

    gamma: float = 0.95               # discount factor


@dataclass
class AlgorithmConfig:
    """Hyperparameters shared across algorithms."""

    # --- Policy Iteration ---
    pi_threshold: float = 1e-6
    pi_max_iter: int = 1000

    # --- Q-Learning / SARSA ---
    alpha: float = 0.1
    epsilon: float = 1.0
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.01
    n_episodes: int = 5000
    max_steps: int = 200

    # --- DQN ---
    dqn_hidden: List[int] = field(default_factory=lambda: [64, 64])
    dqn_lr: float = 1e-3
    dqn_batch_size: int = 64
    dqn_buffer_size: int = 10_000
    dqn_target_update: int = 100      # steps between target-net syncs
    dqn_n_episodes: int = 2000
    dqn_epsilon: float = 1.0
    dqn_epsilon_decay: float = 0.997
    dqn_epsilon_min: float = 0.01
    # --- Exploration starts (randomize episode start state)
    exploring_starts: bool = True
