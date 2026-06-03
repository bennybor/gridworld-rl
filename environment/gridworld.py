from __future__ import annotations
from typing import List, Tuple, Dict
import numpy as np

from config import GridConfig


class GridWorld:
    """Configurable GridWorld environment for RL experiments."""

    # Cell types
    EMPTY    = 0
    WALL     = 1
    SLIPPERY = 2
    GOAL     = 3
    TRAP     = 4

    # Actions: UP RIGHT DOWN LEFT
    UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
    ACTIONS      = [0, 1, 2, 3]
    ACTION_NAMES = ["UP", "RIGHT", "DOWN", "LEFT"]

    # (delta_row, delta_col) per action
    _DELTAS = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    _CHAR_MAP: Dict[str, int] = {
        ".": EMPTY, "#": WALL, "~": SLIPPERY,
        "G": GOAL,  "X": TRAP, "S": EMPTY,
    }

    def __init__(self, config: GridConfig) -> None:
        self.config = config
        self.rows   = config.rows
        self.cols   = config.cols
        self.gamma  = config.gamma
        self._parse_layout()
        self._build_transition_table()
        self.reset()

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _parse_layout(self) -> None:
        self.grid      = np.zeros((self.rows, self.cols), dtype=np.int8)
        self.start_pos = self.config.start

        if self.config.layout is None:
            return

        if len(self.config.layout) != self.rows:
            raise ValueError(
                f"Layout has {len(self.config.layout)} rows; config expects {self.rows}."
            )
        for r, row_str in enumerate(self.config.layout):
            if len(row_str) != self.cols:
                raise ValueError(
                    f"Layout row {r} has {len(row_str)} cols; config expects {self.cols}."
                )
            for c, ch in enumerate(row_str):
                if ch not in self._CHAR_MAP:
                    raise ValueError(f"Unknown cell char '{ch}' at ({r},{c}).")
                self.grid[r, c] = self._CHAR_MAP[ch]
                if ch == "S":
                    self.start_pos = (r, c)

    def _build_transition_table(self) -> None:
        """Pre-compute (prob, next_state, reward, done) lists for every (s, a)."""
        self._transitions: List[List[List[Tuple]]] = [
            [self._compute_transitions(s, a) for a in self.ACTIONS]
            for s in range(self.n_states)
        ]

    def _compute_transitions(self, state: int, action: int) -> List[Tuple]:
        r, c  = divmod(state, self.cols)
        cell  = self.grid[r, c]

        if cell in (self.GOAL, self.TRAP):
            return [(1.0, state, 0.0, True)]

        slip = (
            self.config.slippery_slip_prob
            if cell == self.SLIPPERY
            else self.config.default_slip_prob
        )

        action_probs = {
            action:             1.0 - slip,
            (action + 1) % 4:  slip / 2,
            (action - 1) % 4:  slip / 2,
        }

        # Aggregate outcomes by next-state index
        outcome: Dict[int, Tuple] = {}
        for a, p in action_probs.items():
            if p == 0.0:
                continue
            nr, nc = self._step_pos(r, c, a)
            ns     = nr * self.cols + nc
            reward = self._cell_reward(nr, nc)
            done   = bool(self.grid[nr, nc] in (self.GOAL, self.TRAP))
            if ns in outcome:
                outcome[ns] = (outcome[ns][0] + p, reward, done)
            else:
                outcome[ns] = (p, reward, done)

        return [(p, ns, rw, dn) for ns, (p, rw, dn) in outcome.items()]

    def _step_pos(self, r: int, c: int, action: int) -> Tuple[int, int]:
        dr, dc = self._DELTAS[action]
        nr, nc = r + dr, c + dc
        if 0 <= nr < self.rows and 0 <= nc < self.cols and self.grid[nr, nc] != self.WALL:
            return nr, nc
        return r, c  # bounce

    def _cell_reward(self, r: int, c: int) -> float:
        if (r, c) in self.config.cell_rewards:
            return self.config.cell_rewards[(r, c)]
        cell = self.grid[r, c]
        if cell == self.GOAL:
            return self.config.goal_reward
        if cell == self.TRAP:
            return self.config.trap_reward
        return self.config.step_reward

    # ------------------------------------------------------------------
    # Gym-like interface
    # ------------------------------------------------------------------

    @property
    def n_states(self) -> int:
        return self.rows * self.cols

    @property
    def n_actions(self) -> int:
        return 4

    def reset(self) -> int:
        self.agent_pos = self.start_pos
        self.done      = False
        return self._pos_to_state(*self.agent_pos)

    def step(self, action: int) -> Tuple[int, float, bool]:
        if self.done:
            raise RuntimeError("Episode finished — call reset() first.")
        state       = self._pos_to_state(*self.agent_pos)
        transitions = self._transitions[state][action]
        probs       = [t[0] for t in transitions]
        idx         = np.random.choice(len(transitions), p=probs)
        _, next_state, reward, done = transitions[idx]
        self.agent_pos = divmod(next_state, self.cols)
        self.done      = done
        return next_state, reward, done

    def get_transitions(self, state: int, action: int) -> List[Tuple]:
        """Return [(prob, next_state, reward, done)] — for model-based methods."""
        return self._transitions[state][action]

    def is_terminal(self, state: int) -> bool:
        r, c = divmod(state, self.cols)
        return bool(self.grid[r, c] in (self.GOAL, self.TRAP))

    def is_wall(self, state: int) -> bool:
        r, c = divmod(state, self.cols)
        return bool(self.grid[r, c] == self.WALL)

    def state_to_pos(self, state: int) -> Tuple[int, int]:
        return divmod(state, self.cols)

    def state_one_hot(self, state: int) -> np.ndarray:
        v = np.zeros(self.n_states, dtype=np.float32)
        v[state] = 1.0
        return v

    def _pos_to_state(self, r: int, c: int) -> int:
        return r * self.cols + c
