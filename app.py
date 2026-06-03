"""GridWorld RL Trainer — Streamlit browser app
Run from d:\\gridworld_rl:  streamlit run app.py
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from config import GridConfig, AlgorithmConfig
from environment.gridworld import GridWorld
from algorithms.policy_iteration import policy_iteration
from algorithms.q_learning import q_learning
from algorithms.sarsa import sarsa
from algorithms.dqn import dqn
from visualization.renderer import render_grid

# ── Cell palette ──────────────────────────────────────────────────────
CELLS = [
    ("Empty",    ".",  "#E8E8E8", "#333333", " · "),
    ("Wall",     "#",  "#2C2C2C", "#FFFFFF", " ■ "),
    ("Slippery", "~",  "#A8D8EA", "#1a5276", " ≈ "),
    ("Goal",     "G",  "#52B788", "#FFFFFF", " G "),
    ("Trap",     "X",  "#E63946", "#FFFFFF", " X "),
    ("Start",    "S",  "#FFD166", "#333333", " S "),
]
E, WL, SL, GO, TR, ST = 0, 1, 2, 3, 4, 5
CELL_CHARS = [".", "#", "~", "G", "X", "S"]


# ── Session state helpers ─────────────────────────────────────────────

def _init():
    if "grid" not in st.session_state:
        st.session_state.grid = [[E] * 5 for _ in range(5)]
        st.session_state.grid[0][0] = ST
    if "rows" not in st.session_state:
        st.session_state.rows = 5
    if "cols" not in st.session_state:
        st.session_state.cols = 5
    if "tool" not in st.session_state:
        st.session_state.tool = ST
    if "results" not in st.session_state:
        st.session_state.results = None


def _set_cell(r, c):
    t = st.session_state.tool
    if t == ST:
        for rr in range(st.session_state.rows):
            for cc in range(st.session_state.cols):
                if st.session_state.grid[rr][cc] == ST:
                    st.session_state.grid[rr][cc] = E
    st.session_state.grid[r][c] = t


def _resize(new_rows, new_cols):
    old = st.session_state.grid
    old_rows, old_cols = st.session_state.rows, st.session_state.cols
    new_grid = [[E] * new_cols for _ in range(new_rows)]
    has_start = False
    for r in range(min(new_rows, old_rows)):
        for c in range(min(new_cols, old_cols)):
            new_grid[r][c] = old[r][c]
            if old[r][c] == ST:
                has_start = True
    if not has_start:
        new_grid[0][0] = ST
    st.session_state.grid = new_grid
    st.session_state.rows = new_rows
    st.session_state.cols = new_cols


def _get_layout():
    return [
        "".join(CELL_CHARS[st.session_state.grid[r][c]]
                for c in range(st.session_state.cols))
        for r in range(st.session_state.rows)
    ]


def _validate():
    flat = [st.session_state.grid[r][c]
            for r in range(st.session_state.rows)
            for c in range(st.session_state.cols)]
    if ST not in flat:
        return "Grid needs a Start cell (S)."
    if GO not in flat:
        return "Grid needs at least one Goal cell (G)."
    return None


# ── Page ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="GridWorld RL Trainer", layout="wide")
st.title("GridWorld RL Trainer")
_init()

left, right = st.columns([5, 6], gap="large")

# ── Left: grid editor ─────────────────────────────────────────────────
with left:
    st.subheader("Grid Editor")

    rc1, rc2, rc3, rc4 = st.columns([2, 2, 2, 2])
    with rc1:
        nr = st.number_input("Rows", 2, 14, st.session_state.rows, key="nr_input")
    with rc2:
        nc = st.number_input("Cols", 2, 14, st.session_state.cols, key="nc_input")
    with rc3:
        st.write("")
        if st.button("Resize", use_container_width=True):
            _resize(int(nr), int(nc))
            st.rerun()
    with rc4:
        st.write("")
        if st.button("Clear", use_container_width=True):
            _resize(st.session_state.rows, st.session_state.cols)
            st.rerun()

    st.write("**Paint Tool**")
    tool_cols = st.columns(len(CELLS))
    for i, (name, ch, bg, fg, lbl) in enumerate(CELLS):
        with tool_cols[i]:
            if st.button(lbl, key=f"tool_{i}",
                         type="primary" if st.session_state.tool == i else "secondary",
                         help=name, use_container_width=True):
                st.session_state.tool = i
                st.rerun()

    st.write("**Grid**")
    for r in range(st.session_state.rows):
        row_cols = st.columns(st.session_state.cols)
        for c in range(st.session_state.cols):
            with row_cols[c]:
                t = st.session_state.grid[r][c]
                if st.button(CELLS[t][4], key=f"cell_{r}_{c}",
                             help=f"({r},{c}) {CELLS[t][0]}",
                             use_container_width=True):
                    _set_cell(r, c)
                    st.rerun()

# ── Right: settings + results ─────────────────────────────────────────
with right:
    st.subheader("Settings")

    s1, s2 = st.columns(2)
    with s1:
        step_rew = st.number_input("Step reward", value=-0.04, format="%.3f")
        goal_rew = st.number_input("Goal reward", value=1.0,   format="%.2f")
        trap_rew = st.number_input("Trap reward", value=-1.0,  format="%.2f")
    with s2:
        slip  = st.slider("Slip prob", 0.0, 1.0, 0.3, 0.01)
        gamma = st.slider("Gamma",     0.5, 1.0, 0.95, 0.01)

    algo = st.selectbox("Algorithm",
                        ["Policy Iteration", "Q-Learning", "SARSA", "DQN"])

    if algo != "Policy Iteration":
        ep1, ep2 = st.columns(2)
        with ep1:
            episodes = st.number_input("Episodes", 100, 50000, 3000)
        with ep2:
            alpha = st.number_input("Learning rate (α)", 0.001, 1.0, 0.1, format="%.3f")
    else:
        episodes, alpha = 0, 0.1

    if st.button("▶  Train Agent", type="primary", use_container_width=True):
        err = _validate()
        if err:
            st.error(err)
        else:
            grid_cfg = GridConfig(
                rows=st.session_state.rows,
                cols=st.session_state.cols,
                layout=_get_layout(),
                step_reward=step_rew,
                goal_reward=goal_rew,
                trap_reward=trap_rew,
                slippery_slip_prob=slip,
                gamma=gamma,
            )
            algo_cfg = AlgorithmConfig(
                alpha=alpha,
                n_episodes=int(episodes),
                dqn_n_episodes=int(episodes),
                max_steps=300,
                epsilon=1.0,
                epsilon_decay=0.998,
                epsilon_min=0.01,
            )
            with st.spinner(f"Training with {algo}…"):
                try:
                    env = GridWorld(grid_cfg)
                    if algo == "Policy Iteration":
                        pol, vals, _ = policy_iteration(env, algo_cfg)
                        st.session_state.results = (algo, env, pol, vals, None, None)
                    elif algo == "Q-Learning":
                        pol, Q, rews, lens = q_learning(env, algo_cfg)
                        st.session_state.results = (algo, env, pol, Q.max(axis=1), rews, lens)
                    elif algo == "SARSA":
                        pol, Q, rews, lens = sarsa(env, algo_cfg)
                        st.session_state.results = (algo, env, pol, Q.max(axis=1), rews, lens)
                    elif algo == "DQN":
                        pol, Q, rews, lens = dqn(env, algo_cfg)
                        st.session_state.results = (algo, env, pol, Q.max(axis=1), rews, lens)
                    st.success(f"{algo} finished training.")
                except Exception as ex:
                    st.error(f"Training error: {ex}")

    if st.session_state.results is not None:
        algo_name, env, pol, vals, rews, lens = st.session_state.results
        st.subheader(f"Results — {algo_name}")

        has_curves = rews is not None
        if has_curves:
            fig, (ax_g, ax_r, ax_l) = plt.subplots(1, 3, figsize=(13, 3.8))
        else:
            fig, ax_g = plt.subplots(1, 1, figsize=(5, 4))

        render_grid(env, values=vals, policy=pol,
                    title=f"{algo_name} — Policy & Values", ax=ax_g)

        if has_curves:
            n = len(rews)
            w = max(1, min(50, n // 10))
            sm = lambda d: np.convolve(d, np.ones(w) / w, mode="valid")

            ax_r.plot(rews, alpha=0.2, color="steelblue")
            if n >= w:
                ax_r.plot(range(w - 1, n), sm(rews), color="steelblue", lw=2)
            ax_r.set_xlabel("Episode"); ax_r.set_ylabel("Total Reward")
            ax_r.set_title("Reward per Episode"); ax_r.grid(True, alpha=0.3)

            ax_l.plot(lens, alpha=0.2, color="salmon")
            if n >= w:
                ax_l.plot(range(w - 1, n), sm(lens), color="salmon", lw=2)
            ax_l.set_xlabel("Episode"); ax_l.set_ylabel("Steps")
            ax_l.set_title("Episode Length"); ax_l.grid(True, alpha=0.3)

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
