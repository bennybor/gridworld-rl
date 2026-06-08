"""GridWorld RL Trainer — Streamlit Web UI
Run:  streamlit run streamlit_app.py   (from d:\\gridworld_rl)
"""
import warnings
warnings.filterwarnings("ignore", message="Failed to initialize NumPy")

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import re
import random
import pathlib
import urllib.request
import urllib.error
import json
import torch
import torch.nn as nn
import torch.optim as optim
import time
from datetime import datetime

from config import GridConfig, AlgorithmConfig
from environment.gridworld import GridWorld
from algorithms.policy_iteration import policy_iteration
from algorithms.q_learning import q_learning
from algorithms.sarsa import sarsa
from algorithms.dqn import dqn
from algorithms.dqn import QNetwork, ReplayBuffer

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="GridWorld RL Trainer",
    page_icon="🐕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Header */
h1 { font-size: 1.8rem !important; font-weight: 700 !important;
     background: linear-gradient(135deg, #6366f1, #8b5cf6);
     -webkit-background-clip: text; -webkit-text-fill-color: transparent; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    border-right: 1px solid #334155;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

/* Input / textarea / select — white bg, dark readable text */
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea,
[data-testid="stSidebar"] select {
    color: #0f172a !important;
    background-color: #f8fafc !important;
    caret-color: #0f172a !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 6px !important;
    font-size: 0.82rem !important;
}
[data-testid="stSidebar"] input:focus,
[data-testid="stSidebar"] textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 2px rgba(99,102,241,0.15) !important;
    outline: none !important;
}
/* Selectbox dropdown text */
[data-testid="stSidebar"] [data-baseweb="select"] span { color: #0f172a !important; }
[data-testid="stSidebar"] [data-baseweb="select"] [data-baseweb="tag"] { background: #e0e7ff !important; }

/* Compact widget spacing */
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] { padding: 0 0.4rem; }
[data-testid="stSidebar"] .stSlider,
[data-testid="stSidebar"] .stNumberInput,
[data-testid="stSidebar"] .stCheckbox,
[data-testid="stSidebar"] .stSelectbox { margin-bottom: 0.15rem !important; }
[data-testid="stSidebar"] .element-container { margin-bottom: 0.1rem !important; }

/* Dividers */
[data-testid="stSidebar"] hr { border-color: #334155; margin: 0.5rem 0; }

/* Labels */
[data-testid="stSidebar"] label {
    font-size: 0.75rem !important;
    color: #94a3b8 !important;
    margin-bottom: 0.05rem !important;
    line-height: 1.3 !important;
}

/* Section headings */
[data-testid="stSidebar"] h3 {
    font-size: 0.68rem !important; font-weight: 700 !important;
    color: #818cf8 !important; letter-spacing: 0.1em;
    text-transform: uppercase; margin: 0.4rem 0 0.15rem;
}

/* Buttons */
[data-testid="stSidebar"] .stButton > button {
    background: #1e293b; border: 1px solid #334155;
    color: #e2e8f0 !important; border-radius: 8px;
    font-size: 0.78rem; padding: 0.25rem 0.5rem;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #334155; border-color: #6366f1;
}

/* Caption / help text */
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] small {
    font-size: 0.7rem !important;
    color: #64748b !important;
    line-height: 1.3 !important;
}

/* Main area cards */
[data-testid="stVerticalBlock"] > div {
    gap: 0.5rem;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 12px; padding: 1rem !important;
}
[data-testid="stMetricLabel"] { font-size: 0.75rem !important; color: #64748b; }
[data-testid="stMetricValue"] { font-size: 1.4rem !important; font-weight: 700; color: #0f172a; }
[data-testid="stMetricDelta"] { font-size: 0.75rem; }

/* Primary train button */
[data-testid="stSidebar"] .train-btn > button,
.stButton.train-btn > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-weight: 600 !important;
    font-size: 0.95rem !important; padding: 0.6rem 0 !important;
    box-shadow: 0 4px 15px rgba(99,102,241,0.35);
    transition: all 0.2s ease;
}
[data-testid="stSidebar"] .train-btn > button:hover {
    box-shadow: 0 6px 20px rgba(99,102,241,0.5) !important;
    transform: translateY(-1px);
}

/* Section titles in main area */
.section-title {
    font-size: 0.75rem; font-weight: 600; color: #6366f1;
    text-transform: uppercase; letter-spacing: 0.08em;
    margin-bottom: 0.5rem;
}

/* Info/success boxes */
.stAlert { border-radius: 10px !important; }

/* Divider */
hr { border-color: #e2e8f0; margin: 0.6rem 0; }
</style>
""", unsafe_allow_html=True)

# ── Cell definitions ──────────────────────────────────────────────────
#       name        char   fill        stroke      label   text-fg


# -- Claude model selection UI ---------------------------------------
def _get_claude_model_env():
    # Prefer explicit env var, then Streamlit secrets
    m = os.environ.get("CLAUDE_MODEL")
    if m:
        return m
    try:
        secrets = getattr(st, "secrets", None)
    except Exception:
        secrets = None
    if secrets:
        try:
            return secrets.get("CLAUDE_MODEL")
        except Exception:
            try:
                return secrets["CLAUDE_MODEL"]
            except Exception:
                return None
    return None


def _init_claude_model_selector():
    default = _get_claude_model_env() or "claude-opus-4-8"
    model_options = ["claude-opus-4-8", "claude-3", "claude-3.5", "claude-2", "claude-2.1", "claude-instant", "custom"]
    try:
        st.sidebar.markdown("**Claude settings**")
        idx = model_options.index(default) if default in model_options else len(model_options) - 1
        choice = st.sidebar.selectbox("Claude model", model_options, index=idx)
        if choice == "custom":
            custom = st.sidebar.text_input("Custom Claude model", value=(default if default not in model_options else ""))
            model = custom or default
        else:
            model = choice
        # store in session_state for use elsewhere
        st.session_state["CLAUDE_MODEL"] = model
    except Exception:
        # In non-Streamlit contexts, just set the env/default
        st.session_state.setdefault("CLAUDE_MODEL", default)


# Ensure CLAUDE_MODEL is in session state before any helper runs
if "CLAUDE_MODEL" not in st.session_state:
    st.session_state["CLAUDE_MODEL"] = _get_claude_model_env() or "claude-opus-4-8"

# ── Apply pending parameter changes (from "Try Again") before any widgets render ──
# Streamlit forbids setting widget-bound keys after the widget has been drawn.
# We store changes in _pending_param_changes and apply them here on the next rerun.
_pending = st.session_state.pop("_pending_param_changes", {})
for _ppk, _ppv in _pending.items():
    st.session_state[_ppk] = _ppv
CELLS = [
    ("Empty",    ".",  "#F1F5F9", "#CBD5E1", "",    "#334155"),
    ("Wall",     "#",  "#1E293B", "#0F172A", "■",   "#94A3B8"),
    ("Slippery", "~",  "#BAE6FD", "#7DD3FC", "≈",   "#0369A1"),
    ("Goal",     "G",  "#BBF7D0", "#4ADE80", "🏁",  "#166534"),
    ("Trap",     "X",  "#FECACA", "#F87171", "✗",   "#991B1B"),
    ("Start",    "S",  "#FEF08A", "#FACC15", "🚦",  "#713F12"),
]
CELL_CHAR  = [c[1] for c in CELLS]
CELL_FILL  = [c[2] for c in CELLS]
CELL_LINE  = [c[3] for c in CELLS]
CELL_LBL   = [c[4] for c in CELLS]
CELL_FG    = [c[5] for c in CELLS]
CELL_NAME  = [c[0] for c in CELLS]
CHAR_TO_IDX = {ch: i for i, ch in enumerate(CELL_CHAR)}
E, WL, SL, GO, TR, ST = 0, 1, 2, 3, 4, 5

ARROW_DX = [0.0,  0.22, 0.0,  -0.22]   # UP RIGHT DOWN LEFT
ARROW_DY = [0.22, 0.0,  -0.22, 0.0]    # in Plotly coords (y flipped)

RDYLGN = [
    [0.0,  "#d73027"], [0.15, "#f46d43"], [0.3, "#fdae61"],
    [0.45, "#fee08b"], [0.6,  "#d9ef8b"], [0.75,"#a6d96a"],
    [0.9,  "#66bd63"], [1.0,  "#1a9850"],
]

# ── Session state init ────────────────────────────────────────────────
def _blank_grid(rows, cols):
    g = [[E] * cols for _ in range(rows)]
    g[0][0] = ST
    return g

if "grid" not in st.session_state:
    st.session_state.grid = _blank_grid(5, 5)
    st.session_state.rows = 5
    st.session_state.cols = 5
    st.session_state.tool = E
    st.session_state.results = None


def _safe_rerun() -> None:
    """Attempt to rerun the Streamlit script; ignore if API is unavailable."""
    try:
        # experimental_rerun may not exist on all Streamlit builds
        getattr(st, "experimental_rerun")()
    except Exception:
        # fall back to doing nothing; UI will update on next interaction
        return


# Default values for all training / dynamics parameters (used by the Reset button)
_TRAINING_DEFAULTS: dict = {
    "algo": "Q-Learning",
    "run_mode": "Full",
    "episodes": 1000,
    "alpha": 0.1,
    "epsilon": 1.0,
    "epsilon_decay": 0.998,
    "exploring_starts": True,
    "max_steps": 300,
    "step_rew": 0.0,
    "slip_prob": 0.3,
    "gamma": 0.95,
    "obs_n_neighbors": 0,
    "obs_use_goal_dist": False,
    "dqn_hidden_str": "64,64",
    "dqn_lr": 1e-3,
    "dqn_batch_size": 64,
    "dqn_buffer_size": 10_000,
    "dqn_target_update": 100,
    "dqn_episodes": 1000,
    "dqn_epsilon": 1.0,
    "dqn_epsilon_decay": 0.997,
    "dqn_epsilon_min": 0.01,
    "use_curriculum": False,
    "curriculum_method": "ADR",
    "n_curriculum_setups": 100,
    "curriculum_perf_threshold": 0.5,
}


def _grid_to_layout(grid, rows, cols):
    return ["".join(CELL_CHAR[grid[r][c]] for c in range(cols))
            for r in range(rows)]


def _layout_to_grid(layout):
    return [[CHAR_TO_IDX[ch] for ch in row] for row in layout]


def _reward_cell_colors(cr):
    """Return (fillcolor, linecolor) for non-terminal cells carrying a reward.

    Colour intensity scales smoothly with |cr|, saturating around 5.
    """
    v = float(cr)
    t = min(1.0, abs(v) / 5.0)                 # 0 → faint, 1 → solid
    alpha = round(0.18 + 0.55 * t, 2)
    if v >= 0:
        return f"rgba(74,222,128,{alpha})", "#16A34A"   # green-400
    else:
        return f"rgba(248,113,113,{alpha})", "#DC2626"  # red-400


def _generate_random_layout(scale, rows, cols, hints=None):
    hints = hints or {}
    scale = int(max(1, min(10, hints.get("scale", scale))))

    def _bias_factor(key, default=1.0):
        value = str(hints.get(key, "")).strip().lower()
        if value in {"more", "higher", "increase", "larger", "many", "greater"}:
            return 1.4
        if value in {"fewer", "less", "lower", "smaller", "reduced", "few"}:
            return 0.6
        return default

    wall_factor = _bias_factor("wall_bias", 1.0)
    slippery_factor = _bias_factor("slippery_bias", 1.0)
    pos_factor = _bias_factor("positive_reward_bias", 1.0)
    neg_factor = _bias_factor("negative_reward_bias", 1.0)

    grid = [[E] * cols for _ in range(rows)]
    cell_rewards = {}

    # Start position - use hints if provided, otherwise default to top row
    start_row = hints.get("start_row")
    start_col = hints.get("start_col")
    
    # Validate and apply start position from hints
    if start_row is not None and start_col is not None:
        try:
            start_row = int(start_row)
            start_col = int(start_col)
            # Clamp to valid grid bounds
            start_row = max(0, min(rows - 1, start_row))
            start_col = max(0, min(cols - 1, start_col))
        except (ValueError, TypeError):
            # If hints are invalid, fall back to default
            start_row = 0
            start_col = random.randrange(cols)
    else:
        # Default: start on the upper row with random column
        start_row = 0
        start_col = random.randrange(cols)
    
    grid[start_row][start_col] = ST
    occupied = {(start_row, start_col)}

    # Main goal on the last row (avoid start position)
    last_row_positions = [(rows - 1, c) for c in range(cols)
                          if (rows - 1, c) != (start_row, start_col)]
    if not last_row_positions:
        last_row_positions = [(r, c) for r in range(rows) for c in range(cols)
                              if (r, c) != (start_row, start_col)]
    main_goal = random.choice(last_row_positions)
    grid[main_goal[0]][main_goal[1]] = GO
    occupied.add(main_goal)

    available = [(r, c) for r in range(rows) for c in range(cols)
                 if (r, c) not in occupied]
    random.shuffle(available)

    # Additional goals
    extra_goals = hints.get("extra_goals")
    if extra_goals is None:
        extra_goals = max(0, min(3, (scale - 1) // 3))
    else:
        try:
            extra_goals = max(0, min(3, int(extra_goals)))
        except Exception:
            extra_goals = max(0, min(3, (scale - 1) // 3))

    for _ in range(extra_goals):
        choices = [pos for pos in available if pos[0] != 0]
        if not choices:
            choices = available
        if not choices:
            break
        pos = choices.pop(random.randrange(len(choices)))
        if pos in available:
            available.remove(pos)
        grid[pos[0]][pos[1]] = GO
        occupied.add(pos)
        cell_rewards[pos] = 1 + scale

    # Set main goal as largest goal
    goal_value = hints.get("goal_value")
    if goal_value is None:
        goal_value = 2 + scale
    else:
        try:
            goal_value = int(goal_value)
        except Exception:
            goal_value = 2 + scale
    cell_rewards[main_goal] = max(1, goal_value)

    # Traps
    if hints.get("no_traps"):
        num_traps = 0
    else:
        num_traps = 0 if scale <= 4 else min(5, (scale - 3) // 2)
    trap_value = hints.get("trap_value")
    if trap_value is not None:
        try:
            trap_value = int(trap_value)
            if trap_value >= 0:
                trap_value = -(abs(trap_value) or 1)
        except Exception:
            trap_value = None
    if trap_value is None:
        trap_value = -(1 + random.randint(0, scale))

    for _ in range(num_traps):
        choices = [pos for pos in available if pos[0] != 0]
        if not choices:
            choices = available
        if not choices:
            break
        pos = choices.pop(random.randrange(len(choices)))
        if pos in available:
            available.remove(pos)
        grid[pos[0]][pos[1]] = TR
        occupied.add(pos)
        cell_rewards[pos] = trap_value

    # Walls
    num_walls = 0 if scale <= 2 else min(len(available) - 1,
        max(1, int(rows * cols * min(0.16, (scale - 2) / 10))))
    num_walls = min(len(available) - 1, max(0, int(num_walls * wall_factor)))
    for _ in range(num_walls):
        if not available:
            break
        pos = available.pop(random.randrange(len(available)))
        grid[pos[0]][pos[1]] = WL
        occupied.add(pos)

    # Slippery cells
    num_slippery = 0 if scale <= 3 else min(len(available) - 1,
        max(1, int(rows * cols * min(0.12, (scale - 3) / 10))))
    num_slippery = min(len(available) - 1, max(0, int(num_slippery * slippery_factor)))
    for _ in range(num_slippery):
        if not available:
            break
        pos = available.pop(random.randrange(len(available)))
        grid[pos[0]][pos[1]] = SL
        occupied.add(pos)

    # Positive and negative reward cells
    reward_spots = [pos for pos in available if grid[pos[0]][pos[1]] == E]
    random.shuffle(reward_spots)
    pos_rewards = hints.get("pos_reward_count")
    neg_rewards = hints.get("neg_reward_count")
    if pos_rewards is None:
        pos_rewards = 0 if scale <= 1 else min(len(reward_spots), (scale + 1) // 2)
    else:
        try:
            pos_rewards = max(0, int(pos_rewards))
        except Exception:
            pos_rewards = 0 if scale <= 1 else min(len(reward_spots), (scale + 1) // 2)
    if neg_rewards is None:
        neg_rewards = 0 if scale <= 2 else min(len(reward_spots) - pos_rewards, max(0, (scale - 2) // 2))
    else:
        try:
            neg_rewards = max(0, int(neg_rewards))
        except Exception:
            neg_rewards = 0 if scale <= 2 else min(len(reward_spots) - pos_rewards, max(0, (scale - 2) // 2))

    if pos_factor > 1.1 and not hints.get("no_pos_rewards"):
        pos_rewards = min(len(reward_spots), pos_rewards + 1)
    if neg_factor > 1.1 and not hints.get("no_neg_rewards"):
        neg_rewards = min(len(reward_spots) - pos_rewards, neg_rewards + 1)

    for _ in range(pos_rewards):
        if not reward_spots:
            break
        pos = reward_spots.pop()
        cell_rewards[pos] = random.randint(1, max(1, scale))
    for _ in range(neg_rewards):
        if not reward_spots:
            break
        pos = reward_spots.pop()
        cell_rewards[pos] = -random.randint(1, max(1, scale))

    # Ensure the intended start cell is still ST (don't force row 0 when start_row != 0)
    if grid[start_row][start_col] != ST:
        grid[start_row][start_col] = ST
    if main_goal not in occupied:
        grid[main_goal[0]][main_goal[1]] = GO

    # Post-process: fill specified rows/cols with walls, preserving ST and GO cells
    for wr in hints.get("wall_rows", []):
        try:
            wr = int(wr)
            if 0 <= wr < rows:
                for c in range(cols):
                    if grid[wr][c] not in (ST, GO):
                        grid[wr][c] = WL
        except (ValueError, TypeError):
            pass
    for wc in hints.get("wall_cols", []):
        try:
            wc = int(wc)
            if 0 <= wc < cols:
                for r in range(rows):
                    if grid[r][wc] not in (ST, GO):
                        grid[r][wc] = WL
        except (ValueError, TypeError):
            pass

    return grid, cell_rewards


def _get_claude_api_key():
    key = os.environ.get("CLAUDE_API_KEY")
    if key:
        return key

    try:
        secrets = getattr(st, "secrets", None)
    except Exception:
        return None

    if secrets is None:
        return None

    try:
        return secrets.get("CLAUDE_API_KEY")
    except Exception:
        try:
            return secrets["CLAUDE_API_KEY"]
        except Exception:
            return None


def _extract_json_object(text):
    # Accept either a raw text response or a parsed JSON-like object
    def _find_json_in_obj(obj):
        if isinstance(obj, str):
            s = obj.strip()
            start = s.find("{")
            end = s.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(s[start:end + 1])
                except Exception:
                    return None
            return None
        if isinstance(obj, dict):
            for v in obj.values():
                res = _find_json_in_obj(v)
                if res is not None:
                    return res
        if isinstance(obj, (list, tuple)):
            for v in obj:
                res = _find_json_in_obj(v)
                if res is not None:
                    return res
        return None

    if isinstance(text, (dict, list, tuple)):
        result = _find_json_in_obj(text)
        if result is None:
            raise ValueError("No JSON object found in Claude response (nested)")
        return result

    if not isinstance(text, str):
        raise ValueError("Claude response is not text")

    result = _find_json_in_obj(text)
    if result is None:
        raise ValueError("No JSON object found in Claude response")
    return result


def _normalize_claude_hints(directives):
    hints = {}
    if not isinstance(directives, dict):
        return hints

    def _normalize_bias(value):
        if value is None:
            return None
        text = str(value).strip().lower()
        if text in {"more", "higher", "increase", "larger", "many", "greater", "lots"}:
            return "more"
        if text in {"fewer", "less", "lower", "smaller", "reduced", "few"}:
            return "fewer"
        return text

    if "scale" in directives:
        try:
            hints["scale"] = int(directives["scale"])
        except Exception:
            pass
    if "start_row" in directives:
        try:
            hints["start_row"] = int(directives["start_row"])
        except Exception:
            pass
    if "start_col" in directives:
        try:
            hints["start_col"] = int(directives["start_col"])
        except Exception:
            pass
    if "wall_bias" in directives:
        hints["wall_bias"] = _normalize_bias(directives["wall_bias"])
    if "slippery_bias" in directives:
        hints["slippery_bias"] = _normalize_bias(directives["slippery_bias"])
    if "positive_reward_bias" in directives:
        hints["positive_reward_bias"] = _normalize_bias(directives["positive_reward_bias"])
    if "negative_reward_bias" in directives:
        hints["negative_reward_bias"] = _normalize_bias(directives["negative_reward_bias"])
    if "goal_value" in directives:
        try:
            hints["goal_value"] = int(directives["goal_value"])
        except Exception:
            pass
    if "trap_value" in directives:
        try:
            hints["trap_value"] = int(directives["trap_value"])
        except Exception:
            pass
    if "extra_goals" in directives:
        try:
            hints["extra_goals"] = int(directives["extra_goals"])
        except Exception:
            pass
    if "pos_reward_count" in directives:
        try:
            hints["pos_reward_count"] = int(directives["pos_reward_count"])
        except Exception:
            pass
    if "neg_reward_count" in directives:
        try:
            hints["neg_reward_count"] = int(directives["neg_reward_count"])
        except Exception:
            pass
    if "wall_rows" in directives:
        try:
            hints["wall_rows"] = [int(x) for x in directives["wall_rows"]]
        except Exception:
            pass
    if "wall_cols" in directives:
        try:
            hints["wall_cols"] = [int(x) for x in directives["wall_cols"]]
        except Exception:
            pass
    return hints


def _build_claude_grid_prompt(prompt_text, rows, cols, complexity):
    complexity_desc = (
        "simple open grid with a clear path to the goal"
        if complexity <= 3 else
        "moderate obstacles, some traps, and a few alternative paths"
        if complexity <= 6 else
        "complex maze-like layout with multiple hazards and longer paths"
    )
    return (
        "You are designing a GridWorld environment for a reinforcement learning agent.\n\n"
        f"Grid size: {rows} rows × {cols} cols.  Complexity: {complexity}/10 ({complexity_desc}).\n\n"
        "CELL TYPES — use only these characters:\n"
        "  .  Empty     — agent moves through freely; small step penalty applies\n"
        "  #  Wall      — impassable; agent cannot enter\n"
        "  ~  Slippery  — agent may slide sideways instead of moving as intended\n"
        "  G  Goal      — terminal: large positive reward, episode ends (place 1–3)\n"
        "  X  Trap      — terminal: large negative reward, episode ends\n"
        "  S  Start     — exactly one required; where the agent begins each episode\n\n"
        "HARD RULES:\n"
        "  1. Exactly one S cell.\n"
        "  2. At least one G cell.\n"
        f"  3. Every row must be exactly {cols} characters; there must be exactly {rows} rows.\n"
        "  4. There must be a navigable path (no wall blocking every route) from S to at least one G.\n\n"
        "VALUE & REWARD CONCEPT:\n"
        "  The agent learns which states are valuable (expected future reward).\n"
        "  Cells near goals have high value; cells near traps or dead-ends have low value.\n"
        "  Walls have no value. Slippery cells add stochasticity and make learning harder.\n\n"
        f"USER REQUEST: {prompt_text}\n\n"
        "Return ONLY a JSON object — no prose, no markdown fences:\n"
        "{\"layout\": [\"row0chars\", \"row1chars\", ...]}\n\n"
        f"Example for {rows}×{cols}:\n"
        + "{\"layout\": [" + ", ".join(f'\"{"S" + "." * (cols - 1) if i == 0 else "." * (cols - 1) + "G" if i == rows - 1 else "." * cols}\"' for i in range(rows)) + "]}"
    )


def _validate_claude_layout(layout, rows, cols):
    """Return (True, None) if valid, else (False, error_string)."""
    if not isinstance(layout, list):
        return False, "layout is not a list"
    if len(layout) != rows:
        return False, f"expected {rows} rows, got {len(layout)}"
    valid_chars = set(".#~GXS")
    for i, row in enumerate(layout):
        if not isinstance(row, str):
            return False, f"row {i} is not a string"
        if len(row) != cols:
            return False, f"row {i} has {len(row)} chars, expected {cols}"
        bad = [ch for ch in row if ch not in valid_chars]
        if bad:
            return False, f"row {i} contains invalid character(s): {bad}"
    flat = "".join(layout)
    n_s = flat.count("S")
    n_g = flat.count("G")
    if n_s != 1:
        return False, f"expected exactly 1 'S', got {n_s}"
    if n_g < 1:
        return False, "no goal cell 'G' found"
    return True, None


def _generate_grid_from_claude(prompt_text, rows, cols, complexity):
    """Ask Claude to produce a complete grid layout. Returns (grid, None) or (None, error_str)."""
    try:
        full_prompt = _build_claude_grid_prompt(prompt_text, rows, cols, complexity)
        response_text = _call_claude(full_prompt)
        data = _extract_json_object(response_text)
        layout = data.get("layout")
        ok, err = _validate_claude_layout(layout, rows, cols)
        if not ok:
            return None, f"Layout validation failed: {err}"
        return _layout_to_grid(layout), None
    except Exception as e:
        return None, str(e)


def _build_claude_prompt(prompt_text, rows, cols, complexity):
    """Legacy hints prompt — kept for reference but no longer used."""
    center_row = rows // 2
    center_col = cols // 2
    return (
        "You are a helper that converts a user request into a JSON configuration object for a GridWorld random layout generator. "
        f"rows: {rows}, cols: {cols}, baseline complexity: {complexity}.\n"
        f"Center of grid is approximately row {center_row}, col {center_col}.\n"
        f"User request: {prompt_text}\n"
        "Return only a JSON object with relevant keys. Do not include explanation."
    )


def _call_claude(prompt_text, max_tokens: int = 300):
    api_key = _get_claude_api_key()
    if not api_key:
        raise RuntimeError("Claude API key is not configured. Set CLAUDE_API_KEY or Streamlit secrets.")

    # Use the newer messages endpoint (the /v1/complete endpoint is deprecated)
    url = "https://api.anthropic.com/v1/messages"
    # pick model from session_state (set by UI) or env/secrets
    model = None
    try:
        model = st.session_state.get("CLAUDE_MODEL")
    except Exception:
        model = None
    if not model:
        model = _get_claude_model_env() or "claude-opus-4-8"

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text}
                ]
            }
        ],
        "max_tokens": max_tokens,
    }

    # Basic validation of API key and payload
    if not isinstance(api_key, str) or len(api_key) < 10:
        st.write("**Debug - Claude API key appears invalid or too short**")
    if not payload.get("model"):
        raise RuntimeError("Invalid payload for Claude: missing 'model' field")
    # Accept either legacy 'prompt' (complete endpoint) or new 'messages' (messages endpoint)
    if not (payload.get("prompt") or payload.get("messages")):
        st.write("**Debug - Payload keys:**")
        st.write(list(payload.keys()))
        raise RuntimeError("Invalid payload for Claude: missing 'prompt' or 'messages' field")

    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "Anthropic-Version": "2023-06-01",
    }
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode("utf-8", errors="replace")
        except Exception:
            err_body = "<failed to read error body>"
        st.write(f"**Debug - Claude HTTPError:** {e.code} {getattr(e, 'reason', '')}")
        st.write("**Debug - Response body:**")
        st.write(err_body)
        # If the error indicates 'temperature' is deprecated for this model, retry once without it
        if "temperature" in err_body and "deprecated" in err_body.lower():
            try:
                st.write("**Debug - Retrying without 'temperature' parameter**")
                payload.pop("temperature", None)
                data = json.dumps(payload).encode("utf-8")
                request = urllib.request.Request(url, data=data, headers=headers, method="POST")
                with urllib.request.urlopen(request, timeout=30) as response:
                    body = response.read().decode("utf-8")
            except urllib.error.HTTPError as e2:
                try:
                    err_body2 = e2.read().decode("utf-8", errors="replace")
                except Exception:
                    err_body2 = "<failed to read error body>"
                st.write(f"**Debug - Retry HTTPError:** {e2.code}")
                st.write(err_body2)
                raise RuntimeError(f"Claude HTTPError on retry: {e2.code} - {err_body2}")
            except Exception as e2:
                st.write(f"**Debug - Retry request error:** {e2}")
                raise
        else:
            try:
                st.write("**Debug - Response headers:**")
                st.write(dict(e.headers))
            except Exception:
                pass
            raise RuntimeError(f"Claude HTTPError: {e.code} - {err_body}")
    except urllib.error.URLError as e:
        st.write(f"**Debug - Claude URLError:** {getattr(e, 'reason', e)}")
        raise
    except Exception as e:
        st.write(f"**Debug - Claude request error:** {e}")
        raise

    try:
        response_json = json.loads(body)
    except Exception:
        st.write("**Debug - Could not parse JSON from Claude response body:**")
        st.write(body)
        raise

    # Helper: extract text from various possible Anthropic response shapes
    def _extract_text_from_response(rj):
        if not isinstance(rj, dict):
            return None
        # direct completion
        if isinstance(rj.get("completion"), str):
            return rj.get("completion")
        # /v1/messages response: content at top level
        content = rj.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    return part.get("text")
        if isinstance(content, str):
            return content
        # message object (nested)
        msg = rj.get("message") or rj.get("messages")
        if isinstance(msg, dict):
            content = msg.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        return part.get("text")
            if isinstance(content, str):
                return content
        # choices array (like other chat APIs)
        choices = rj.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            # try message content
            if isinstance(first.get("message"), dict):
                c = first["message"].get("content")
                if isinstance(c, list):
                    for part in c:
                        if isinstance(part, dict) and isinstance(part.get("text"), str):
                            return part.get("text")
                if isinstance(c, str):
                    return c
            # try text field
            if isinstance(first.get("text"), str):
                return first.get("text")
        # nested output keys
        for k in ("response", "output", "result", "text"):
            v = rj.get(k)
            if isinstance(v, str):
                return v
        return None

    completion = _extract_text_from_response(response_json)
    if completion is None:
        # last resort: try to stringify the whole body
        st.write("**Debug - Unable to locate text in Claude response JSON; returning full JSON as string**")
        completion = json.dumps(response_json)
    return completion


def _infer_layout_directives_from_prompt(prompt_text, rows, cols):
    hints = {}
    normalized = prompt_text.lower()
    if "fewer walls" in normalized or "less walls" in normalized or "fewer wall" in normalized:
        hints["wall_bias"] = "fewer"
    if "more walls" in normalized or "more wall" in normalized:
        hints["wall_bias"] = "more"
    if "less slippery" in normalized or "fewer slippery" in normalized:
        hints["slippery_bias"] = "fewer"
    if "more slippery" in normalized:
        hints["slippery_bias"] = "more"
    if "large positive" in normalized or "large terminal" in normalized or "larger goal" in normalized:
        hints["goal_value"] = 2 + max(1, int(min(10, len(normalized))))
    if "more negative" in normalized or "more negative rewards" in normalized or "negative rewards" in normalized:
        hints["negative_reward_bias"] = "more"
    if "less negative" in normalized or "fewer negative" in normalized:
        hints["negative_reward_bias"] = "fewer"
    # Start position hints
    if "center" in normalized or "middle" in normalized:
        hints["start_row"] = rows // 2
        hints["start_col"] = cols // 2
    elif "top" in normalized:
        hints["start_row"] = 0
        hints["start_col"] = cols // 2
    elif "bottom" in normalized:
        hints["start_row"] = rows - 1
        hints["start_col"] = cols // 2
    elif "left" in normalized:
        hints["start_row"] = rows // 2
        hints["start_col"] = 0
    elif "right" in normalized:
        hints["start_row"] = rows // 2
        hints["start_col"] = cols - 1

    # Row / column wall hints
    wall_rows = []
    wall_cols = []
    if any(p in normalized for p in ("bottom row", "lower row", "last row", "bottom wall")):
        wall_rows.append(rows - 1)
    if any(p in normalized for p in ("top row", "upper row", "first row", "top wall")):
        wall_rows.append(0)
    if any(p in normalized for p in ("left column", "left wall", "first column")):
        wall_cols.append(0)
    if any(p in normalized for p in ("right column", "right wall", "last column")):
        wall_cols.append(cols - 1)
    if wall_rows:
        hints["wall_rows"] = wall_rows
    if wall_cols:
        hints["wall_cols"] = wall_cols

    # Trap hints
    _no_trap_phrases = (
        "no trap", "no traps", "without trap", "without traps",
        "do not place trap", "don't place trap", "dont place trap",
        "remove trap", "avoid trap", "0 trap", "zero trap",
    )
    if any(p in normalized for p in _no_trap_phrases):
        hints["no_traps"] = True

    # Single / limited goal hints
    _one_goal_phrases = (
        "one goal", "only one goal", "single goal", "1 goal",
        "one g ", "place only one", "just one goal", "only 1 goal",
    )
    if any(p in normalized for p in _one_goal_phrases):
        hints["extra_goals"] = 0

    # No positive cell-reward hints
    _no_pos_phrases = (
        "no positive reward", "no positive rewards",
        "without positive reward", "without positive rewards",
        "no cell reward", "no extra reward", "no bonus reward",
        "remove positive reward", "0 positive", "zero positive",
    )
    if any(p in normalized for p in _no_pos_phrases):
        hints["no_pos_rewards"] = True
        hints["pos_reward_count"] = 0

    # No negative cell-reward hints (non-trap cells)
    _no_neg_phrases = (
        "no negative reward", "no negative rewards",
        "without negative reward", "without negative rewards",
        "remove negative reward", "0 negative", "zero negative",
    )
    if any(p in normalized for p in _no_neg_phrases):
        hints["no_neg_rewards"] = True
        hints["neg_reward_count"] = 0

    # Numeric goal reward — parse "reward of 100", "goal reward 100",
    # "goal.*reward.*\d+", "reward.*\d+.*goal", etc.
    _gr_match = (
        re.search(r'goal\s+(?:reward|value)\s*(?:of\s*)?(\d+(?:\.\d+)?)', normalized)
        or re.search(r'reward\s+of\s+(\d+(?:\.\d+)?)', normalized)
        or re.search(r'reward\s*[=:]\s*(\d+(?:\.\d+)?)', normalized)
        or re.search(r'(\d+(?:\.\d+)?)\s+(?:goal\s+)?reward', normalized)
    )
    if _gr_match:
        try:
            _gv = float(_gr_match.group(1))
            if _gv > 0:
                hints["goal_value"] = int(_gv) if _gv == int(_gv) else _gv
        except (ValueError, IndexError):
            pass

    return hints


def _generate_layout_hints_from_prompt(prompt_text, rows, cols, complexity):
    hints = _infer_layout_directives_from_prompt(prompt_text, rows, cols)
    if not _get_claude_api_key():
        return hints

    try:
        request_text = _build_claude_prompt(prompt_text, rows, cols, complexity)
        response_text = _call_claude(request_text)
        directives = _extract_json_object(response_text)
        claude_hints = _normalize_claude_hints(directives)
        # Debug: print what Claude returns
        st.write(f"**Debug - Claude Response JSON:** {directives}")
        st.write(f"**Debug - Parsed Hints:** {claude_hints}")
        hints.update(claude_hints)
    except Exception as e:
        st.write(f"**Debug - Claude parsing error:** {e}")

    return hints


def _cell_px(rows, cols):
    return max(48, min(80, 380 // max(rows, cols)))


def _to_json_serializable(value):
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _to_json_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_serializable(v) for v in value]
    return value


def _friendly_timestamp(dt: datetime = None) -> str:
    dt = dt or datetime.now()
    # Format: day-month-year hour:minute (no seconds), e.g. 27-5-2026 15:11
    return f"{dt.day}-{dt.month}-{dt.year} {dt.hour:02d}:{dt.minute:02d}"


def _filter_config_for_algo(cfg: dict, algo: str) -> dict:
    if not isinstance(cfg, dict):
        return cfg
    algo = (algo or "").lower()
    if "q-learning" in algo or algo == "q-learning" or "sarsa" in algo:
        keys = ["alpha", "n_episodes", "max_steps", "epsilon", "epsilon_decay", "epsilon_min", "exploring_starts"]
    elif "dqn" in algo:
        keys = [
            "dqn_hidden", "dqn_lr", "dqn_batch_size", "dqn_buffer_size",
            "dqn_target_update", "dqn_n_episodes", "dqn_epsilon", "dqn_epsilon_decay", "dqn_epsilon_min",
        ]
    elif "policy" in algo or "policy iteration" in algo:
        keys = ["pi_threshold", "pi_max_iter"]
    else:
        # fallback: show everything
        return cfg
    return {k: cfg.get(k) for k in keys if k in cfg}


def _format_param_value(v):
    # Booleans as 'true'/'false'
    if isinstance(v, bool):
        return "true" if v else "false"
    # Numpy ints
    try:
        import numpy as _np
        if isinstance(v, _np.integer):
            return int(v)
        if isinstance(v, _np.floating):
            fv = float(v)
            return int(fv) if abs(fv - int(fv)) < 1e-8 else fv
    except Exception:
        pass
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(v) if abs(v - int(v)) < 1e-8 else v
    if v is None:
        return ""
    if isinstance(v, (list, tuple, dict)):
        return json.dumps(v)
    return v


def _make_training_save_payload(results, grid_cfg, algo_cfg, name, description):
    return {
        "schema": "gridworld_rl_training_result",
        "version": 1,
        "metadata": {
            "name": name or "Untitled training result",
            "description": description or "",
            # friendly local timestamp without milliseconds
            "saved_at": _friendly_timestamp(),
            "algorithm": results.get("algo"),
            "config": _to_json_serializable(vars(algo_cfg)),
            "grid_config": {
                "rows": grid_cfg.rows,
                "cols": grid_cfg.cols,
                "layout": grid_cfg.layout,
                "step_reward": grid_cfg.step_reward,
                "goal_reward": grid_cfg.goal_reward,
                "trap_reward": grid_cfg.trap_reward,
                "cell_rewards": {f"{r},{c}": float(v) for (r, c), v in grid_cfg.cell_rewards.items()},
                "slip_prob": grid_cfg.slippery_slip_prob,
                "gamma": grid_cfg.gamma,
            },
        },
        "results": {
            "policy": _to_json_serializable(results.get("policy")),
            "values": _to_json_serializable(results.get("values")),
            "rewards": _to_json_serializable(results.get("rewards")),
            "lengths": _to_json_serializable(results.get("lengths")),
            "visits": _to_json_serializable(results.get("visits")),
            "episode_times": _to_json_serializable(results.get("episode_times")),
            "off_policy_steps": _to_json_serializable(results.get("off_policy_steps")),
            "epsilon_values": _to_json_serializable(results.get("epsilon_values")),
            "episode_traces": _to_json_serializable(results.get("episode_traces")),
        },
    }


def _load_training_payload(data):
    if data.get("schema") != "gridworld_rl_training_result":
        raise ValueError("This file is not a GridWorld training result.")
    meta = data["metadata"]
    grid_data = meta["grid_config"]
    st.session_state.grid = _layout_to_grid(grid_data["layout"])
    st.session_state.rows = int(grid_data["rows"])
    st.session_state.cols = int(grid_data["cols"])
    st.session_state.step_rew = float(grid_data.get("step_reward", 0.0))
    st.session_state.slip_prob = float(grid_data.get("slip_prob", 0.3))
    st.session_state.gamma = float(grid_data.get("gamma", 0.95))
    st.session_state.cell_rewards = {
        tuple(map(int, k.split(","))): float(v)
        for k, v in grid_data.get("cell_rewards", {}).items()
    }
    st.session_state.algo = meta.get("algorithm", st.session_state.get("algo", "Policy Iteration"))
    cfg = meta.get("config", {})
    for key, value in cfg.items():
        if key in st.session_state:
            st.session_state[key] = value
    env_cfg = GridConfig(
        rows=int(grid_data["rows"]), cols=int(grid_data["cols"]), layout=grid_data["layout"],
        step_reward=float(grid_data.get("step_reward", 0.0)),
        goal_reward=float(grid_data.get("goal_reward", 1.0)),
        trap_reward=float(grid_data.get("trap_reward", -1.0)),
        cell_rewards={tuple(map(int, k.split(","))): float(v)
                      for k, v in grid_data.get("cell_rewards", {}).items()},
        slippery_slip_prob=float(grid_data.get("slip_prob", 0.3)),
        gamma=float(grid_data.get("gamma", 0.95)),
    )
    env = GridWorld(env_cfg)
    results = data.get("results", {})
    results["env"] = env
    results["algo"] = meta.get("algorithm", results.get("algo"))
    st.session_state.results = results
    st.session_state.loaded_training_metadata = meta
    return meta


# ── Plotly helpers ────────────────────────────────────────────────────


def make_editor_fig(grid, rows, cols, trace=None, current_step=None, cell_rewards=None):
    """Render the grid as a Plotly figure.

    Parameters
    ----------
    cell_rewards : dict | None
        Optional ``{(row, col): float}`` override.  When *None* the function
        falls back to ``st.session_state["cell_rewards"]`` (the live editor
        state).  Pass an explicit dict when rendering curriculum / saved layouts
        that are not currently loaded in the editor.
    """
    cell = _cell_px(rows, cols)
    fig = go.Figure()

    for r in range(rows):
        for c in range(cols):
            t  = grid[r][c]
            y0 = rows - 1 - r

            # ── Resolve custom reward for this cell ───────────────────
            if cell_rewards is not None:
                cr = cell_rewards.get((r, c), None)
            else:
                try:
                    cr = st.session_state.get("cell_rewards", {}).get((r, c), None)
                except Exception:
                    cr = None

            # ── Cell background — tint non-terminal cells by reward ───
            fill = CELL_FILL[t]
            border = CELL_LINE[t]
            if cr is not None and float(cr) != 0 and t in (E, SL):
                fill, border = _reward_cell_colors(cr)

            fig.add_shape(
                type="rect", layer="below",
                x0=c+0.05, y0=y0+0.05, x1=c+0.95, y1=y0+0.95,
                fillcolor=fill,
                line=dict(color=border, width=1.5),
            )

            # ── Cell label ────────────────────────────────────────────
            lbl = CELL_LBL[t]
            if lbl:
                # Emoji labels (Start 🚦 / Goal 🏁) must not be wrapped in
                # HTML <b> tags — Plotly renders them as plain text.
                is_emoji = any(ord(ch) > 127 for ch in lbl)
                txt = lbl if is_emoji else f"<b>{lbl}</b>"
                fig.add_annotation(
                    x=c+0.5, y=y0+0.5, text=txt,
                    showarrow=False,
                    font=dict(size=max(13, cell // 4), color=CELL_FG[t]),
                    xanchor="center", yanchor="middle",
                )

            # ── Reward badge (bottom-right corner) ───────────────────
            if cr is not None and float(cr) != 0:
                badge_color = "#ECFDF5" if float(cr) >= 0 else "#FEF3F2"
                badge_border = "#86EFAC" if float(cr) >= 0 else "#FCA5A5"
                badge_fg = "#065F46" if float(cr) >= 0 else "#9B1C1C"
                fig.add_annotation(
                    x=c+0.78, y=y0+0.22,
                    text=(
                        f"<span style='background:{badge_color};"
                        f"border:1px solid {badge_border};"
                        f"padding:2px 5px;border-radius:8px;"
                        f"color:{badge_fg};font-weight:600'>"
                        f"{float(cr):+.2f}</span>"
                    ),
                    showarrow=False, font=dict(size=9),
                    xanchor="center", yanchor="middle", opacity=0.95,
                )

    if trace is not None:
        states = trace.get("states", [])
        actions = trace.get("actions", [])
        rewards = trace.get("rewards", [])
        shown_states = states if current_step is None else states[: current_step + 1]
        xs = []
        ys = []
        customdata = []
        for idx, s in enumerate(shown_states):
            r, c = divmod(s, cols)
            xs.append(c + 0.5)
            ys.append(rows - 1 - r + 0.5)
            if idx < len(actions):
                action_name = GridWorld.ACTION_NAMES[actions[idx]]
                reward_value = rewards[idx]
            else:
                action_name = "END"
                reward_value = 0.0
            customdata.append((f"{idx}", f"({r},{c})", action_name, f"{reward_value:+.2f}"))

        if xs:
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines+markers",
                line=dict(color="#7c3aed", width=3),
                marker=dict(size=10, color="#a78bfa", line=dict(color="white", width=2)),
                hovertemplate="<b>Step %{customdata[0]}</b><br>Pos %{customdata[1]}<br>Action %{customdata[2]}<br>Reward %{customdata[3]}<extra></extra>",
                customdata=customdata,
                showlegend=False,
            ))
            fig.add_trace(go.Scatter(
                x=[xs[-1]], y=[ys[-1]], mode="markers",
                marker=dict(size=18, color="#ef4444", symbol="x"),
                hoverinfo="skip", showlegend=False,
            ))
            fig.add_annotation(
                x=xs[0], y=ys[0], text="Start",
                showarrow=True, arrowhead=2, arrowsize=1.2,
                ax=0, ay=-20, font=dict(color="#065f46", size=11),
            )
            fig.add_annotation(
                x=xs[-1], y=ys[-1], text="End",
                showarrow=True, arrowhead=2, arrowsize=1.2,
                ax=0, ay=20, font=dict(color="#b91c1c", size=11),
            )

    xs = [c + 0.5 for r in range(rows) for c in range(cols)]
    ys = [rows - 1 - r + 0.5 for r in range(rows) for c in range(cols)]

    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers",
        marker=dict(size=max(28, cell * 0.6), opacity=0.01, color="#fff"),
        hovertemplate="<b>%{customdata[0]}</b> (%{customdata[1]},%{customdata[2]})"
                      "<br><i>click to paint</i><extra></extra>",
        customdata=[(CELL_NAME[grid[r][c]], r, c)
                    for r in range(rows) for c in range(cols)],
    ))

    fig.update_layout(
        width=cols * cell, height=rows * cell,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False, dragmode=False,
        plot_bgcolor="#F8FAFC", paper_bgcolor="#F8FAFC",
        xaxis=dict(visible=False, range=[-0.1, cols+0.1], fixedrange=True),
        yaxis=dict(visible=False, range=[-0.1, rows+0.1], fixedrange=True),
    )
    return fig


def make_replay_fig(env, trace):
    rows, cols = env.rows, env.cols
    cell = _cell_px(rows, cols)
    fig = go.Figure()

    # Use the episode's own grid if stored (curriculum may vary per episode)
    _trace_grid_raw = trace.get("grid")
    import numpy as _np_local
    if _trace_grid_raw is not None:
        _display_grid = _np_local.array(_trace_grid_raw, dtype=int)
    else:
        _display_grid = env.grid

    # Cell rewards: prefer trace-stored (accurate per episode), fall back to session state
    _trace_cr_raw = trace.get("cell_rewards") or {}
    _trace_cr = {
        tuple(int(x) for x in k.split(",")): float(v)
        for k, v in _trace_cr_raw.items()
    } if _trace_cr_raw else st.session_state.get("cell_rewards", {})

    # Start position stored in trace, or fall back to env
    _trace_start = trace.get("start_pos")
    _start_rc = tuple(_trace_start) if _trace_start is not None else getattr(env, "start_pos", None)

    for r in range(rows):
        for c in range(cols):
            t = int(_display_grid[r, c])
            y0 = rows - 1 - r

            # Mark start cell (stored as EMPTY in grid; overlay a Start-style fill)
            _is_start = _start_rc is not None and (r, c) == tuple(_start_rc)
            fill = CELL_FILL[5] if _is_start else CELL_FILL[t]
            border = CELL_LINE[5] if _is_start else CELL_LINE[t]

            fig.add_shape(
                type="rect", layer="below",
                x0=c + 0.05, y0=y0 + 0.05, x1=c + 0.95, y1=y0 + 0.95,
                fillcolor=fill,
                line=dict(color=border, width=1.5),
            )
            lbl = CELL_LBL[5] if _is_start else CELL_LBL[t]
            fg  = CELL_FG[5]  if _is_start else CELL_FG[t]
            if lbl:
                fig.add_annotation(
                    x=c + 0.5, y=y0 + 0.5, text=f"<b>{lbl}</b>",
                    showarrow=False,
                    font=dict(size=max(13, cell // 4), color=fg),
                    xanchor="center", yanchor="middle",
                )
            cr = _trace_cr.get((r, c), None)
            if cr is not None:
                badge_color = "#ECFDF5" if float(cr) >= 0 else "#FEF3F2"
                badge_border = "#86EFAC" if float(cr) >= 0 else "#FCA5A5"
                badge_fg = "#065F46" if float(cr) >= 0 else "#9B1C1C"
                fig.add_annotation(
                    x=c + 0.78, y=y0 + 0.22,
                    text=f"<span style='background:{badge_color};border:1px solid {badge_border};padding:2px 6px;border-radius:8px;color:{badge_fg};font-weight:600'>{float(cr):+.2f}</span>",
                    showarrow=False, font=dict(size=9), xanchor="center", yanchor="middle", opacity=0.95,
                )

    states = trace.get("states", [])
    actions = trace.get("actions", [])
    rewards = trace.get("rewards", [])
    xs, ys, customdata = [], [], []
    for idx, s in enumerate(states):
        r, c = env.state_to_pos(s)
        xs.append(c + 0.5)
        ys.append(rows - 1 - r + 0.5)
        if idx < len(actions):
            action_name = GridWorld.ACTION_NAMES[actions[idx]]
            reward_value = rewards[idx]
        else:
            action_name = "END"
            reward_value = 0.0
        customdata.append((f"{idx}", f"({r},{c})", action_name, f"{reward_value:+.2f}"))

    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines+markers",
        line=dict(color="#7c3aed", width=3),
        marker=dict(size=10, color="#a78bfa", line=dict(color="white", width=2)),
        hovertemplate="<b>Step %{customdata[0]}</b><br>Pos %{customdata[1]}<br>Action %{customdata[2]}<br>Reward %{customdata[3]}<extra></extra>",
        customdata=customdata,
    ))

    if xs:
        fig.add_annotation(
            x=xs[0], y=ys[0], text="Start",
            showarrow=True, arrowhead=2, arrowsize=1.2,
            ax=0, ay=-20, font=dict(color="#065f46", size=11),
        )
        fig.add_annotation(
            x=xs[-1], y=ys[-1], text="End",
            showarrow=True, arrowhead=2, arrowsize=1.2,
            ax=0, ay=20, font=dict(color="#b91c1c", size=11),
        )

    _gamma_r = getattr(env, "gamma", 0.99)
    total_reward = sum((_gamma_r ** t) * r for t, r in enumerate(rewards))
    _complexity = trace.get("complexity", 0.0)
    _cx_tag = (
        f" · <span style='color:#10B981'>complexity {_complexity:.1f}/10</span>"
        if _complexity > 0 else ""
    )
    _title_text = f"Episode replay — return {total_reward:+.3f} (γ={_gamma_r}){_cx_tag}"
    fig.update_layout(
        width=cols * cell, height=rows * cell,
        margin=dict(l=0, r=0, t=48, b=0),
        title=dict(text=_title_text, x=0.5, xanchor="center", font=dict(size=13)),
        plot_bgcolor="#F8FAFC", paper_bgcolor="#F8FAFC",
        xaxis=dict(visible=False, range=[-0.1, cols + 0.1], fixedrange=True),
        yaxis=dict(visible=False, range=[-0.1, rows + 0.1], fixedrange=True),
    )
    return fig


def _evenly_subsample_traces(traces, limit):
    n = len(traces)
    if n <= limit:
        return list(traces)
    if limit <= 1:
        return [traces[0]]
    indices = [round(i * (n - 1) / (limit - 1)) for i in range(limit)]
    return [traces[i] for i in indices]


def make_policy_fig(env, policy, values):
    rows, cols = env.rows, env.cols
    cell = _cell_px(rows, cols)

    valid = [float(values[s]) for s in range(env.n_states)
             if not env.is_wall(s) and not env.is_terminal(s)]
    vmin = min(valid) if valid else -1.0
    vmax = max(valid) if valid else  1.0
    if abs(vmax - vmin) < 1e-8:
        vmin -= 1; vmax += 1

    def val_color(v):
        t = max(0.0, min(1.0, (v - vmin) / (vmax - vmin)))
        stops = RDYLGN
        for i in range(len(stops) - 1):
            if t <= stops[i+1][0]:
                lo, hi = stops[i][0], stops[i+1][0]
                frac = (t - lo) / (hi - lo) if hi > lo else 0.0
                def _hex(c):
                    r1 = int(c[1:3], 16); g1 = int(c[3:5], 16); b1 = int(c[5:7], 16)
                    return r1, g1, b1
                r1,g1,b1 = _hex(stops[i][1])
                r2,g2,b2 = _hex(stops[i+1][1])
                rr = int(r1 + (r2-r1)*frac)
                gg = int(g1 + (g2-g1)*frac)
                bb = int(b1 + (b2-b1)*frac)
                return f"rgb({rr},{gg},{bb})"
        return RDYLGN[-1][1]

    fig = go.Figure()

    for r in range(rows):
        for c in range(cols):
            s   = r * cols + c
            t   = int(env.grid[r, c])
            y0  = rows - 1 - r
            cx, cy = c + 0.5, y0 + 0.5

            if env.is_wall(s):
                fill = "#1E293B"
            elif env.is_terminal(s):
                fill = CELL_FILL[GO] if t == GridWorld.GOAL else CELL_FILL[TR]
            else:
                fill = val_color(float(values[s]))

            fig.add_shape(
                type="rect", layer="below",
                x0=c+0.04, y0=y0+0.04, x1=c+0.96, y1=y0+0.96,
                fillcolor=fill,
                line=dict(color="#CBD5E1", width=0.8),
            )

            # Value label
            if not env.is_wall(s):
                fig.add_annotation(
                    x=cx, y=y0 + 0.22, text=f"{float(values[s]):.2f}",
                    showarrow=False, font=dict(size=8, color="#374151"),
                    xanchor="center", yanchor="middle",
                )

            # Terminal label
            if env.is_terminal(s):
                lbl = "G" if t == GridWorld.GOAL else "X"
                color = "#166534" if t == GridWorld.GOAL else "#991B1B"
                fig.add_annotation(
                    x=cx, y=cy + 0.12, text=f"<b>{lbl}</b>",
                    showarrow=False, font=dict(size=max(14, cell//4), color=color),
                    xanchor="center", yanchor="middle",
                )
                # Terminal reward badge
                reward = env.config.goal_reward if t == GridWorld.GOAL else env.config.trap_reward
                badge_color = "#ECFDF5" if reward >= 0 else "#FEF3F2"
                badge_border = "#86EFAC" if reward >= 0 else "#FCA5A5"
                badge_fg = "#065F46" if reward >= 0 else "#9B1C1C"
                fig.add_annotation(
                    x=cx, y=y0 + 0.22,
                    text=f"<span style='background:{badge_color};border:1px solid {badge_border};padding:2px 6px;border-radius:8px;color:{badge_fg};font-weight:600'>{reward:+.2f}</span>",
                    showarrow=False, font=dict(size=9), xanchor="center", yanchor="middle", opacity=0.95,
                )

            # Slippery marker
            if t == GridWorld.SLIPPERY and not env.is_terminal(s):
                fig.add_annotation(
                    x=cx, y=y0 + 0.78, text="≈",
                    showarrow=False, font=dict(size=10, color="#0369A1"),
                    xanchor="center", yanchor="middle",
                )

            # Start marker
            if (r, c) == env.start_pos and not env.is_wall(s):
                fig.add_annotation(
                    x=c+0.14, y=y0+0.88, text="S",
                    showarrow=False, font=dict(size=8, color="#64748B", style="italic"),
                )

            # Policy arrow
            if not env.is_wall(s) and not env.is_terminal(s):
                a  = int(policy[s])
                dx = ARROW_DX[a] * 0.32
                dy = ARROW_DY[a] * 0.32
                fig.add_annotation(
                    x=cx+dx, y=cy+dy, ax=cx-dx, ay=cy-dy,
                    xref="x", yref="y", axref="x", ayref="y",
                    showarrow=True, arrowhead=3, arrowsize=1.2,
                    arrowwidth=1.8, arrowcolor="#312E81",
                )
                symbol = ["↑", "→", "↓", "←"][a]
                fig.add_annotation(
                    x=cx, y=y0 + 0.75, text=f"<b>{symbol}</b>",
                    showarrow=False, font=dict(size=max(14, cell // 3), color="#312E81"),
                    xanchor="center", yanchor="middle",
                )

    # Colour-bar via invisible scatter
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(
            colorscale=RDYLGN, cmin=vmin, cmax=vmax, showscale=True,
            colorbar=dict(
                title=dict(text="Value", side="right"),
                thickness=12, len=0.85, tickfont=dict(size=9),
                bgcolor="rgba(255,255,255,0.8)",
            ),
            color=[0],
        ),
        hoverinfo="none",
    ))

    fig.update_layout(
        width=cols*cell, height=rows*cell,
        margin=dict(l=0, r=55, t=0, b=0),
        showlegend=False,
        plot_bgcolor="#F8FAFC", paper_bgcolor="#F8FAFC",
        xaxis=dict(visible=False, range=[-0.1, cols+0.1], fixedrange=True),
        yaxis=dict(visible=False, range=[-0.1, rows+0.1], fixedrange=True),
    )
    return fig


def make_visit_fig(env, visits):
    """Render a grid showing integer visit counts per cell."""
    rows, cols = env.rows, env.cols
    cell = _cell_px(rows, cols)
    fig = go.Figure()

    maxv = int(max(1, max(int(v) for v in visits)))
    def heat_color(v):
        t = v / maxv
        r = int(255 * (1 - t))
        g = int(255 * (1 - t*0.6))
        b = 255
        return f"rgb({r},{g},{b})"

    for r in range(rows):
        for c in range(cols):
            s = r * cols + c
            y0 = rows - 1 - r
            t = env.grid[r, c]
            fill = "#1E293B" if env.is_wall(s) else (CELL_FILL[GO] if env.is_terminal(s) and int(t)==GridWorld.GOAL else CELL_FILL[TR] if env.is_terminal(s) else heat_color(int(visits[s])))
            fig.add_shape(
                type="rect", layer="below",
                x0=c+0.04, y0=y0+0.04, x1=c+0.96, y1=y0+0.96,
                fillcolor=fill,
                line=dict(color="#CBD5E1", width=0.8),
            )
            # visit count label and terminal reward (if terminal)
            if not env.is_wall(s):
                if env.is_terminal(s):
                    # Show terminal label and reward
                    lbl = "G" if t == GridWorld.GOAL else "X"
                    color = "#166534" if t == GridWorld.GOAL else "#991B1B"
                    fig.add_annotation(
                        x=c+0.5, y=y0+0.65, text=f"<b>{lbl}</b>",
                        showarrow=False, font=dict(size=max(12, cell//4), color=color),
                        xanchor="center", yanchor="middle",
                    )
                    # Terminal reward badge
                    reward = env.config.goal_reward if t == GridWorld.GOAL else env.config.trap_reward
                    badge_color = "#ECFDF5" if reward >= 0 else "#FEF3F2"
                    badge_border = "#86EFAC" if reward >= 0 else "#FCA5A5"
                    badge_fg = "#065F46" if reward >= 0 else "#9B1C1C"
                    fig.add_annotation(
                        x=c+0.5, y=y0+0.30,
                        text=f"<span style='background:{badge_color};border:1px solid {badge_border};padding:2px 6px;border-radius:8px;color:{badge_fg};font-weight:600'>{reward:+.2f}</span>",
                        showarrow=False, font=dict(size=8), xanchor="center", yanchor="middle", opacity=0.95,
                    )
                    # visit count in smaller font at bottom
                    fig.add_annotation(
                        x=c+0.5, y=y0+0.08, text=f"visits: {int(visits[s])}",
                        showarrow=False, font=dict(size=7, color="#64748B"),
                        xanchor="center", yanchor="middle",
                    )
                else:
                    # Non-terminal: show visit count
                    fig.add_annotation(
                        x=c+0.5, y=y0+0.5, text=str(int(visits[s])),
                        showarrow=False, font=dict(size=10, color="#0F172A"),
                        xanchor="center", yanchor="middle",
                    )

    fig.update_layout(
        width=cols*cell, height=rows*cell,
        margin=dict(l=0, r=0, t=0, b=0), showlegend=False,
        plot_bgcolor="#F8FAFC", paper_bgcolor="#F8FAFC",
        xaxis=dict(visible=False, range=[-0.1, cols+0.1], fixedrange=True),
        yaxis=dict(visible=False, range=[-0.1, rows+0.1], fixedrange=True),
    )
    return fig


def make_curves_fig(rewards, lengths, algo, off_policy_steps=None):
    n = len(rewards)
    w = max(1, min(50, n // 10))

    def smooth(d):
        return np.convolve(d, np.ones(w) / w, mode="valid").tolist()

    xs = list(range(n))
    xs_sm = list(range(w - 1, n))
    include_off_policy = off_policy_steps is not None
    cols = 3 if include_off_policy else 2
    subplot_titles = ("Reward per Episode", "Episode Length")
    if include_off_policy:
        subplot_titles = ("Reward per Episode", "Episode Length", "Off-policy Steps")

    fig = make_subplots(
        rows=1, cols=cols, horizontal_spacing=0.08,
        subplot_titles=subplot_titles,
    )

    fig.add_trace(go.Scatter(
        x=xs, y=rewards, mode="lines", name="raw",
        line=dict(color="rgba(99,102,241,0.18)", width=1), showlegend=False,
    ), row=1, col=1)
    if n >= w:
        fig.add_trace(go.Scatter(
            x=xs_sm, y=smooth(rewards), mode="lines", name="smoothed",
            line=dict(color="#6366F1", width=2.5), showlegend=False,
            fill="tozeroy", fillcolor="rgba(99,102,241,0.07)",
        ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=xs, y=lengths, mode="lines", name="raw",
        line=dict(color="rgba(239,68,68,0.18)", width=1), showlegend=False,
    ), row=1, col=2)
    if n >= w:
        fig.add_trace(go.Scatter(
            x=xs_sm, y=smooth(lengths), mode="lines", name="smoothed",
            line=dict(color="#EF4444", width=2.5), showlegend=False,
            fill="tozeroy", fillcolor="rgba(239,68,68,0.07)",
        ), row=1, col=2)

    if include_off_policy:
        fig.add_trace(go.Scatter(
            x=xs, y=off_policy_steps, mode="lines", name="raw",
            line=dict(color="rgba(168,85,247,0.18)", width=1), showlegend=False,
        ), row=1, col=3)
        if n >= w:
            fig.add_trace(go.Scatter(
                x=xs_sm, y=smooth(off_policy_steps), mode="lines", name="smoothed",
                line=dict(color="#a855f7", width=2.5), showlegend=False,
                fill="tozeroy", fillcolor="rgba(168,85,247,0.07)",
            ), row=1, col=3)

    fig.update_xaxes(title_text="Episode", gridcolor="#F1F5F9", zeroline=False)
    fig.update_yaxes(gridcolor="#F1F5F9", zeroline=False)
    fig.update_yaxes(title_text="Total Reward", row=1, col=1)
    fig.update_yaxes(title_text="Steps", row=1, col=2)
    if include_off_policy:
        fig.update_yaxes(title_text="Off-policy steps", row=1, col=3)
    fig.update_layout(
        height=300, margin=dict(l=40, r=20, t=35, b=40),
        plot_bgcolor="#FAFBFF", paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", size=11, color="#374151"),
        title=dict(
            text=f"<b>{algo}</b> — Training Progress",
            x=0.5, font=dict(size=13, color="#0F172A"),
        ),
    )
    return fig


def make_complexity_fig(episode_complexities, algo):
    """Rolling-average complexity curve (1–10 scale) over training episodes."""
    n = len(episode_complexities)
    if n == 0:
        return None

    w = max(1, min(50, n // 10))
    xs = list(range(n))

    def smooth(d):
        return np.convolve(d, np.ones(w) / w, mode="valid").tolist()

    xs_sm = list(range(w - 1, n))
    raw_color  = "rgba(16,185,129,0.20)"
    line_color = "#10B981"
    fill_color = "rgba(16,185,129,0.08)"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=episode_complexities,
        mode="lines", name="per episode",
        line=dict(color=raw_color, width=1),
        showlegend=False,
    ))
    if n >= w:
        fig.add_trace(go.Scatter(
            x=xs_sm, y=smooth(episode_complexities),
            mode="lines", name=f"rolling avg (w={w})",
            line=dict(color=line_color, width=2.5),
            fill="tozeroy", fillcolor=fill_color,
            showlegend=True,
        ))

    fig.update_xaxes(title_text="Episode", gridcolor="#F1F5F9", zeroline=False)
    fig.update_yaxes(
        title_text="Complexity (1–10)", gridcolor="#F1F5F9",
        zeroline=False, range=[0, 10.5],
    )
    fig.update_layout(
        height=280, margin=dict(l=40, r=20, t=45, b=40),
        plot_bgcolor="#FAFBFF", paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", size=11, color="#374151"),
        title=dict(
            text=f"<b>{algo}</b> — Curriculum Complexity over Training",
            x=0.5, font=dict(size=13, color="#0F172A"),
        ),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.18),
    )
    return fig


def make_timing_fig(episode_times, algo):
    """Visualize time per episode and cumulative training time."""
    n = len(episode_times)
    w = max(1, min(50, n // 10))

    def smooth(d):
        return np.convolve(d, np.ones(w) / w, mode="valid").tolist()

    xs = list(range(n))
    xs_sm = list(range(w - 1, n))
    cumulative_time = np.cumsum(episode_times).tolist()

    fig = make_subplots(
        rows=1, cols=2, horizontal_spacing=0.08,
        subplot_titles=("Time per Episode (ms)", "Cumulative Training Time (s)"),
    )

    # Time per episode
    episode_times_ms = [t * 1000 for t in episode_times]  # Convert to milliseconds
    fig.add_trace(go.Scatter(
        x=xs, y=episode_times_ms, mode="lines", name="raw",
        line=dict(color="rgba(168,85,247,0.18)", width=1), showlegend=False,
    ), row=1, col=1)
    if n >= w:
        smoothed_times = [t * 1000 for t in smooth(episode_times)]
        fig.add_trace(go.Scatter(
            x=xs_sm, y=smoothed_times, mode="lines", name="smoothed",
            line=dict(color="#A855F7", width=2.5), showlegend=False,
            fill="tozeroy", fillcolor="rgba(168,85,247,0.07)",
        ), row=1, col=1)

    # Cumulative time
    cumulative_time_s = [t for t in cumulative_time]  # Already in seconds
    fig.add_trace(go.Scatter(
        x=xs, y=cumulative_time_s, mode="lines", name="cumulative",
        line=dict(color="#14B8A6", width=2.5), showlegend=False,
        fill="tozeroy", fillcolor="rgba(20,184,166,0.07)",
    ), row=1, col=2)

    fig.update_xaxes(title_text="Episode", gridcolor="#F1F5F9", zeroline=False)
    fig.update_yaxes(gridcolor="#F1F5F9", zeroline=False)
    fig.update_yaxes(title_text="Time (ms)", row=1, col=1)
    fig.update_yaxes(title_text="Time (s)", row=1, col=2)
    fig.update_layout(
        height=270, margin=dict(l=40, r=20, t=35, b=40),
        plot_bgcolor="#FAFBFF", paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", size=11, color="#374151"),
        title=dict(
            text=f"<b>{algo}</b> — Training Time Analysis",
            x=0.5, font=dict(size=13, color="#0F172A"),
        ),
    )
    return fig


def make_timing_fig(episode_times, algo):
    """Visualize time per episode and cumulative training time."""
    n = len(episode_times)
    w = max(1, min(50, n // 10))

    def smooth(d):
        return np.convolve(d, np.ones(w) / w, mode="valid").tolist()

    xs = list(range(n))
    xs_sm = list(range(w - 1, n))
    cumulative_time = np.cumsum(episode_times).tolist()

    fig = make_subplots(
        rows=1, cols=2, horizontal_spacing=0.08,
        subplot_titles=("Time per Episode (ms)", "Cumulative Training Time (s)"),
    )

    # Time per episode
    episode_times_ms = [t * 1000 for t in episode_times]  # Convert to milliseconds
    fig.add_trace(go.Scatter(
        x=xs, y=episode_times_ms, mode="lines", name="raw",
        line=dict(color="rgba(168,85,247,0.18)", width=1), showlegend=False,
    ), row=1, col=1)
    if n >= w:
        smoothed_times = [t * 1000 for t in smooth(episode_times)]
        fig.add_trace(go.Scatter(
            x=xs_sm, y=smoothed_times, mode="lines", name="smoothed",
            line=dict(color="#A855F7", width=2.5), showlegend=False,
            fill="tozeroy", fillcolor="rgba(168,85,247,0.07)",
        ), row=1, col=1)

    # Cumulative time
    cumulative_time_s = [t for t in cumulative_time]  # Already in seconds
    fig.add_trace(go.Scatter(
        x=xs, y=cumulative_time_s, mode="lines", name="cumulative",
        line=dict(color="#14B8A6", width=2.5), showlegend=False,
        fill="tozeroy", fillcolor="rgba(20,184,166,0.07)",
    ), row=1, col=2)

    fig.update_xaxes(title_text="Episode", gridcolor="#F1F5F9", zeroline=False)
    fig.update_yaxes(gridcolor="#F1F5F9", zeroline=False)
    fig.update_yaxes(title_text="Time (ms)", row=1, col=1)
    fig.update_yaxes(title_text="Time (s)", row=1, col=2)
    fig.update_layout(
        height=270, margin=dict(l=40, r=20, t=35, b=40),
        plot_bgcolor="#FAFBFF", paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", size=11, color="#374151"),
        title=dict(
            text=f"<b>{algo}</b> — Training Time Analysis",
            x=0.5, font=dict(size=13, color="#0F172A"),
        ),
    )
    return fig


def valid_actions(env, state):
    """Return legal actions that do not bump into walls or leave the grid."""
    if env.is_terminal(state):
        return []
    r, c = divmod(state, env.cols)
    valid = [a for a in env.ACTIONS if env._step_pos(r, c, a) != (r, c)]
    return valid or env.ACTIONS


def format_duration(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def make_epsilon_fig(epsilon_values, algo):
    """Visualize epsilon (exploration rate) decay over training."""
    n = len(epsilon_values)
    if n == 0:
        return go.Figure()
    
    xs = list(range(n))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=epsilon_values, mode="lines", name="epsilon",
        line=dict(color="#F59E0B", width=2.5),
        fill="tozeroy", fillcolor="rgba(245,158,11,0.1)",
    ))
    
    fig.update_xaxes(title_text="Episode", gridcolor="#F1F5F9", zeroline=False)
    fig.update_yaxes(title_text="Epsilon (ε)", gridcolor="#F1F5F9", zeroline=False)
    fig.update_layout(
        height=270, margin=dict(l=40, r=20, t=35, b=40),
        plot_bgcolor="#FAFBFF", paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", size=11, color="#374151"),
        title=dict(
            text=f"<b>{algo}</b> — Exploration Rate Decay",
            x=0.5, font=dict(size=13, color="#0F172A"),
        ),
        hovermode="x unified",
    )
    return fig


# ── Observation helpers ───────────────────────────────────────────────

def _obs_dim(env, n_neighbors, use_goal_dist):
    """Input vector length for DQN with the given observation settings.

    n_neighbors=0, use_goal_dist=False → one-hot over n_states (legacy).
    Otherwise → [row/rows, col/cols, <neighbor cell types>, [dx, dy]].
    """
    if n_neighbors == 0 and not use_goal_dist:
        return env.n_states
    n_cells = (2 * n_neighbors + 1) ** 2 - 1 if n_neighbors > 0 else 0
    return 2 + n_cells + (2 if use_goal_dist else 0)


def _build_obs(env, state, n_neighbors, use_goal_dist):
    """Build the flat observation vector for DQN.

    n_neighbors=0, use_goal_dist=False → one-hot (backwards-compatible).
    Neighbor cells are ordered row-major around the agent, skipping (0,0).
    Out-of-bounds cells are encoded as WALL (1/4 = 0.25).
    Goal distance components: dx = (goal_col - col)/(cols-1),
                               dy = (goal_row - row)/(rows-1).
    """
    if n_neighbors == 0 and not use_goal_dist:
        v = [0.0] * env.n_states
        v[state] = 1.0
        return v

    rows, cols = env.rows, env.cols
    r, c = divmod(state, cols)
    obs = [r / max(1, rows - 1), c / max(1, cols - 1)]

    if n_neighbors > 0:
        for dr in range(-n_neighbors, n_neighbors + 1):
            for dc in range(-n_neighbors, n_neighbors + 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    ct = int(env.grid[nr, nc])
                else:
                    ct = GridWorld.WALL
                obs.append(ct / 4.0)

    if use_goal_dist:
        best_d = float("inf")
        goal_r, goal_c = r, c
        for gr in range(rows):
            for gc in range(cols):
                gs = gr * cols + gc
                if env.is_terminal(gs) and int(env.grid[gr, gc]) == GridWorld.GOAL:
                    d = abs(gr - r) + abs(gc - c)
                    if d < best_d:
                        best_d = d
                        goal_r, goal_c = gr, gc
        dx = (goal_c - c) / max(1, cols - 1)
        dy = (goal_r - r) / max(1, rows - 1)
        obs.extend([dx, dy])

    return obs


# ── Curriculum helpers ────────────────────────────────────────────────

def _generate_curriculum_setups(n_setups, rows, cols,
                                 step_reward=0.0, slip_prob=0.3, gamma=0.95,
                                 hints=None,
                                 min_complexity=1.0, max_complexity=10.0):
    """Return a list of `n_setups` layout dicts spanning complexity min→max.

    ``hints`` is the dict produced by :func:`_infer_layout_directives_from_prompt`
    and is forwarded to every :func:`_generate_random_layout` call so that
    constraints such as *no traps* or *one goal* are respected throughout the
    entire curriculum.
    """
    hints = hints or {}
    min_complexity = max(1.0, min(float(min_complexity), 10.0))
    max_complexity = max(min_complexity, min(float(max_complexity), 10.0))
    setups = []
    for i in range(n_setups):
        frac = i / max(1, n_setups - 1)
        complexity = min_complexity + frac * (max_complexity - min_complexity)
        grid, cell_rewards = _generate_random_layout(
            max(1, round(complexity)), rows, cols, hints=hints
        )
        layout = _grid_to_layout(grid, rows, cols)
        setups.append({
            "index": i,
            "complexity": round(complexity, 2),
            "rows": rows,
            "cols": cols,
            "layout": layout,
            "cell_rewards": {f"{r},{c}": float(v) for (r, c), v in cell_rewards.items()},
            "step_reward": step_reward,
            "slip_prob": slip_prob,
            "gamma": gamma,
        })
    return setups


def _bfs_goal_distances(env) -> list[float]:
    """BFS from all goal cells outward.  Returns a list indexed by state id
    where each value is the shortest-path distance to the nearest goal.
    Walls are impassable; unreachable states get ``inf``.
    """
    from collections import deque
    dist = [float("inf")] * env.n_states
    q = deque()
    for s in range(env.n_states):
        r, c = divmod(s, env.cols)
        if int(env.grid[r, c]) == env.GOAL:
            dist[s] = 0.0
            q.append(s)
    while q:
        s = q.popleft()
        r, c = divmod(s, env.cols)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < env.rows and 0 <= nc < env.cols:
                ns = nr * env.cols + nc
                if dist[ns] == float("inf") and int(env.grid[nr, nc]) != env.WALL:
                    dist[ns] = dist[s] + 1.0
                    q.append(ns)
    return dist


def _potential_shaping(s, ns, dist, gamma, alpha):
    """Potential-based shaping bonus F(s,s') = γ·Φ(s') − Φ(s).

    Φ(s) = −alpha · dist(s, goal)  →  closer = higher potential.
    Returns 0 if either state is unreachable (dist = inf).
    """
    phi_s  = -alpha * dist[s]  if dist[s]  != float("inf") else 0.0
    phi_ns = -alpha * dist[ns] if dist[ns] != float("inf") else 0.0
    return gamma * phi_ns - phi_s


def _setup_to_env(setup):
    """Build a GridWorld from a curriculum setup dict."""
    cr = {tuple(map(int, k.split(","))): float(v)
          for k, v in setup.get("cell_rewards", {}).items()}
    cfg = GridConfig(
        rows=setup["rows"], cols=setup["cols"], layout=setup["layout"],
        step_reward=float(setup.get("step_reward", 0.0)),
        goal_reward=1.0, trap_reward=-1.0, cell_rewards=cr,
        slippery_slip_prob=float(setup.get("slip_prob", 0.3)),
        gamma=float(setup.get("gamma", 0.95)),
    )
    return GridWorld(cfg)


def _run_dqn_test(q_net, test_setups, complexity_target, n_games, n_nb, use_gd, max_steps=300):
    """Run the trained DQN greedily on test setups near *complexity_target*.

    Returns
    -------
    dict with keys: win_rate, avg_reward, avg_steps, n_games, n_setups_used, per_game
    """
    # Find the setups closest to the requested complexity (window ±1.5, widen if too few)
    window = 1.5
    nearby = [s for s in test_setups if abs(s["complexity"] - complexity_target) <= window]
    if not nearby:
        nearby = sorted(test_setups, key=lambda s: abs(s["complexity"] - complexity_target))[:max(5, len(test_setups)//4)]

    per_game = []
    for _ in range(n_games):
        setup = random.choice(nearby)
        ep_env = _setup_to_env(setup)
        s = ep_env.reset()
        total_r = 0.0
        steps = 0
        won = False
        for _ in range(max_steps):
            obs = _build_obs(ep_env, s, n_nb, use_gd)
            with torch.no_grad():
                action = int(q_net(torch.tensor([obs], dtype=torch.float32)).argmax(dim=1).item())
            s, r, done = ep_env.step(action)
            total_r += r
            steps += 1
            if done:
                sr, sc = divmod(s, ep_env.cols)
                won = ep_env.grid[sr, sc] == ep_env.GOAL
                break
        per_game.append({
            "reward": total_r,
            "steps": steps,
            "won": won,
            "complexity": setup["complexity"],
        })

    wins = sum(1 for g in per_game if g["won"])
    return {
        "win_rate": wins / len(per_game) if per_game else 0.0,
        "avg_reward": float(np.mean([g["reward"] for g in per_game])) if per_game else 0.0,
        "avg_steps": float(np.mean([g["steps"] for g in per_game])) if per_game else 0.0,
        "n_games": len(per_game),
        "n_setups_used": len(nearby),
        "per_game": per_game,
    }


def _auto_save_run(training_summary: dict, res: dict) -> str | None:
    """Persist a training run (setup + key metrics) to the results/ folder.

    Filename:  YYYY-MM-DD_HH-MM-SS_<Algo>_<R>x<C>.json
    Returns the file path string on success, or None on failure.
    """
    try:
        folder = pathlib.Path("results")
        folder.mkdir(exist_ok=True)

        ts_str  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        algo_fs = (training_summary.get("algo") or "unknown").replace(" ", "_")
        rows    = training_summary.get("rows", 0)
        cols    = training_summary.get("cols", 0)
        fname   = f"{ts_str}_{algo_fs}_{rows}x{cols}.json"
        fpath   = folder / fname

        rews     = res.get("rewards") or []
        ep_times = res.get("episode_times") or []

        # Downsample rewards to ≤ 500 points for compact storage
        stride   = max(1, len(rews) // 500)
        curve    = [round(float(r), 4) for r in rews[::stride]]

        payload = {
            "saved_at":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "filename":        fname,
            "training_summary": {
                k: v for k, v in training_summary.items() if k != "hp_rows"
            },
            "results": {
                "episodes_run":          len(rews),
                "avg_reward_last_100":   round(float(np.mean(rews[-100:])),  4) if rews else None,
                "avg_reward_last_20":    round(float(np.mean(rews[-20:])),   4) if rews else None,
                "avg_reward_first_100":  round(float(np.mean(rews[:100])),   4) if len(rews) >= 100 else None,
                "avg_reward_mid_third":  round(float(np.mean(rews[len(rews)//3 : 2*len(rews)//3])), 4)
                                         if len(rews) > 60 else None,
                "peak_reward":           round(float(max(rews)),   4) if rews else None,
                "final_reward":          round(float(rews[-1]),    4) if rews else None,
                "total_training_time_s": round(float(sum(ep_times)), 2) if ep_times else None,
                "rewards_curve_downsampled": curve,
            },
        }

        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)

        return str(fpath)
    except Exception:
        return None


# ─── Valid parameter keys the analysis can suggest ───────────────────────────
_SUGGESTION_PARAM_TYPES: dict[str, type] = {
    "alpha":            float,
    "epsilon":          float,
    "epsilon_decay":    float,
    "gamma":            float,
    "max_steps":        int,
    "exploring_starts": bool,
    "episodes":         int,
    "step_rew":         float,
    "slip_prob":        float,
    "dqn_lr":           float,
    "dqn_hidden_str":   str,
    "dqn_batch_size":   int,
    "dqn_buffer_size":  int,
    "dqn_target_update":int,
    "dqn_epsilon":      float,
    "dqn_epsilon_decay":float,
    "dqn_epsilon_min":  float,
    "dqn_episodes":     int,
    "obs_n_neighbors":  int,
    "obs_use_goal_dist":bool,
}


def _call_claude_analysis(training_summary: dict, res: dict) -> dict:
    """Ask Claude to diagnose a training run and return structured suggestions.

    Returns a dict:
        {"issues": [str, ...], "suggestions": [{"text", "explanation", "param", "value"}, ...]}
    """
    rews     = res.get("rewards") or []
    ep_times = res.get("episode_times") or []
    n_ep     = len(rews)
    algo     = training_summary.get("algo", "unknown")
    cc       = training_summary.get("cell_counts", {})

    def _stat(lst, lo, hi):
        seg = lst[lo:hi]
        return round(float(np.mean(seg)), 3) if seg else None

    first_avg = _stat(rews, 0,           min(100, n_ep))
    mid_avg   = _stat(rews, n_ep // 3,   2 * n_ep // 3)
    last_avg  = _stat(rews, max(0, n_ep - 100), n_ep)
    last20    = _stat(rews, max(0, n_ep - 20),  n_ep)
    peak      = round(float(max(rews)), 3) if rews else None

    # Build algo-specific parameter block
    if algo == "DQN":
        algo_params = (
            f"  lr={training_summary.get('dqn_lr')}  hidden={training_summary.get('dqn_hidden')}\n"
            f"  batch={training_summary.get('dqn_batch')}  buffer={training_summary.get('dqn_buffer'):,}\n"
            f"  epsilon: {training_summary.get('dqn_epsilon')} decay {training_summary.get('dqn_epsilon_decay')} "
            f"min {training_summary.get('dqn_epsilon_min')}\n"
            f"  obs: n_neighbors={training_summary.get('obs_n_neighbors')}  "
            f"goal_dist={training_summary.get('obs_use_goal_dist')}  "
            f"input_dim={training_summary.get('obs_input_dim')}\n"
            f"  curriculum: {training_summary.get('curriculum_active')}  "
            f"method={training_summary.get('curriculum_method')}  "
            f"threshold={training_summary.get('curriculum_threshold')}"
        )
    elif algo in ("Q-Learning", "SARSA"):
        algo_params = (
            f"  alpha={training_summary.get('alpha')}  "
            f"epsilon: {training_summary.get('epsilon')} decay {training_summary.get('epsilon_decay')} min 0.01"
        )
    else:
        algo_params = "  (Policy Iteration — no hyperparameters)"

    param_list = "\n".join(
        f'  "{k}" ({t.__name__}): {desc}'
        for k, t, desc in [
            ("alpha",            float, "Q/SARSA learning rate (0.001–0.5)"),
            ("epsilon_decay",    float, "Q/SARSA epsilon decay per episode (0.99–0.9999)"),
            ("epsilon",          float, "Q/SARSA initial epsilon (0.5–1.0)"),
            ("gamma",            float, "discount factor (0.8–0.999)"),
            ("max_steps",        int,   "max steps per episode (50–2000)"),
            ("exploring_starts", bool,  "randomise start state each episode"),
            ("episodes",         int,   "Q/SARSA episodes (500–20000)"),
            ("step_rew",         float, "step penalty, negative encourages efficiency (-1.0–0.0)"),
            ("dqn_lr",           float, "DQN learning rate (0.00001–0.01)"),
            ("dqn_hidden_str",   str,   'DQN hidden layers e.g. "128,64"'),
            ("dqn_batch_size",   int,   "DQN mini-batch size (32–512)"),
            ("dqn_buffer_size",  int,   "DQN replay buffer size (1000–200000)"),
            ("dqn_target_update",int,   "DQN target-net update interval in steps (10–1000)"),
            ("dqn_epsilon",      float, "DQN initial epsilon (0.5–1.0)"),
            ("dqn_epsilon_decay",float, "DQN epsilon decay per episode (0.99–0.9999)"),
            ("dqn_epsilon_min",  float, "DQN minimum epsilon (0.001–0.2)"),
            ("dqn_episodes",     int,   "DQN episodes (500–20000)"),
            ("obs_n_neighbors",  int,   "DQN observation neighbor radius 0=one-hot 1-3=local patch"),
            ("obs_use_goal_dist",bool,  "DQN include goal dx/dy in observation"),
        ]
        if algo == "DQN" or k in ("alpha","epsilon_decay","epsilon","gamma","max_steps",
                                   "exploring_starts","episodes","step_rew")
    )

    prompt = f"""You are an expert reinforcement learning engineer. Analyse the following training run and return ONLY a valid JSON object — no explanation, no markdown fences, no extra text.

=== TRAINING SETUP ===
Algorithm : {algo}
Grid      : {training_summary.get('rows')}×{training_summary.get('cols')} ({training_summary.get('rows',0)*training_summary.get('cols',0)} states)
Walls={cc.get('walls',0)}  Goals={cc.get('goals',0)}  Traps={cc.get('traps',0)}  Slippery={cc.get('slippery',0)}
Step reward={training_summary.get('step_rew')}  Goal reward={training_summary.get('goal_rew')}  Trap reward={training_summary.get('trap_rew')}
Gamma={training_summary.get('gamma')}  Slip prob={training_summary.get('slip_prob')}
Max steps/ep={training_summary.get('max_steps')}  Exploring starts={training_summary.get('exploring_starts')}
Algorithm params:
{algo_params}

=== RESULTS ===
Episodes run   : {n_ep:,}
Avg reward (first 100 ep) : {first_avg}
Avg reward (middle third) : {mid_avg}
Avg reward (last 100 ep)  : {last_avg}
Avg reward (last 20 ep)   : {last20}
Peak reward               : {peak}
Total training time       : {round(sum(ep_times),1) if ep_times else 'unknown'} s

=== OUTPUT FORMAT (strict) ===
{{
  "issues": ["<concise issue description>", ...],
  "suggestions": [
    {{
      "text": "<short label ≤ 8 words>",
      "explanation": "<one sentence why this helps>",
      "param": "<exact key from list below>",
      "value": <new value, correct type>
    }},
    ...
  ]
}}

=== VALID PARAM KEYS ===
Only use keys from this list (only suggest params applicable to {algo}):
{param_list}

Provide 2–5 specific, concrete, actionable suggestions based on the observed training behaviour. Think about: convergence speed, exploration/exploitation balance, catastrophic forgetting, reward shaping, and network capacity."""

    raw = _call_claude(prompt, max_tokens=1200)

    # Strip markdown fences if present
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        # Try to extract the first {...} block
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            result = json.loads(m.group())
        else:
            raise ValueError(f"Claude returned non-JSON output: {raw[:300]}")

    # Coerce value types
    for sug in result.get("suggestions", []):
        p = sug.get("param")
        v = sug.get("value")
        if p in _SUGGESTION_PARAM_TYPES and v is not None:
            try:
                sug["value"] = _SUGGESTION_PARAM_TYPES[p](v)
            except (TypeError, ValueError):
                pass

    return result


def _call_claude_chat(
    messages: list[dict],
    system: str = "",
    max_tokens: int = 800,
) -> str:
    """Send a multi-turn conversation to Claude and return the assistant reply text.

    `messages` is a list of {"role": "user"|"assistant", "content": str} dicts.
    `system` is an optional system-prompt string for context injection.
    """
    api_key = _get_claude_api_key()
    if not api_key:
        raise RuntimeError(
            "Claude API key is not configured. Set CLAUDE_API_KEY or Streamlit secrets."
        )

    url = "https://api.anthropic.com/v1/messages"
    model = None
    try:
        model = st.session_state.get("CLAUDE_MODEL")
    except Exception:
        model = None
    if not model:
        model = _get_claude_model_env() or "claude-opus-4-8"

    # Convert our message list to the API format
    api_messages = [
        {"role": m["role"], "content": [{"type": "text", "text": m["content"]}]}
        for m in messages
    ]

    payload: dict = {
        "model": model,
        "messages": api_messages,
        "max_tokens": max_tokens,
    }
    if system:
        payload["system"] = system

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    import requests as _req
    resp = _req.post(url, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data["content"][0]["text"]


def _render_training_summary_card(s: dict, res: dict | None) -> None:
    """Render a complete, styled training-setup card in the current Streamlit column."""

    def _section(title: str, rows_kv: list) -> str:
        rows_html = "".join(
            f'<tr>'
            f'<td style="color:#64748b;padding:2px 10px 2px 0;white-space:nowrap;'
            f'font-size:0.75rem;vertical-align:top">{k}</td>'
            f'<td style="font-weight:600;color:#1e293b;font-size:0.75rem">{v}</td>'
            f'</tr>'
            for k, v in rows_kv
        )
        return (
            f'<div style="margin-bottom:12px">'
            f'<div style="font-size:0.63rem;font-weight:700;text-transform:uppercase;'
            f'letter-spacing:0.08em;color:#6366f1;margin-bottom:3px">{title}</div>'
            f'<table style="border-collapse:collapse;width:100%">{rows_html}</table>'
            f'</div><div style="border-top:1px solid #f1f5f9;margin-bottom:10px"></div>'
        )

    algo  = s.get("algo", "—")
    rows  = s.get("rows", 0)
    cols  = s.get("cols", 0)
    parts = []

    # ── Algorithm ──────────────────────────────────────────────────────
    parts.append(_section("🧠 Algorithm", [
        ("Algorithm",        algo),
        ("Mode",             s.get("run_mode", "Full")),
        ("Episodes",         f"{s.get('total_ep', 0):,}"),
        ("Max steps / ep",   str(s.get("max_steps", "—"))),
        ("Exploring starts", "✓" if s.get("exploring_starts") else "✗"),
    ]))

    # ── Grid ───────────────────────────────────────────────────────────
    cc = s.get("cell_counts", {})
    parts.append(_section("📐 Grid", [
        ("Size",           f"{rows} × {cols}  ({rows * cols} states)"),
        ("Walls",          str(cc.get("walls",    "—"))),
        ("Goals",          str(cc.get("goals",    "—"))),
        ("Traps",          str(cc.get("traps",    "—"))),
        ("Slippery cells", str(cc.get("slippery", "—"))),
        ("Custom rewards", str(s.get("n_custom_cell_rewards", 0))),
    ]))

    # ── Rewards & Dynamics ─────────────────────────────────────────────
    parts.append(_section("🎯 Rewards & Dynamics", [
        ("Step reward",      f"{s.get('step_rew', 0):+.3f}"),
        ("Goal reward",      f"{s.get('goal_rew', 0):+.3f}"),
        ("Trap reward",      f"{s.get('trap_rew', 0):+.3f}"),
        ("Discount  γ",      f"{s.get('gamma', 0.95):.3f}"),
        ("Slip probability", f"{s.get('slip_prob', 0):.3f}"),
    ]))

    # ── Hyperparameters (algo-specific) ───────────────────────────────
    if algo in ("Q-Learning", "SARSA"):
        parts.append(_section("⚙️ Hyperparameters", [
            ("Learning rate  α", f"{s.get('alpha', 0):.4f}"),
            ("ε  initial",       f"{s.get('epsilon', 0):.3f}"),
            ("ε  decay",         f"{s.get('epsilon_decay', 0):.4f}"),
            ("ε  minimum",       f"{s.get('epsilon_min', 0.01):.3f}"),
        ]))
    elif algo == "DQN":
        hidden     = s.get("dqn_hidden") or []
        hidden_str = " → ".join(str(h) for h in hidden) if hidden else "—"
        parts.append(_section("⚙️ DQN Parameters", [
            ("Learning rate",  f"{s.get('dqn_lr', 0):.5f}"),
            ("Hidden layers",  hidden_str),
            ("Batch size",     str(s.get("dqn_batch", 64))),
            ("Replay buffer",  f"{s.get('dqn_buffer', 0):,}"),
            ("Target update",  f"every {s.get('dqn_target_update', 0)} steps"),
            ("ε  initial",     f"{s.get('dqn_epsilon', 0):.3f}"),
            ("ε  decay",       f"{s.get('dqn_epsilon_decay', 0):.5f}"),
            ("ε  minimum",     f"{s.get('dqn_epsilon_min', 0):.3f}"),
        ]))

        # ── Observation space ──────────────────────────────────────────
        n_nb     = s.get("obs_n_neighbors", 0)
        obs_type = f"Local patch  (n={n_nb})" if n_nb > 0 else "One-hot position"
        parts.append(_section("👁 Observation Space", [
            ("Type",             obs_type),
            ("Neighbor radius",  str(n_nb)),
            ("Goal distance",    "✓" if s.get("obs_use_goal_dist") else "✗"),
            ("Input dimension",  str(s.get("obs_input_dim", "—"))),
        ]))

        # ── Curriculum ────────────────────────────────────────────────
        if s.get("curriculum_active"):
            parts.append(_section("🎓 Curriculum", [
                ("Method",                  s.get("curriculum_method", "—")),
                ("Training setups",         str(s.get("curriculum_n_train", 0))),
                ("Test setups (held-out)",  str(s.get("curriculum_n_test",  0))),
                ("Advancement threshold",   f"{s.get('curriculum_threshold', 0):.2f}"),
            ]))

    # ── Results (populated after training) ────────────────────────────
    if res is not None:
        rews     = res.get("rewards") or []
        ep_times = res.get("episode_times") or []
        res_rows = []
        if rews:
            res_rows += [
                ("Episodes run",          f"{len(rews):,}"),
                ("Avg reward (last 100)", f"{float(np.mean(rews[-100:])):+.3f}"),
                ("Peak reward",           f"{max(rews):+.3f}"),
                ("Final reward",          f"{rews[-1]:+.3f}"),
            ]
        if ep_times:
            res_rows.append(("Total training time", format_duration(sum(ep_times))))
            res_rows.append(("Avg time / episode",
                             f"{float(np.mean(ep_times)) * 1000:.0f} ms"))
        if res_rows:
            parts.append(_section("📊 Results", res_rows))

    body     = "".join(parts)
    stopped  = s.get("stopped_early", False)
    n_done   = s.get("episodes_completed")
    if stopped:
        badge = (
            '<span style="background:#fef3c7;color:#92400e;border:1px solid #fcd34d;'
            'border-radius:4px;padding:2px 8px;font-size:0.7rem;font-weight:700;'
            'margin-left:8px;vertical-align:middle">⏹ STOPPED EARLY</span>'
        )
        ep_note = (
            f'<div style="font-size:0.72rem;color:#92400e;margin-top:4px">'
            f'Completed {n_done:,} episodes before stopping.</div>'
        ) if n_done is not None else ""
    else:
        badge    = ""
        ep_note  = ""

    card_html = (
        '<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;'
        'padding:14px 16px;font-family:Inter,system-ui,sans-serif">'
        '<div style="font-weight:700;font-size:0.88rem;color:#4F46E5;'
        'margin-bottom:4px;padding-bottom:8px;border-bottom:2px solid #e0e7ff">'
        f'🏋️ Training Setup{badge}'
        '</div>'
        + ep_note
        + '<div style="margin-top:10px">'
        + body
        + '</div></div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)


def _grid_thumbnail_html(layout, rows, cols, cell_rewards=None):
    """Compact inline-HTML mini-grid for curriculum thumbnails (no Plotly needed).

    Parameters
    ----------
    cell_rewards : dict | None
        Optional ``{(row, col): float}`` map.  Non-terminal cells with rewards
        are tinted green (positive) or red (negative).
    """
    _CHAR_BASE = {
        ".": "#F1F5F9", "#": "#1E293B", "~": "#BAE6FD",
        "G": "#BBF7D0", "X": "#FECACA", "S": "#FEF08A",
    }
    cell_px = max(5, min(14, 70 // max(rows, cols)))
    cr_map = cell_rewards or {}

    divs = []
    for ri, row_str in enumerate(layout):
        for ci, ch in enumerate(row_str):
            bg = _CHAR_BASE.get(ch, "#F1F5F9")
            # Tint empty / slippery cells that carry a non-zero reward
            if ch in (".", "~"):
                rv = cr_map.get((ri, ci))
                if rv is not None and float(rv) != 0:
                    v = float(rv)
                    t = min(1.0, abs(v) / 5.0)
                    a = round(0.18 + 0.55 * t, 2)
                    bg = (
                        f"rgba(74,222,128,{a})" if v >= 0
                        else f"rgba(248,113,113,{a})"
                    )
            # Emoji overlay for Start / Goal (only if cell is large enough)
            label = ""
            if cell_px >= 10:
                if ch == "S":
                    label = f'<span style="font-size:{cell_px - 2}px;line-height:1">🚦</span>'
                elif ch == "G":
                    label = f'<span style="font-size:{cell_px - 2}px;line-height:1">🏁</span>'
            divs.append(
                f'<div style="width:{cell_px}px;height:{cell_px}px;'
                f'background:{bg};outline:0.5px solid rgba(0,0,0,0.08);'
                f'display:flex;align-items:center;justify-content:center">'
                f"{label}</div>"
            )

    return (
        f'<div style="display:inline-grid;'
        f'grid-template-columns:repeat({cols},{cell_px}px);'
        f'gap:0;border:1.5px solid #CBD5E1;border-radius:4px;overflow:hidden">'
        + "".join(divs)
        + "</div>"
    )


# ── Session state defaults ────────────────────────────────────────────
for _k, _v in {
    "episodes": 1000,
    "alpha": 0.1,
    "dqn_hidden_str": "64,64",
    "dqn_lr": 1e-3,
    "dqn_batch_size": 64,
    "dqn_buffer_size": 10_000,
    "dqn_target_update": 100,
    "dqn_episodes": 1000,
    "dqn_epsilon": 1.0,
    "dqn_epsilon_decay": 0.997,
    "dqn_epsilon_min": 0.01,
    "record_limit": 100,
}.items():
    st.session_state.setdefault(_k, _v)


# ── Curriculum preview dialog ─────────────────────────────────────────
@st.dialog("🎓 Curriculum Setups", width="large")
def _curriculum_preview_dialog():
    """Modal popup: thumbnail grid sorted by complexity → click to drill into detail."""
    _all = sorted(
        st.session_state.get("curriculum_setups", []),
        key=lambda _s: _s["complexity"],
    )
    if not _all:
        st.warning("No curriculum setups available. Create them first.")
        return

    _sel = st.session_state.get("_cur_sel", None)

    if _sel is not None and 0 <= _sel < len(_all):
        # ── Detail view ──────────────────────────────────────────────
        _setup = _all[_sel]
        _g = _layout_to_grid(_setup["layout"])

        st.markdown(
            f"### Setup #{_setup['index'] + 1} &nbsp;·&nbsp; "
            f"Complexity **{_setup['complexity']:.1f} / 10**",
            unsafe_allow_html=True,
        )

        _m1, _m2, _m3 = st.columns(3)
        _m1.metric("Complexity", f"{_setup['complexity']:.1f}")
        _m2.metric("Grid size", f"{_setup['rows']} × {_setup['cols']}")
        _m3.metric("Position", f"{_sel + 1} / {len(_all)}")

        _setup_cr = {
            tuple(map(int, k.split(","))): float(v)
            for k, v in _setup.get("cell_rewards", {}).items()
        }
        st.plotly_chart(
            make_editor_fig(_g, _setup["rows"], _setup["cols"],
                            cell_rewards=_setup_cr),
            use_container_width=False,
            key=f"_cur_detail_fig_{_sel}",
        )

        _n1, _n2, _n3 = st.columns([1, 1, 2])
        if _n1.button(
            "← Prev", use_container_width=True, disabled=(_sel == 0),
            key="_cur_nav_prev",
        ):
            st.session_state["_cur_sel"] = _sel - 1
        if _n2.button(
            "Next →", use_container_width=True, disabled=(_sel == len(_all) - 1),
            key="_cur_nav_next",
        ):
            st.session_state["_cur_sel"] = _sel + 1
        if _n3.button(
            "↩  Back to all setups", use_container_width=True,
            key="_cur_nav_back",
        ):
            st.session_state["_cur_sel"] = None

    else:
        # ── Thumbnail grid ───────────────────────────────────────────
        _n_show = min(30, len(_all))
        _shown = [
            round(i * (len(_all) - 1) / max(1, _n_show - 1))
            for i in range(_n_show)
        ]
        st.markdown(
            f"**{len(_all)} setups** sorted by complexity · "
            f"showing {_n_show} samples · click **View** to inspect a setup"
        )
        st.divider()

        _NCOLS = 5
        for _rs in range(0, len(_shown), _NCOLS):
            _row = _shown[_rs: _rs + _NCOLS]
            _cols = st.columns(len(_row))
            for _ci, _idx in enumerate(_row):
                _s = _all[_idx]
                _s_cr = {
                    tuple(map(int, k.split(","))): float(v)
                    for k, v in _s.get("cell_rewards", {}).items()
                }
                _thumb = _grid_thumbnail_html(
                    _s["layout"], _s["rows"], _s["cols"], cell_rewards=_s_cr
                )
                _cols[_ci].markdown(
                    f'<div style="text-align:center;margin-bottom:2px">{_thumb}</div>',
                    unsafe_allow_html=True,
                )
                _cols[_ci].caption(f"c = {_s['complexity']:.1f}")
                if _cols[_ci].button(
                    "View",
                    key=f"_cur_th_{_idx}",
                    use_container_width=True,
                ):
                    st.session_state["_cur_sel"] = _idx


# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🐕 GridWorld RL")

    # ── Game Setup ────────────────────────────────────────────────────
    with st.expander("🗺️ Game Setup", expanded=True):
        st.markdown("### Grid Size")
        _c1, _c2 = st.columns(2)
        n_rows = _c1.number_input("Rows", 2, 14, st.session_state.rows, key="inp_rows")
        n_cols = _c2.number_input("Cols", 2, 14, st.session_state.cols, key="inp_cols")
        _b1, _b2 = st.columns(2)
        if _b1.button("Resize", use_container_width=True):
            _old = st.session_state.grid
            _g = _blank_grid(n_rows, n_cols)
            for _rr in range(min(n_rows, len(_old))):
                for _cc in range(min(n_cols, len(_old[0]))):
                    _g[_rr][_cc] = _old[_rr][_cc]
            st.session_state.grid = _g
            st.session_state.rows = n_rows
            st.session_state.cols = n_cols
            st.rerun()
        if _b2.button("Clear", use_container_width=True):
            st.session_state.grid = _blank_grid(st.session_state.rows, st.session_state.cols)
            st.rerun()

        st.divider()

        st.markdown("### Paint Tool & Rewards")
        st.radio(
            "Paint tool", options=list(range(6)),
            format_func=lambda i: f"  {CELL_LBL[i] or '·'}  {CELL_NAME[i]}",
            label_visibility="collapsed",
            key="tool",
        )
        st.markdown("#### Step Reward")
        step_rew = st.number_input("Step", value=st.session_state.get("step_rew", 0.0), step=0.01, format="%.2f", key="step_rew")
        st.checkbox(
            "🧲 Potential-based goal shaping",
            value=st.session_state.get("use_potential_shaping", True),
            key="use_potential_shaping",
            help=(
                "Adds a shaping bonus F(s,s') = γ·Φ(s') − Φ(s) where "
                "Φ(s) = −α·BFS_dist(s, goal).  Guaranteed to preserve the "
                "optimal policy (Ng et al., 1999).  α = goal_reward / (rows+cols)."
            ),
        )
        st.markdown("#### Paint-time Cell Reward")
        _pc1, _pc2 = st.columns([2, 1])
        _pc1.number_input("Cell reward", value=st.session_state.get("paint_reward_value", 0.0), step=0.01, format="%.2f", key="paint_reward_value")
        _pc2.checkbox("Apply", value=False, key="paint_apply_reward", help="Apply reward when painting")
        st.checkbox("Clear reward when painting", value=False, key="paint_clear_reward")

        st.divider()

        st.markdown("### Dynamics")
        slip_prob = st.slider("Slip probability", 0.0, 1.0, st.session_state.get("slip_prob", 0.3), 0.01,
                              help="Chance a slippery cell deflects the agent sideways", key="slip_prob")
        gamma = st.slider("Discount γ", 0.5, 1.0, st.session_state.get("gamma", 0.95), 0.01, key="gamma")

        st.divider()

        st.markdown("### Observation Space")
        st.caption("DQN only — controls what the agent sees beyond its own position.")
        st.slider(
            "Neighbor radius n", 0, 4,
            int(st.session_state.get("obs_n_neighbors", 0)),
            help=(
                "n=0: position only (one-hot, default).  "
                "n=1: position + 8 direct neighbours.  "
                "n=2: position + 24 neighbours, etc."
            ),
            key="obs_n_neighbors",
        )
        st.checkbox(
            "Include goal distance (dx, dy)",
            value=bool(st.session_state.get("obs_use_goal_dist", False)),
            key="obs_use_goal_dist",
            help="Appends normalised horizontal & vertical distance to the nearest Goal cell.",
        )

        st.divider()

        st.markdown("### Layout")
        _layout_file = st.file_uploader(
            "Upload Layout", type=["json"],
            help="Load a saved game layout with walls, goals, traps, rewards, and discount factor.",
        )
        if _layout_file is not None:
            try:
                _content = json.load(_layout_file)
                _rows = int(_content["rows"])
                _cols = int(_content["cols"])
                _lines = _content["layout"]
                if len(_lines) != _rows:
                    raise ValueError("Layout rows do not match saved rows.")
                if any(len(_row) != _cols for _row in _lines):
                    raise ValueError("Layout columns do not match saved cols.")
                st.session_state.grid = _layout_to_grid(_lines)
                st.session_state.rows = _rows
                st.session_state.cols = _cols
                st.session_state.step_rew = float(_content.get("step_reward", st.session_state.get("step_rew", 0.0)))
                st.session_state.slip_prob = float(_content.get("slip_prob", st.session_state.get("slip_prob", 0.3)))
                st.session_state.gamma = float(_content.get("gamma", st.session_state.get("gamma", 0.95)))
                _cr = {}
                for _k, _v in _content.get("cell_rewards", {}).items():
                    try:
                        _rs, _cs = _k.split(",")
                        _cr[(int(_rs), int(_cs))] = float(_v)
                    except Exception:
                        continue
                st.session_state.cell_rewards = _cr
                st.success("Layout loaded successfully.")
                _safe_rerun()
            except Exception as _exc:
                st.error(f"Unable to load layout: {_exc}")

        st.text_area(
            "Generation prompt",
            value=st.session_state.get("layout_prompt", ""),
            help="Describe layout preferences for the random generator.",
            key="layout_prompt",
        )
        st.checkbox(
            "Use Claude prompt to influence generation",
            value=st.session_state.get("use_claude_prompt", False),
            key="use_claude_prompt",
            help="If enabled, the prompt is sent to Claude to bias the generated layout.",
        )
        if st.session_state.get("use_claude_prompt") and st.session_state.get("layout_prompt", "").strip() and not _get_claude_api_key():
            st.warning("Claude API key is not configured. Set CLAUDE_API_KEY in the environment or Streamlit secrets.")
        st.slider(
            "Random layout complexity", 1, 10, st.session_state.get("layout_complexity", 1),
            help="Choose how complex the randomly generated grid should be.",
            key="layout_complexity",
        )
        if st.button("Generate random layout", use_container_width=True):
            _rows = st.session_state.rows
            _cols = st.session_state.cols
            _cell_rewards = {}
            _grid = None
            _use_claude = st.session_state.get("use_claude_prompt", False)
            _prompt = st.session_state.get("layout_prompt", "")
            _complexity = st.session_state.get("layout_complexity", 1)
            if _use_claude and _get_claude_api_key() and _prompt.strip():
                with st.spinner("Asking Claude to design the grid…"):
                    _grid, _err = _generate_grid_from_claude(_prompt, _rows, _cols, _complexity)
                if _grid is not None:
                    st.success("Claude designed the grid from your prompt.")
                else:
                    st.warning(f"Claude grid generation failed ({_err}). Falling back to random generator.")
                    _grid = None
            if _grid is None:
                _hints = None
                if _prompt.strip():
                    _hints = _infer_layout_directives_from_prompt(_prompt, _rows, _cols)
                    if _hints:
                        st.info("Using local prompt parsing to bias random layout generation.")
                _grid, _cell_rewards = _generate_random_layout(_complexity, _rows, _cols, hints=_hints)
                if not (_use_claude and _get_claude_api_key()):
                    st.success("Generated a new random layout.")
            st.session_state.grid = _grid
            st.session_state.cell_rewards = _cell_rewards
            st.session_state.results = None
            st.session_state.train_run = None
            st.session_state.loaded_training_metadata = None
            st.session_state.play_trace = None
            _safe_rerun()

        _ld = {
            "rows": st.session_state.rows,
            "cols": st.session_state.cols,
            "layout": _grid_to_layout(st.session_state.grid, st.session_state.rows, st.session_state.cols),
            "step_reward": float(st.session_state.get("step_rew", 0.0)),
            "goal_reward": 1.0,
            "trap_reward": -1.0,
            "slip_prob": float(st.session_state.get("slip_prob", 0.3)),
            "gamma": float(st.session_state.get("gamma", 0.95)),
            "cell_rewards": {f"{_r},{_c}": float(_v) for (_r, _c), _v in st.session_state.get("cell_rewards", {}).items()},
        }
        st.download_button(
            "Download Layout",
            json.dumps(_ld, indent=2),
            file_name=f"gridworld_{_ld['rows']}x{_ld['cols']}.json",
            mime="application/json",
        )

    # ── Run Setup ─────────────────────────────────────────────────────
    with st.expander("▶ Run Setup", expanded=False):
        st.markdown("### Algorithm")
        algo = st.selectbox(
            "Algorithm", ["Policy Iteration", "Q-Learning", "SARSA", "DQN"],
            label_visibility="collapsed",
            key="algo",
        )
        run_mode = st.selectbox(
            "Run mode", ["Full", "Step-through"], index=0, label_visibility="collapsed",
            key="run_mode",
        )

        if algo != "Policy Iteration":
            st.markdown("### Hyperparameters")
            episodes = st.number_input("Episodes", 200, 20_000, int(st.session_state.get("episodes", 1000)), 200, key="episodes")
            alpha = st.number_input("Learning rate α", 0.001, 1.0, float(st.session_state.get("alpha", 0.1)), 0.005,
                                    format="%.3f", key="alpha")
            epsilon = st.slider("Exploration ε", 0.0, 1.0, float(st.session_state.get("epsilon", 1.0)), 0.01,
                                help="Initial exploration probability for ε-greedy policies.", key="epsilon")
            epsilon_decay = st.number_input(
                "Epsilon decay (multiplicative)", 0.900, 1.000, float(st.session_state.get("epsilon_decay", 0.998)), 0.0005,
                format="%.4f", help="Per-episode multiplicative decay for ε", key="epsilon_decay",
            )
            exploring_starts = st.checkbox(
                "Exploring starts (randomize episode start)", value=bool(st.session_state.get("exploring_starts", True)),
                help="If enabled, each episode starts in a random non-terminal cell.", key="exploring_starts",
            )
            max_steps = st.number_input("Max steps per episode", 1, 1000, int(st.session_state.get("max_steps", 300)), 10,
                                        help="Maximum number of steps before the episode ends.", key="max_steps")

            if algo == "DQN":
                st.markdown("---")
                st.markdown("**DQN Parameters**")
                dqn_hidden_str = st.text_input(
                    "Hidden layers (comma-separated)", value=st.session_state.get("dqn_hidden_str", "64,64"),
                    help="Sizes for hidden linear layers, e.g. 64,64", key="dqn_hidden_str",
                )
                try:
                    dqn_hidden = [int(s.strip()) for s in dqn_hidden_str.split(",") if s.strip()] or [64, 64]
                except Exception:
                    dqn_hidden = [64, 64]
                dqn_lr = st.number_input("DQN learning rate", 1e-6, 1.0, float(st.session_state.get("dqn_lr", 1e-3)), format="%.5f", key="dqn_lr")
                dqn_batch_size = st.number_input("DQN batch size", 8, 4096, int(st.session_state.get("dqn_batch_size", 64)), step=1, key="dqn_batch_size")
                dqn_buffer_size = st.number_input("DQN buffer size", 100, 1_000_000, int(st.session_state.get("dqn_buffer_size", 10_000)), step=100, key="dqn_buffer_size")
                dqn_target_update = st.number_input("DQN target update (steps)", 1, 100_000, int(st.session_state.get("dqn_target_update", 100)), step=1, key="dqn_target_update")
                dqn_episodes = st.number_input("DQN episodes", 1, 50_000, int(st.session_state.get("dqn_episodes", 1000)), step=100, key="dqn_episodes")
                dqn_epsilon = st.number_input("DQN epsilon (initial)", 0.0, 1.0, float(st.session_state.get("dqn_epsilon", 1.0)), step=0.01, format="%.3f", key="dqn_epsilon")
                dqn_epsilon_decay = st.number_input("DQN epsilon decay", 0.900, 1.000, float(st.session_state.get("dqn_epsilon_decay", 0.997)), step=0.0005, format="%.4f", key="dqn_epsilon_decay")
                dqn_epsilon_min = st.number_input("DQN epsilon min", 0.0, 1.0, float(st.session_state.get("dqn_epsilon_min", 0.01)), step=0.01, format="%.3f", key="dqn_epsilon_min")

                st.markdown("---")
                st.markdown("**Curriculum Learning**")
                st.checkbox(
                    "Enable curriculum", key="use_curriculum",
                    value=bool(st.session_state.get("use_curriculum", False)),
                    help="Train DQN across environments of increasing difficulty.",
                )
                if st.session_state.get("use_curriculum"):
                    st.selectbox(
                        "Method", ["ADR", "Performance Threshold"],
                        index=0, key="curriculum_method",
                        help=(
                            "ADR: automatically adjust difficulty based on recent reward.  "
                            "Performance Threshold: advance stage when avg reward > threshold."
                        ),
                    )
                    st.number_input(
                        "Number of setups", 10, 10_000,
                        int(st.session_state.get("n_curriculum_setups", 100)),
                        step=10, key="n_curriculum_setups",
                        help="Total number of grid layouts to generate across all difficulty levels.",
                    )
                    st.number_input(
                        "Advancement threshold", -100.0, 100.0,
                        float(st.session_state.get("curriculum_perf_threshold", 0.5)),
                        step=0.1, format="%.2f", key="curriculum_perf_threshold",
                        help="Avg episode reward (over last 20 eps) required to advance difficulty.",
                    )
                    _cx_lo, _cx_hi = st.columns(2)
                    _cx_lo.number_input(
                        "Min complexity", 1.0, 9.9,
                        float(st.session_state.get("curriculum_min_complexity", 1.0)),
                        step=0.5, format="%.1f", key="curriculum_min_complexity",
                        help="Easiest grid difficulty to include (1 = trivial, 10 = hardest).",
                    )
                    _cx_hi.number_input(
                        "Max complexity", 1.1, 10.0,
                        float(st.session_state.get("curriculum_max_complexity", 10.0)),
                        step=0.5, format="%.1f", key="curriculum_max_complexity",
                        help="Hardest grid difficulty to include.",
                    )

                    _cc1, _cc2 = st.columns(2)
                    if _cc1.button("Create Setups", use_container_width=True):
                        _n_cur = int(st.session_state.get("n_curriculum_setups", 100))
                        # Derive layout constraints from the Game Setup prompt so
                        # that e.g. "no traps / one goal" is respected in every
                        # generated curriculum setup.
                        _cur_prompt = st.session_state.get("layout_prompt", "")
                        _cur_hints = (
                            _infer_layout_directives_from_prompt(
                                _cur_prompt,
                                st.session_state.rows,
                                st.session_state.cols,
                            )
                            if _cur_prompt.strip()
                            else {}
                        )
                        _cx_min_v = float(st.session_state.get("curriculum_min_complexity", 1.0))
                        _cx_max_v = float(st.session_state.get("curriculum_max_complexity", 10.0))
                        _cx_min_v, _cx_max_v = min(_cx_min_v, _cx_max_v - 0.1), max(_cx_max_v, _cx_min_v + 0.1)
                        with st.spinner(f"Generating {_n_cur} setups (complexity {_cx_min_v:.1f}→{_cx_max_v:.1f})…"):
                            _new_setups = _generate_curriculum_setups(
                                _n_cur,
                                st.session_state.rows,
                                st.session_state.cols,
                                step_reward=float(st.session_state.get("step_rew", 0.0)),
                                slip_prob=float(st.session_state.get("slip_prob", 0.3)),
                                gamma=float(st.session_state.get("gamma", 0.95)),
                                hints=_cur_hints,
                                min_complexity=_cx_min_v,
                                max_complexity=_cx_max_v,
                            )
                            _n_train = max(1, int(len(_new_setups) * 0.8))
                            st.session_state["curriculum_setups"] = _new_setups[:_n_train]
                            st.session_state["curriculum_test_setups"] = _new_setups[_n_train:]
                        _n_tr  = len(st.session_state.get("curriculum_setups", []))
                        _n_te  = len(st.session_state.get("curriculum_test_setups", []))
                        st.success(f"✓ {_n_tr} training setups + {_n_te} held-out test setups (complexity {_cx_min_v:.1f} → {_cx_max_v:.1f}).")

                    _cur_existing = st.session_state.get("curriculum_setups")
                    _has_setups = bool(_cur_existing)
                    if _cc2.button("View Sample", use_container_width=True, disabled=not _has_setups):
                        st.session_state["_cur_sel"] = None  # always open at thumbnail grid
                        _curriculum_preview_dialog()

                    if _has_setups:
                        _n_te_ex = len(st.session_state.get("curriculum_test_setups", []))
                        st.caption(f"✓ {len(_cur_existing)} training + {_n_te_ex} test setups ready.")
                        _cur_json = json.dumps({
                            "schema": "gridworld_curriculum_setups",
                            "version": 1,
                            "n_setups": len(_cur_existing),
                            "setups": _cur_existing,
                        }, indent=2)
                        st.download_button(
                            "💾 Download Setups", _cur_json,
                            file_name="curriculum_setups.json",
                            mime="application/json",
                        )

                    _cur_upload = st.file_uploader(
                        "Load Setups from file", type=["json"],
                        key="curriculum_upload",
                    )
                    if _cur_upload is not None:
                        try:
                            _loaded = json.load(_cur_upload)
                            if _loaded.get("schema") == "gridworld_curriculum_setups":
                                st.session_state["curriculum_setups"] = _loaded["setups"]
                                st.success(f"Loaded {len(_loaded['setups'])} curriculum setups.")
                            else:
                                st.error("Not a valid curriculum setups file.")
                        except Exception as _e:
                            st.error(f"Could not load: {_e}")
            else:
                dqn_hidden = [64, 64]
                dqn_lr = 1e-3
                dqn_batch_size = 64
                dqn_buffer_size = 10_000
                dqn_target_update = 100
                dqn_episodes = episodes
                dqn_epsilon = 1.0
                dqn_epsilon_decay = epsilon_decay
                dqn_epsilon_min = 0.01
        else:
            episodes = 0
            alpha = 0.1
            epsilon = st.session_state.get("epsilon", 1.0)
            epsilon_decay = st.session_state.get("epsilon_decay", 0.998)
            exploring_starts = st.session_state.get("exploring_starts", True)
            max_steps = st.session_state.get("max_steps", 300)
            dqn_hidden = [64, 64]
            dqn_lr = 1e-3
            dqn_batch_size = 64
            dqn_buffer_size = 10_000
            dqn_target_update = 100
            dqn_episodes = 3_000
            dqn_epsilon = 1.0
            dqn_epsilon_decay = 0.997
            dqn_epsilon_min = 0.01

        st.divider()
        st.markdown("### Episode Recording")
        st.checkbox(
            "Record episodes for replay",
            value=bool(st.session_state.get("record_episodes", False)),
            key="record_episodes",
            help="Store episode traces so you can replay episodes after training.",
        )
        st.selectbox(
            "Replay retention", ["Most recent", "Subsample evenly"],
            index=0, key="record_strategy",
            help="Most recent keeps the latest episodes; subsample evenly keeps a periodic sample.",
        )
        st.number_input(
            "Max recorded episodes", min_value=1, max_value=1000,
            value=int(st.session_state.get("record_limit", 100)), step=1,
            help="Keep this many recorded episodes for replay.", key="record_limit",
        )

        st.divider()
        st.markdown("### Load Training Result")
        _training_file = st.file_uploader(
            "Load Training Result", type=["json"],
            help="Load a previously saved training result.",
            key="load_training_file",
        )
        if _training_file is not None:
            try:
                _payload = json.load(_training_file)
                _load_training_payload(_payload)
                st.success("Training result loaded successfully.")
            except Exception as _exc:
                st.error(f"Unable to load training result: {_exc}")

    # ── Analysis ──────────────────────────────────────────────────────
    with st.expander("📊 Analysis", expanded=False):
        st.markdown("### Play From Start")
        play_max_steps = st.number_input("Play max steps", 1, 1000, int(st.session_state.get("play_max_steps", 100)), key="play_max_steps")
        play_delay = st.number_input("Delay per step (s)", 0.01, 2.0, float(st.session_state.get("play_delay", 0.25)), step=0.05, format="%.2f", key="play_delay")
        play_policy = st.selectbox("Policy for play", ["Policy (if available)", "Random"], index=0, key="play_policy")
        if st.button("Play from Start"):
            try:
                grid = st.session_state.grid
                rows = st.session_state.rows
                cols = st.session_state.cols
                layout = _grid_to_layout(grid, rows, cols)
                grid_cfg = GridConfig(
                    rows=rows, cols=cols, layout=layout,
                    step_reward=float(st.session_state.get("step_rew", 0.0)),
                    goal_reward=1.0, trap_reward=-1.0,
                    cell_rewards=st.session_state.get("cell_rewards", {}),
                    slippery_slip_prob=float(st.session_state.get("slip_prob", 0.3)),
                    gamma=float(st.session_state.get("gamma", 0.95)),
                )
                env_play = GridWorld(grid_cfg)
                s = env_play.reset()
                res = st.session_state.get("results") or {}
                pol = res.get("policy") if isinstance(res, dict) else None
                trace = {"states": [s], "actions": [], "rewards": [], "next_states": []}
                for _ in range(play_max_steps):
                    valid = valid_actions(env_play, s)
                    if not valid:
                        break
                    if play_policy == "Policy (if available)" and pol is not None and len(pol) == env_play.n_states:
                        a = int(pol[s]) if int(pol[s]) in valid else int(random.choice(valid))
                    else:
                        a = int(random.choice(valid))
                    ns, r, done = env_play.step(a)
                    trace["actions"].append(a)
                    trace["rewards"].append(r)
                    trace["next_states"].append(ns)
                    trace["states"].append(ns)
                    s = ns
                    if done:
                        break
                st.session_state["play_trace"] = trace
                st.success(f"Play finished — steps: {len(trace['actions'])}, reward: {sum(trace['rewards']):+.2f}")
            except Exception as exc:
                st.error(f"Play failed: {exc}")

        st.divider()
        st.markdown("### Claude Settings")
        _default_model = _get_claude_model_env() or "claude-opus-4-8"
        _model_options = ["claude-opus-4-8", "claude-3", "claude-3.5", "claude-2", "claude-2.1", "claude-instant", "custom"]
        _midx = _model_options.index(_default_model) if _default_model in _model_options else len(_model_options) - 1
        _choice = st.selectbox("Claude model", _model_options, index=_midx, key="claude_model_choice")
        if _choice == "custom":
            _cur = st.session_state.get("CLAUDE_MODEL", _default_model)
            _custom_val = _cur if _cur not in _model_options else ""
            _custom = st.text_input("Custom Claude model", value=_custom_val, key="claude_model_custom")
            st.session_state["CLAUDE_MODEL"] = _custom or _default_model
        else:
            st.session_state["CLAUDE_MODEL"] = _choice

    st.divider()

    # ── Reset-to-defaults ─────────────────────────────────────────────
    _rb1, _rb2 = st.columns([3, 2])
    _rb1.caption("Parameters are saved within the session.")
    if _rb2.button("↩ Reset Defaults", help="Restore all training & dynamics parameters to their initial defaults"):
        for _dk, _dv in _TRAINING_DEFAULTS.items():
            st.session_state[_dk] = _dv
        st.session_state.pop("training_summary", None)
        st.rerun()

    st.markdown('<div class="train-btn">', unsafe_allow_html=True)
    train_clicked = (
        st.button("▶  Train Agent", use_container_width=True)
        or st.session_state.pop("auto_retrain", False)
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Stop Training button (visible while a run is in progress) ──────
    _sidebar_tr = st.session_state.get("train_run")
    if _sidebar_tr and not _sidebar_tr.get("finished", True):
        if st.button("⏹ Stop Training", key="stop_training_btn",
                     use_container_width=True,
                     help="Stop the current run and keep results gathered so far"):
            st.session_state["stop_training_requested"] = True
            st.rerun()

    if run_mode == "Step-through":
        st.markdown("---")
        if st.button("Start Step-through", use_container_width=True):
            st.session_state.step_run = {
                "algo": algo,
                "cfg": None,
                "env": None,
                "state": None,
                "finished": False,
            }
            _safe_rerun()

goal_rew = 1.0
trap_rew = -1.0



# ── Main layout ───────────────────────────────────────────────────────
st.markdown("# 🐕  GridWorld RL Trainer")
st.markdown(
    "Design your environment, choose an algorithm, and watch the agent learn.",
    help="Click cells in the grid to paint them, then press **Train Agent**.",
)
st.divider()

left_col  = st.container()
right_col = st.container()

# ── Grid editor ───────────────────────────────────────────────────────
with left_col:
    st.markdown('<p class="section-title">🗺️ Grid Editor</p>', unsafe_allow_html=True)

    tool_name = CELL_NAME[st.session_state.tool]
    tool_lbl  = CELL_LBL[st.session_state.tool] or "·"
    st.caption(f"Active tool: **{tool_lbl} {tool_name}** — click a cell to paint")

    play_trace = st.session_state.get("play_trace")
    editor_placeholder = left_col.empty()
    editor_fig = make_editor_fig(
        st.session_state.grid,
        st.session_state.rows,
        st.session_state.cols,
        trace=play_trace,
    )
    event = editor_placeholder.plotly_chart(
        editor_fig,
        on_select="rerun",
        selection_mode=["points"],
        use_container_width=False,
        key="editor",
    )

    # Handle click-to-paint
    if event and event.selection and event.selection.points:
        pt  = event.selection.points[0]
        idx = pt["point_index"]
        r, c = divmod(idx, st.session_state.cols)
        t = st.session_state.tool

        # Apply or clear per-cell reward based on Paint Tool controls
        if st.session_state.get("paint_clear_reward", False):
            crmap = st.session_state.setdefault("cell_rewards", {})
            if (r, c) in crmap:
                del crmap[(r, c)]
        elif st.session_state.get("paint_apply_reward", False):
            crmap = st.session_state.setdefault("cell_rewards", {})
            crmap[(r, c)] = float(st.session_state.get("paint_reward_value", 0.0))

        if t == ST:
            for rr in range(st.session_state.rows):
                for cc in range(st.session_state.cols):
                    if st.session_state.grid[rr][cc] == ST:
                        st.session_state.grid[rr][cc] = E
        st.session_state.grid[r][c] = t
        st.rerun()

    # Legend chips
    chip_html = "".join(
        f'<span style="display:inline-block;margin:2px 4px;padding:2px 10px;'
        f'border-radius:20px;background:{CELL_FILL[i]};border:1.5px solid {CELL_LINE[i]};'
        f'font-size:0.7rem;font-weight:600;color:{CELL_FG[i]}">'
        f'{CELL_LBL[i] or "·"} {CELL_NAME[i]}</span>'
        for i in range(6)
    )
    st.markdown(chip_html, unsafe_allow_html=True)

    legend_html = (
        '<div style="margin-top:6px;font-size:0.85rem;color:#475569">'
        '<b>Custom rewards:</b> non-terminal cells can have overrides. '
        '<span style="display:inline-block;margin-left:8px;padding:2px 6px;border-radius:8px;background:#ECFDF5;border:1px solid #86EFAC;color:#065F46;font-weight:600">+0.50</span>'
        '<span style="display:inline-block;margin-left:6px;padding:2px 6px;border-radius:8px;background:#FEF3F2;border:1px solid #FCA5A5;color:#9B1C1C;font-weight:600">-0.50</span>'
        '</div>'
    )
    st.markdown(legend_html, unsafe_allow_html=True)

    if play_trace:
        play_steps = len(play_trace.get("actions", []))
        play_reward = sum(play_trace.get("rewards", []))
        left_col.markdown("---")
        c1, c2 = left_col.columns(2)
        c1.metric("Play steps", f"{play_steps}")
        c2.metric("Reward", f"{play_reward:+.2f}")
        if left_col.button("Clear play overlay"):
            del st.session_state["play_trace"]
            _safe_rerun()

    


# ── Results panel ─────────────────────────────────────────────────────
with right_col:
    # Placeholder filled by the training block when Train is clicked
    _train_progress_ph = st.empty()

    st.markdown('<p class="section-title">📊 Results</p>', unsafe_allow_html=True)

    res = st.session_state.results
    tr = st.session_state.get("train_run")
    # True while a training batch is running (batches rerun every ~5 s)
    _training_active = (
        isinstance(tr, dict)
        and not tr.get("finished", True)
    )


    if res is not None and not _training_active:
        loaded_meta = st.session_state.get("loaded_training_metadata")
        if loaded_meta:
            st.markdown("---")
            st.markdown(f"**Loaded result:** {loaded_meta.get('name', 'Unnamed')}")
            if loaded_meta.get('description'):
                st.caption(loaded_meta.get('description'))
            st.write(f"Saved at: {loaded_meta.get('saved_at', 'unknown')}")
            st.markdown(f"**Algorithm:** {loaded_meta.get('algorithm', 'unknown')}")
            cfg = loaded_meta.get('config', {})
            if cfg:
                filtered = _filter_config_for_algo(cfg, loaded_meta.get('algorithm', ''))
                st.markdown("**Training parameters**")
                # Render as a compact two-column table for readability
                rows = []
                for k, v in filtered.items():
                    val = _format_param_value(v)
                    rows.append({"Parameter": k, "Value": val})
                if rows:
                    st.table(rows)
    if res is None or _training_active:
        st.markdown("""
        <div style="background:#F8FAFC;border:2px dashed #CBD5E1;border-radius:16px;
                    padding:3rem;text-align:center;margin-top:1rem">
            <div style="font-size:3rem">🐕</div>
            <div style="font-size:1.1rem;font-weight:600;color:#475569;margin-top:0.5rem">
                No results yet</div>
            <div style="font-size:0.85rem;color:#94A3B8;margin-top:0.3rem">
                Design your grid and press <b>Train Agent</b></div>
        </div>
        """, unsafe_allow_html=True)
    else:
        env    = res["env"]
        policy = res["policy"]
        values = res["values"]
        rews   = res["rewards"]
        lens   = res["lengths"]
        algo_  = res["algo"]

        _tsummary = st.session_state.get("training_summary")
        if _tsummary:
            _sum_col, _graph_col = st.columns([4, 7])
            with _sum_col:
                _render_training_summary_card(_tsummary, res)
        else:
            _graph_col = st.container()

        with _graph_col:

            # ── Stopped-early banner ─────────────────────────────────────
            _stopped_ts = st.session_state.get("training_summary", {})
            if _stopped_ts.get("stopped_early"):
                _n_ep_done = _stopped_ts.get("episodes_completed", 0)
                _n_ep_total = _stopped_ts.get("total_ep", _n_ep_done)
                st.warning(
                    f"⏹ **Training was stopped early** — {_n_ep_done:,} of "
                    f"{_n_ep_total:,} planned episodes completed. "
                    f"Results below reflect partial training.",
                    icon=None,
                )

            # Live training progress (if running)
            if tr is not None:
                st.markdown("---")
                state = tr.get("state", {})
                eps = state.get("episode", state.get("episodes", 0))
                last_reward = None
                if state.get("rewards"):
                    last_reward = state["rewards"][-1]
                run_episodes = state.get("episode", state.get("episodes", 0))
                elapsed = time.time() - tr.get("started_at", time.time())
                c1, c2, c3 = st.columns(3)
                c1.metric("Current episode", f"{eps}")
                c2.metric("Episodes performed", f"{run_episodes}")
                c3.metric("Training time", format_duration(elapsed))
                st.markdown(f"**Last reward:** {last_reward if last_reward is not None else '—'}")

            # Metrics row
            if rews is not None:
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Avg reward (last 100)", f"{np.mean(rews[-100:]):.3f}")
                m2.metric("Peak reward",           f"{max(rews):.3f}")
                m3.metric("Episodes",              f"{len(rews):,}")
                off_steps = res.get("off_policy_steps")
                if off_steps is not None and len(off_steps) > 0:
                    m4.metric("Avg off-policy steps", f"{np.mean(off_steps[-100:]):.2f}")
                else:
                    m4.metric("Avg off-policy steps", "—")
                total_time = sum(res.get("episode_times", []))
                m5.metric("Training time", format_duration(total_time))

            # Policy figure
            st.plotly_chart(
                make_policy_fig(env, policy, values),
                use_container_width=False,
                key="policy_fig",
            )

            # Visit counts grid (if available)
            if res.get("visits") is not None:
                st.markdown("<div class=\"section-title\">🔢 Visit Counts</div>", unsafe_allow_html=True)
                st.plotly_chart(
                    make_visit_fig(env, res.get("visits")),
                    use_container_width=False,
                    key="visits_fig",
                )

            # Training curves
            if rews is not None:
                st.plotly_chart(
                    make_curves_fig(rews, lens, algo_, res.get("off_policy_steps")),
                    use_container_width=True,
                    key="curves_fig",
                )

                # Training time analysis
                episode_times = res.get("episode_times")
                if episode_times is not None and len(episode_times) > 0:
                    st.plotly_chart(
                        make_timing_fig(episode_times, algo_),
                        use_container_width=True,
                        key="timing_fig",
                    )

                # Epsilon decay analysis
                epsilon_values = res.get("epsilon_values")
                if epsilon_values is not None and len(epsilon_values) > 0:
                    st.plotly_chart(
                        make_epsilon_fig(epsilon_values, algo_),
                        use_container_width=True,
                        key="epsilon_fig",
                    )

                # Curriculum complexity curve
                _ep_cx = res.get("episode_complexities")
                if _ep_cx:
                    _cx_fig = make_complexity_fig(_ep_cx, algo_)
                    if _cx_fig is not None:
                        st.plotly_chart(
                            _cx_fig,
                            use_container_width=True,
                            key="complexity_fig",
                        )

                # Episode replay
                episode_traces = res.get("episode_traces")
                if episode_traces:
                    st.markdown("<div class=\"section-title\">🎬 Episode Replay</div>", unsafe_allow_html=True)
                    # Build selector labels — include complexity when present
                    _replay_gamma = getattr(env, "gamma", 0.99)
                    def _ep_label(i, _g=_replay_gamma):
                        t = episode_traces[i - 1]
                        cx = t.get("complexity", 0.0)
                        cx_str = f" · complexity {cx:.1f}" if cx > 0 else ""
                        _rews = t.get("rewards", [])
                        ep_r = sum((_g ** step) * r for step, r in enumerate(_rews))
                        return f"Episode {i}{cx_str}  (return {ep_r:+.3f})"

                    episode_index = st.selectbox(
                        "Episode to replay",
                        options=list(range(1, len(episode_traces) + 1)),
                        index=0,
                        format_func=_ep_label,
                        key="replay_episode",
                    )
                    trace = episode_traces[episode_index - 1]
                    ep_steps = len(trace.get("actions", []))
                    _trace_rews = trace.get("rewards", [])
                    # Discounted return — matches what the training curves record
                    _gamma = getattr(env, "gamma", 0.99)
                    ep_reward = sum(
                        (_gamma ** t) * r for t, r in enumerate(_trace_rews)
                    )
                    # Undiscounted for reference
                    ep_reward_raw = sum(_trace_rews)
                    ep_off_policy = trace.get("off_policy_steps", 0)
                    ep_complexity = trace.get("complexity", 0.0)

                    _has_cx = ep_complexity > 0
                    _metric_cols = st.columns([1, 1, 1, 1, 1.2] if _has_cx else [1, 1, 1, 1.4])
                    _metric_cols[0].metric("Replay episode", f"{episode_index}")
                    _metric_cols[1].metric("Steps", f"{ep_steps}")
                    _metric_cols[2].metric("Off-policy steps", f"{ep_off_policy}")
                    _metric_cols[3].metric(
                        "Return (discounted)",
                        f"{ep_reward:+.3f}",
                        f"raw Σr = {ep_reward_raw:+.2f}",
                        help=f"Discounted return Σγᵗrₜ with γ={_gamma}. "
                             "Delta shows the undiscounted sum for comparison.",
                    )
                    if _has_cx:
                        # Complexity badge — colour-coded low→high
                        _cx_color = (
                            "#10B981" if ep_complexity <= 3
                            else "#F59E0B" if ep_complexity <= 7
                            else "#EF4444"
                        )
                        _metric_cols[4].markdown(
                            f"<div style='text-align:center;margin-top:6px'>"
                            f"<div style='font-size:0.75rem;color:#6B7280;margin-bottom:2px'>Complexity</div>"
                            f"<span style='font-size:1.5rem;font-weight:700;color:{_cx_color}'>"
                            f"{ep_complexity:.1f}</span>"
                            f"<span style='font-size:0.8rem;color:#9CA3AF'> / 10</span>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                    st.plotly_chart(
                        make_replay_fig(env, trace),
                        use_container_width=False,
                        key="replay_fig",
                    )

                st.markdown("---")
                st.markdown('<div class="section-title">💾 Save Training Result</div>', unsafe_allow_html=True)
                save_name = st.text_input(
                    "Result name", value=st.session_state.get("save_result_name", f"{algo_} training"),
                    key="save_result_name",
                )
                save_description = st.text_area(
                    "Description",
                    value=st.session_state.get("save_result_description", ""),
                    key="save_result_description",
                    help="Optional notes about this training run.",
                )
                save_payload = _make_training_save_payload(
                    res,
                    GridConfig(
                        rows=st.session_state.rows,
                        cols=st.session_state.cols,
                        layout=_grid_to_layout(st.session_state.grid, st.session_state.rows, st.session_state.cols),
                        step_reward=float(st.session_state.get("step_rew", 0.0)),
                        goal_reward=1.0,
                        trap_reward=-1.0,
                        cell_rewards=st.session_state.get("cell_rewards", {}),
                        slippery_slip_prob=float(st.session_state.get("slip_prob", 0.3)),
                        gamma=float(st.session_state.get("gamma", 0.95)),
                    ),
                    AlgorithmConfig(
                        alpha=st.session_state.get("alpha", 0.1),
                        n_episodes=st.session_state.get("episodes", 1000),
                        dqn_n_episodes=st.session_state.get("episodes", 1000),
                        max_steps=st.session_state.get("max_steps", 200),
                        epsilon=st.session_state.get("epsilon", 1.0),
                        epsilon_decay=st.session_state.get("epsilon_decay", 0.998),
                        epsilon_min=st.session_state.get("epsilon", 1.0) if st.session_state.get("epsilon", 1.0) else 0.01,
                        dqn_epsilon_decay=st.session_state.get("epsilon_decay", 0.998),
                        exploring_starts=st.session_state.get("exploring_starts", False),
                    ),
                    save_name,
                    save_description,
                )
                save_json = json.dumps(save_payload, indent=2)
                st.download_button(
                    "Download training result",
                    save_json,
                    file_name=f"{save_name or 'training_result'}.json",
                    mime="application/json",
                )

                # ── DQN post-training test ────────────────────────────────
                _test_setups = st.session_state.get("curriculum_test_setups", [])
                if algo_ == "DQN" and res.get("q_net") is not None and _test_setups:
                    st.markdown("---")
                    st.markdown('<div class="section-title">🧪 Test Agent</div>', unsafe_allow_html=True)
                    st.caption(
                        f"{len(_test_setups)} held-out test environments "
                        f"(complexity {min(s['complexity'] for s in _test_setups):.1f}–"
                        f"{max(s['complexity'] for s in _test_setups):.1f}) were never seen during training."
                    )

                    _ta, _tb = st.columns([2, 1])
                    _cx_target = _ta.slider(
                        "Target complexity",
                        min_value=1.0, max_value=10.0,
                        value=float(st.session_state.get("test_complexity", 5.0)),
                        step=0.5,
                        key="test_complexity",
                        help="The agent is tested on held-out boards whose complexity is closest to this value (±1.5).",
                    )
                    _n_test_games = _tb.number_input(
                        "Games to play", min_value=10, max_value=500,
                        value=int(st.session_state.get("n_test_games", 50)),
                        step=10, key="n_test_games",
                    )

                    # Show which test setups are in range
                    _in_range = [s for s in _test_setups if abs(s["complexity"] - _cx_target) <= 1.5]
                    if _in_range:
                        st.caption(f"↳ {len(_in_range)} test boards in range "
                                   f"(complexity {min(s['complexity'] for s in _in_range):.1f}–"
                                   f"{max(s['complexity'] for s in _in_range):.1f})")
                    else:
                        st.caption("↳ No test boards within ±1.5 — will use the closest available boards.")

                    if st.button("▶ Run Test", key="run_dqn_test_btn", use_container_width=True):
                        with st.spinner(f"Testing on {_n_test_games} games at complexity ~{_cx_target:.1f}…"):
                            _tq = res["q_net"]
                            _tn = int(res.get("obs_n_neighbors", 0))
                            _tg = bool(res.get("obs_use_goal_dist", False))
                            _tres = _run_dqn_test(
                                _tq, _test_setups, _cx_target,
                                int(_n_test_games), _tn, _tg,
                            )
                            st.session_state["dqn_test_result"] = _tres

                    _tshow = st.session_state.get("dqn_test_result")
                    if _tshow:
                        _r1, _r2, _r3, _r4 = st.columns(4)
                        _r1.metric("Win rate",    f"{_tshow['win_rate']:.1%}")
                        _r2.metric("Avg reward",  f"{_tshow['avg_reward']:+.3f}")
                        _r3.metric("Avg steps",   f"{_tshow['avg_steps']:.1f}")
                        _r4.metric("Games played", f"{_tshow['n_games']}")

                        # Reward distribution bar chart
                        _pg = _tshow["per_game"]
                        _rw_vals = [g["reward"] for g in _pg]
                        import math
                        _n_bins = max(5, min(20, int(math.sqrt(len(_rw_vals)))))
                        _mn, _mx = min(_rw_vals), max(_rw_vals)
                        _bw = (_mx - _mn) / _n_bins if _mx > _mn else 1.0
                        _bins = [_mn + i * _bw for i in range(_n_bins + 1)]
                        _counts = [0] * _n_bins
                        for rv in _rw_vals:
                            bi = min(_n_bins - 1, int((rv - _mn) / _bw))
                            _counts[bi] += 1
                        _bin_labels = [f"{_bins[i]:.2f}" for i in range(_n_bins)]
                        _bar_colors = [
                            "#4ADE80" if ((_bins[i] + _bins[i+1]) / 2) >= 0 else "#F87171"
                            for i in range(_n_bins)
                        ]
                        _test_fig = go.Figure(go.Bar(
                            x=_bin_labels, y=_counts,
                            marker_color=_bar_colors,
                            hovertemplate="Reward: %{x}<br>Count: %{y}<extra></extra>",
                        ))
                        _test_fig.update_layout(
                            title=dict(text="Reward distribution across test games", font_size=13),
                            xaxis_title="Episode reward",
                            yaxis_title="# games",
                            height=260,
                            margin=dict(l=30, r=10, t=40, b=40),
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            xaxis=dict(showgrid=False),
                            yaxis=dict(gridcolor="#e2e8f0"),
                        )
                        st.plotly_chart(_test_fig, use_container_width=True, key="dqn_test_fig")
                        st.caption(
                            f"Tested against {_tshow['n_setups_used']} held-out board(s) "
                            f"(greedy policy, ε=0)."
                        )
            else:
                pi_iters = res.get("pi_iters", "?")
                st.success(f"Policy Iteration converged in **{pi_iters}** sweeps.", icon="✅")
                st.info(
                    "Policy Iteration state values are expected returns from each state. "
                    "If the next action enters the goal, the adjacent state's value can be 1.0 "
                    "because the goal reward is received immediately upon reaching it.",
                    icon="ℹ️",
                )

        # ── Analyze Last Run ────────────────────────────────────────────────
        _saved_path = st.session_state.get("last_saved_run_path")
        _an_summary = st.session_state.get("training_summary")
        if _an_summary:
            st.markdown("---")
            st.markdown('<div class="section-title">🔍 Analyze Last Run</div>',
                        unsafe_allow_html=True)

            _an_col1, _an_col2 = st.columns([1, 2])
            with _an_col1:
                if _saved_path:
                    st.caption(f"💾 Auto-saved: `{_saved_path}`")
                if st.button("🔍 Analyze with AI", key="analyze_run_btn",
                             use_container_width=True,
                             help="Ask Claude to diagnose issues and suggest parameter changes"):
                    with st.spinner("Analysing training run…"):
                        try:
                            _analysis_result = _call_claude_analysis(_an_summary, res)
                            st.session_state["last_run_analysis"] = _analysis_result
                            # Reset per-suggestion checkbox defaults
                            for _si in range(len(_analysis_result.get("suggestions", []))):
                                st.session_state.setdefault(f"sug_check_{_si}", True)
                        except Exception as _ae:
                            st.error(f"Analysis failed: {_ae}")

            _analysis = st.session_state.get("last_run_analysis")
            if _analysis:
                # Issues
                _issues = _analysis.get("issues", [])
                if _issues:
                    with _an_col2:
                        st.markdown("**⚠️ Issues detected**")
                        for _iss in _issues:
                            st.warning(_iss)

                # Suggestions with checkboxes
                _sugs = _analysis.get("suggestions", [])
                if _sugs:
                    st.markdown("**💡 Suggested improvements**")
                    for _si, _sug in enumerate(_sugs):
                        _sc1, _sc2 = st.columns([0.05, 0.95])
                        _is_checked = _sc1.checkbox(
                            "", key=f"sug_check_{_si}",
                            value=st.session_state.get(f"sug_check_{_si}", True),
                        )
                        with _sc2:
                            _cur_val = st.session_state.get(_sug.get("param"), "—")
                            st.markdown(
                                f"**{_sug.get('text', '—')}**  "
                                f"<span style='color:#64748b;font-size:0.8rem'>"
                                f"`{_sug.get('param')}`: "
                                f"{_cur_val} → **{_sug.get('value')}**</span>",
                                unsafe_allow_html=True,
                            )
                            st.caption(_sug.get("explanation", ""))

                    st.markdown("")
                    if st.button("🔄 Try Again with selected suggestions",
                                 key="try_again_btn", use_container_width=False):
                        # Collect into a buffer — Streamlit forbids writing widget-bound
                        # keys after they've been rendered; the buffer is applied at
                        # script start on the next rerun (before widgets are drawn).
                        _patch = {}
                        for _si, _sug in enumerate(_sugs):
                            if st.session_state.get(f"sug_check_{_si}", True):
                                _pk = _sug.get("param")
                                _pv = _sug.get("value")
                                if _pk and _pk in _SUGGESTION_PARAM_TYPES:
                                    _patch[_pk] = _pv
                        st.session_state["_pending_param_changes"] = _patch
                        # Clear stale results & trigger retrain
                        st.session_state.results   = None
                        st.session_state.pop("train_run", None)
                        st.session_state.pop("last_run_analysis", None)
                        st.session_state.pop("analysis_chat", None)
                        st.session_state["auto_retrain"] = True
                        st.rerun()

                # ── Conversation thread ──────────────────────────────────
                st.markdown("---")
                st.markdown("**💬 Ask about this run**")
                st.caption(
                    "Ask a follow-up question about a suggestion, "
                    "or anything about your training setup and results."
                )

                # Build a rich system prompt so Claude has full context
                _chat_system = (
                    "You are an expert reinforcement learning assistant helping a user "
                    "analyse and improve a training run in a GridWorld RL playground.\n\n"
                    "## Training setup\n"
                    + "\n".join(
                        f"- {k}: {v}"
                        for k, v in (_an_summary or {}).items()
                        if k not in ("hp_rows",)
                    )
                    + "\n\n## Analysis results\n"
                    + "Issues detected:\n"
                    + "\n".join(f"- {i}" for i in _analysis.get("issues", []))
                    + "\n\nSuggestions:\n"
                    + "\n".join(
                        f"- {s.get('text')} "
                        f"(param `{s.get('param')}`: {st.session_state.get(s.get('param'), '?')} → {s.get('value')}): "
                        f"{s.get('explanation', '')}"
                        for s in _analysis.get("suggestions", [])
                    )
                    + "\n\nAnswer clearly and concisely. "
                    "If the user asks about a specific suggestion, explain the reasoning in depth."
                )

                # Display existing conversation history
                _chat_history: list[dict] = st.session_state.get("analysis_chat", [])
                for _msg in _chat_history:
                    with st.chat_message(_msg["role"]):
                        st.markdown(_msg["content"])

                # Chat input
                _chat_input = st.chat_input(
                    "Ask a question about this run or any suggestion…",
                    key="analysis_chat_input",
                )
                if _chat_input:
                    _chat_history.append({"role": "user", "content": _chat_input})
                    with st.chat_message("user"):
                        st.markdown(_chat_input)
                    with st.chat_message("assistant"):
                        with st.spinner("Thinking…"):
                            try:
                                _reply = _call_claude_chat(
                                    messages=_chat_history,
                                    system=_chat_system,
                                    max_tokens=800,
                                )
                            except Exception as _ce:
                                _reply = f"⚠️ Could not reach Claude: {_ce}"
                        st.markdown(_reply)
                    _chat_history.append({"role": "assistant", "content": _reply})
                    st.session_state["analysis_chat"] = _chat_history

                # Clear conversation button
                if _chat_history:
                    if st.button("🗑 Clear conversation", key="clear_chat_btn"):
                        st.session_state.pop("analysis_chat", None)
                        st.rerun()

    # Step-through controls (advance one episode / sweep)
    if "step_run" in st.session_state:
        sr = st.session_state.step_run
        st.markdown("---")
        st.markdown(f"**Step-through mode** — algo: **{sr.get('algo', '?')}**")
        if sr.get("finished"):
            st.success("Step-through finished.")
            if st.button("Reset Step-through"):
                del st.session_state.step_run
                st.session_state.results = None
                _safe_rerun()
        else:
            c1, c2 = st.columns([1, 1])
            if c1.button("Next"):
                # Advance one episode (Q/SARSA/DQN) or one sweep (Policy Iteration)
                algo_name = sr.get("algo")
                env = sr.get("env")
                cfg = sr.get("cfg")

                # If user started Step-through without pressing Train, initialize env/cfg now
                if env is None or cfg is None:
                    grid = st.session_state.grid
                    rows = st.session_state.rows
                    cols = st.session_state.cols
                    layout = _grid_to_layout(grid, rows, cols)
                    grid_cfg = GridConfig(
                        rows=rows, cols=cols, layout=layout,
                        step_reward=step_rew, goal_reward=goal_rew, trap_reward=trap_rew,
                        slippery_slip_prob=slip_prob, gamma=gamma,
                    )
                    cfg = AlgorithmConfig(
                        alpha=alpha, n_episodes=episodes, dqn_n_episodes=episodes,
                        max_steps=max_steps, epsilon=epsilon,
                        epsilon_decay=epsilon_decay, epsilon_min=0.01,
                        dqn_epsilon_decay=epsilon_decay,
                    )
                    env = GridWorld(grid_cfg)
                    sr["env"] = env
                    sr["cfg"] = cfg

                # Q-Learning step
                if algo_name == "Q-Learning":
                    state = sr.get("state") or {}
                    sr["state"] = state
                    if not state:
                        state["Q"] = np.zeros((env.n_states, env.n_actions), dtype=np.float64)
                        state["epsilon"] = cfg.epsilon
                        state["episode"] = 0
                        state["rewards"] = []
                        state["lengths"] = []
                        state["visits"] = np.zeros(env.n_states, dtype=int)

                    Q = state["Q"]
                    eps = state["epsilon"]

                    # run one episode
                    s = env.reset()
                    total_reward = 0.0
                    steps = 0
                    for _ in range(cfg.max_steps):
                        state["visits"][s] += 1
                        if np.random.random() < eps:
                            a = np.random.randint(env.n_actions)
                        else:
                            a = int(np.argmax(Q[s]))
                        ns, r, done = env.step(a)
                        # count arrival
                        state["visits"][ns] += 1
                        total_reward += (env.gamma ** steps) * r
                        steps += 1
                        best_next = 0.0 if done else float(np.max(Q[ns]))
                        Q[s, a] += cfg.alpha * (r + env.gamma * best_next - Q[s, a])
                        s = ns
                        if done:
                            break

                    eps = max(cfg.epsilon_min, eps * cfg.epsilon_decay)
                    state["epsilon"] = eps
                    state["episode"] += 1
                    state["rewards"].append(total_reward)
                    state["lengths"].append(steps)

                    policy = np.argmax(Q, axis=1)
                    st.session_state.results = dict(
                        algo=algo_name, env=env, policy=policy, values=Q.max(axis=1),
                        rewards=state["rewards"], lengths=state["lengths"], visits=state.get("visits"),
                    )
                    sr["state"] = state

                # SARSA step
                elif algo_name == "SARSA":
                    state = sr.get("state") or {}
                    sr["state"] = state
                    if not state:
                        state["Q"] = np.zeros((env.n_states, env.n_actions), dtype=np.float64)
                        state["epsilon"] = cfg.epsilon
                        state["episode"] = 0
                        state["rewards"] = []
                        state["lengths"] = []
                        state["visits"] = np.zeros(env.n_states, dtype=int)

                    Q = state["Q"]
                    eps = state["epsilon"]

                    def eps_greedy(s_):
                        # Return a legal action; if state is terminal return a dummy action
                        if env.is_terminal(s_):
                            return 0
                        valid = valid_actions(env, s_)
                        if np.random.random() < eps:
                            return int(random.choice(valid))
                        return int(max(valid, key=lambda act: Q[s_, act])) if valid else int(np.argmax(Q[s_]))

                    s = env.reset()
                    a = eps_greedy(s)
                    total_reward = 0.0
                    steps = 0
                    for _ in range(cfg.max_steps):
                        state["visits"][s] += 1
                        ns, r, done = env.step(a)
                        # count arrival
                        state["visits"][ns] += 1
                        na = eps_greedy(ns)
                        total_reward += (env.gamma ** steps) * r
                        steps += 1
                        next_q = 0.0 if done else Q[ns, na]
                        Q[s, a] += cfg.alpha * (r + env.gamma * next_q - Q[s, a])
                        s, a = ns, na
                        if done:
                            break

                    eps = max(cfg.epsilon_min, eps * cfg.epsilon_decay)
                    state["epsilon"] = eps
                    state["episode"] += 1
                    state["rewards"].append(total_reward)
                    state["lengths"].append(steps)

                    policy = np.argmax(Q, axis=1)
                    st.session_state.results = dict(
                        algo=algo_name, env=env, policy=policy, values=Q.max(axis=1),
                        rewards=state["rewards"], lengths=state["lengths"], visits=state.get("visits"),
                    )
                    sr["state"] = state

                # Policy Iteration step (one sweep)
                elif algo_name == "Policy Iteration":
                    state = sr.get("state") or {}
                    sr["state"] = state
                    if not state:
                        n = env.n_states
                        state["V"] = np.zeros(n, dtype=np.float64)
                        state["policy"] = np.zeros(n, dtype=np.int64)
                        state["sweep"] = 0

                    V = state["V"]
                    policy = state["policy"]
                    # Policy evaluation (to threshold)
                    for _ in range(cfg.pi_max_iter):
                        delta = 0.0
                        for s in range(env.n_states):
                            if env.is_wall(s) or env.is_terminal(s):
                                continue
                            v = sum(
                                p * (r + env.gamma * V[ns])
                                for p, ns, r, _ in env.get_transitions(s, policy[s])
                            )
                            delta = max(delta, abs(v - V[s]))
                            V[s] = v
                        if delta < cfg.pi_threshold:
                            break

                    # Policy improvement
                    policy_stable = True
                    for s in range(env.n_states):
                        if env.is_wall(s) or env.is_terminal(s):
                            continue
                        old = policy[s]
                        q_vals = [
                            sum(p * (r + env.gamma * V[ns]) for p, ns, r, _ in env.get_transitions(s, a))
                            for a in env.ACTIONS
                        ]
                        policy[s] = int(np.argmax(q_vals))
                        if policy[s] != old:
                            policy_stable = False

                    state["sweep"] += 1
                    sr["state"] = state
                    st.session_state.results = dict(
                        algo=algo_name, env=env, policy=policy, values=V,
                        rewards=None, lengths=None, pi_iters=state["sweep"],
                    )

                    if policy_stable:
                        sr["finished"] = True

                # DQN per-episode step
                elif algo_name == "DQN":
                    state = sr.get("state") or {}
                    sr["state"] = state
                    cfg = sr.get("cfg")
                    if not state:
                        _n_nb = int(st.session_state.get("obs_n_neighbors", 0))
                        _use_gd = bool(st.session_state.get("obs_use_goal_dist", False))
                        _inp_dim = _obs_dim(env, _n_nb, _use_gd)
                        device = torch.device("cpu")
                        q_net = QNetwork(_inp_dim, env.n_actions, cfg.dqn_hidden).to(device)
                        target_net = QNetwork(_inp_dim, env.n_actions, cfg.dqn_hidden).to(device)
                        target_net.load_state_dict(q_net.state_dict())
                        target_net.eval()
                        optimizer = optim.Adam(q_net.parameters(), lr=cfg.dqn_lr)
                        buffer = ReplayBuffer(cfg.dqn_buffer_size)
                        eps = cfg.dqn_epsilon
                        state.update({
                            "q_net": q_net,
                            "target_net": target_net,
                            "optimizer": optimizer,
                            "buffer": buffer,
                            "epsilon": eps,
                            "total_steps": 0,
                            "episodes": 0,
                            "rewards": [],
                            "lengths": [],
                            "visits": np.zeros(env.n_states, dtype=int),
                            "obs_n_neighbors": _n_nb,
                            "obs_use_goal_dist": _use_gd,
                            "adr_difficulty": 0.0,
                            "curriculum_stage": 0,
                        })

                    q_net = state["q_net"]
                    target_net = state["target_net"]
                    optimizer = state["optimizer"]
                    buffer = state["buffer"]
                    eps = state["epsilon"]
                    total_steps = state["total_steps"]
                    _n_nb = state.get("obs_n_neighbors", 0)
                    _use_gd = state.get("obs_use_goal_dist", False)

                    # ── Curriculum: pick episode environment ──────────
                    _st_use_curric = st.session_state.get("use_curriculum", False)
                    _st_curric_setups = st.session_state.get("curriculum_setups") or []
                    _st_curric_method = st.session_state.get("curriculum_method", "ADR")
                    _st_curric_thr = float(st.session_state.get("curriculum_perf_threshold", 0.5))

                    if _st_use_curric and _st_curric_setups:
                        if _st_curric_method == "ADR":
                            _adr = state.get("adr_difficulty", 0.0)
                            _tgt = int(_adr * (len(_st_curric_setups) - 1))
                            _spd = max(1, len(_st_curric_setups) // 10)
                            _idx = max(0, min(len(_st_curric_setups) - 1,
                                              _tgt + random.randint(-_spd, _spd)))
                            episode_env = _setup_to_env(_st_curric_setups[_idx])
                            if len(state["rewards"]) >= 20:
                                _ar = float(np.mean(state["rewards"][-20:]))
                                if _ar > _st_curric_thr:
                                    state["adr_difficulty"] = min(1.0, _adr + 0.02)
                                elif _ar < _st_curric_thr * 0.3:
                                    state["adr_difficulty"] = max(0.0, _adr - 0.01)
                        else:  # Performance Threshold
                            _ns = min(10, len(_st_curric_setups))
                            _stg = state.get("curriculum_stage", 0)
                            _ps = max(1, len(_st_curric_setups) // _ns)
                            _ss = _stg * _ps
                            _se = min(len(_st_curric_setups), _ss + _ps)
                            _idx = random.randint(_ss, _se - 1)
                            episode_env = _setup_to_env(_st_curric_setups[_idx])
                            if len(state["rewards"]) >= 20:
                                _ar = float(np.mean(state["rewards"][-20:]))
                                if _ar > _st_curric_thr and _stg < _ns - 1:
                                    state["curriculum_stage"] = _stg + 1
                    else:
                        episode_env = env

                    def state_vec(s, _e=episode_env):
                        return _build_obs(_e, s, _n_nb, _use_gd)

                    # Reset episode env
                    if getattr(cfg, "exploring_starts", False):
                        _vstates = [s for s in range(episode_env.n_states)
                                    if not episode_env.is_wall(s) and not episode_env.is_terminal(s)]
                        s = int(random.choice(_vstates)) if _vstates else episode_env.reset()
                        episode_env.agent_pos = divmod(s, episode_env.cols)
                        episode_env.done = False
                    else:
                        s = episode_env.reset()

                    sv = state_vec(s)
                    total_rew = 0.0
                    steps = 0

                    for _ in range(cfg.max_steps):
                        state["visits"][s] += 1
                        valid = valid_actions(episode_env, s)
                        if random.random() < eps:
                            action = int(random.choice(valid)) if valid else random.randrange(episode_env.n_actions)
                        else:
                            with torch.no_grad():
                                s_t = torch.tensor([sv], dtype=torch.float32)
                                q_values = q_net(s_t).squeeze(0)
                                action = (
                                    int(max(valid, key=lambda a: float(q_values[a])))
                                    if valid else int(q_values.argmax().item())
                                )

                        ns, r, done = episode_env.step(action)
                        state["visits"][ns] += 1
                        next_sv = state_vec(ns)
                        total_rew += (episode_env.gamma ** steps) * r
                        steps += 1
                        total_steps += 1

                        buffer.push(sv, action, float(np.clip(r, -10.0, 10.0)), next_sv, 1.0 if done else 0.0)
                        s = ns
                        sv = next_sv

                        # Training step
                        if len(buffer) >= cfg.dqn_batch_size:
                            s_b, a_b, r_b, ns_b, d_b = buffer.sample(cfg.dqn_batch_size)

                            s_t = torch.tensor(s_b, dtype=torch.float32)
                            a_t = torch.tensor(a_b, dtype=torch.long)
                            r_t = torch.tensor(r_b, dtype=torch.float32)
                            ns_t = torch.tensor(ns_b, dtype=torch.float32)
                            d_t = torch.tensor(d_b, dtype=torch.float32)

                            current_q = q_net(s_t).gather(1, a_t.unsqueeze(1)).squeeze(1)
                            with torch.no_grad():
                                max_next_q = target_net(ns_t).max(dim=1)[0]
                                target_q = r_t + episode_env.gamma * max_next_q * (1.0 - d_t)

                            loss = nn.functional.mse_loss(current_q, target_q)
                            if not torch.isnan(loss) and not torch.isinf(loss):
                                optimizer.zero_grad()
                                loss.backward()
                                torch.nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=10.0)
                                optimizer.step()

                        if total_steps % cfg.dqn_target_update == 0:
                            target_net.load_state_dict(q_net.state_dict())

                        if done:
                            break

                    eps = max(cfg.dqn_epsilon_min, eps * cfg.dqn_epsilon_decay)
                    state["epsilon"] = eps
                    state["total_steps"] = total_steps
                    state["episodes"] += 1
                    state["rewards"].append(total_rew)
                    state["lengths"].append(steps)

                    # Extract Q-table against main env for visualisation
                    all_obs_st = [_build_obs(env, _si, _n_nb, _use_gd)
                                  for _si in range(env.n_states)]
                    with torch.no_grad():
                        Q_list = q_net(torch.tensor(all_obs_st, dtype=torch.float32)).tolist()
                    Q_table = np.array(Q_list, dtype=np.float64)
                    policy = np.argmax(Q_table, axis=1)

                    st.session_state.results = dict(
                        algo=algo_name, env=env, policy=policy, values=Q_table.max(axis=1),
                        rewards=state["rewards"], lengths=state["lengths"], visits=state.get("visits"),
                    )
                    sr["state"] = state

                else:
                    st.warning("Step-through not supported for this algorithm (use Full mode).")

                _safe_rerun()

            if c2.button("Reset Step-through"):
                del st.session_state.step_run
                st.session_state.results = None
                _safe_rerun()


# ── Handle stop-training request ──────────────────────────────────────
# Runs at script start (before any widget rendering) so we can safely write
# to widget-bound keys and then trigger a clean rerun.
if st.session_state.pop("stop_training_requested", False):
    _stop_tr = st.session_state.get("train_run")
    if _stop_tr and not _stop_tr.get("finished", True):
        _stop_state = _stop_tr.get("state") or {}
        _stop_env   = _stop_tr.get("env")
        _stop_algo  = _stop_tr.get("algo", "")

        # ── Reconstruct results from training state ──────────────────
        # Per-episode session_state.results writes may not have committed
        # if Streamlit interrupted the training thread mid-episode.
        # We always rebuild from train_run["state"] which is updated every
        # episode via the live dict reference (tr["state"] = state).
        _stop_rews  = list(_stop_state.get("rewards") or [])
        _n_done     = len(_stop_rews)

        if _stop_env and _n_done > 0:
            _stop_Q = _stop_state.get("Q")
            if _stop_Q is not None:
                # Q-Learning / SARSA
                _stop_policy = np.argmax(_stop_Q, axis=1)
                _stop_values = _stop_Q.max(axis=1)
                st.session_state.results = dict(
                    algo          = _stop_algo,
                    env           = _stop_env,
                    policy        = _stop_policy,
                    values        = _stop_values,
                    rewards       = _stop_rews,
                    lengths       = list(_stop_state.get("lengths") or []),
                    visits        = _stop_state.get("visits"),
                    episode_times = _stop_state.get("episode_times"),
                    off_policy_steps     = _stop_state.get("off_policy_steps"),
                    epsilon_values       = _stop_state.get("epsilon_values"),
                    episode_traces       = _stop_state.get("episode_traces"),
                    episode_complexities = list(_stop_state.get("episode_complexities") or []),
                )
            else:
                # DQN — per-episode results are richer; fall back to whatever
                # was last committed, but at minimum patch in the rewards list
                _existing = st.session_state.results
                if isinstance(_existing, dict) and _existing.get("env") is not None:
                    _existing["rewards"] = _stop_rews
                    _existing["lengths"] = list(_stop_state.get("lengths") or [])
                    _existing["episode_times"] = _stop_state.get("episode_times")
                    _existing["epsilon_values"] = _stop_state.get("epsilon_values")
                    st.session_state.results = _existing
                # If even that is absent, leave st.session_state.results as-is
                # (DQN sets it every episode so it should exist after ep > 0)

        # ── Annotate training summary with partial-run metadata ──────
        _stop_ts = st.session_state.get("training_summary") or {}
        _stop_ts["stopped_early"]      = True
        _stop_ts["episodes_completed"] = _n_done
        _stop_ts["total_ep"]           = _n_done
        st.session_state["training_summary"] = _stop_ts

        # ── Clean up the active training run ─────────────────────────
        st.session_state.pop("train_run", None)

        # ── Auto-save partial results ─────────────────────────────────
        try:
            _stop_path = _auto_save_run(_stop_ts, st.session_state.results or {})
            if _stop_path:
                st.session_state["last_saved_run_path"] = _stop_path
        except Exception:
            pass

    # Rerun so the results panel renders cleanly with the saved partial data
    st.rerun()


# ── Training logic ────────────────────────────────────────────────────
# Resume condition: training was started but not finished, and user hasn't clicked Stop
_tr_resume = (
    isinstance(st.session_state.get("train_run"), dict)
    and not st.session_state.get("train_run", {}).get("finished", True)
    and not st.session_state.get("stop_training_requested")
)
if train_clicked or _tr_resume:
    grid = st.session_state.grid
    rows = st.session_state.rows
    cols = st.session_state.cols

    flat = [grid[r][c] for r in range(rows) for c in range(cols)]

    # When DQN curriculum is active the drawn grid is used only for policy
    # visualisation after training; actual episodes run on curriculum envs.
    _curriculum_active = (
        algo == "DQN"
        and bool(st.session_state.get("use_curriculum", False))
        and bool(st.session_state.get("curriculum_setups"))
    )

    if ST not in flat:
        st.sidebar.error("❌ Grid needs a Start cell (S).")
        st.stop()
    if GO not in flat and not _curriculum_active:
        st.sidebar.error("❌ Grid needs at least one Goal cell (G).")
        st.stop()
    if GO not in flat and _curriculum_active:
        st.sidebar.info(
            "ℹ️ No Goal on the drawn grid — training will run entirely on the "
            f"{len(st.session_state['curriculum_setups'])} curriculum setups. "
            "The drawn grid is used only for final policy visualisation.",
            icon="🎓",
        )

    layout = _grid_to_layout(grid, rows, cols)

    grid_cfg = GridConfig(
        rows=rows, cols=cols, layout=layout,
        step_reward=step_rew, goal_reward=goal_rew, trap_reward=trap_rew,
        cell_rewards=st.session_state.get("cell_rewards", {}),
        slippery_slip_prob=slip_prob, gamma=gamma,
    )
    algo_cfg = AlgorithmConfig(
        alpha=alpha,
        n_episodes=episodes,
        dqn_n_episodes=dqn_episodes,
        max_steps=max_steps,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=0.01,
        dqn_hidden=dqn_hidden,
        dqn_lr=dqn_lr,
        dqn_batch_size=dqn_batch_size,
        dqn_buffer_size=dqn_buffer_size,
        dqn_target_update=dqn_target_update,
        dqn_epsilon=dqn_epsilon,
        dqn_epsilon_decay=dqn_epsilon_decay,
        dqn_epsilon_min=dqn_epsilon_min,
        exploring_starts=exploring_starts,
    )

    _n_cur   = len(st.session_state["curriculum_setups"]) if _curriculum_active else 0
    _method  = st.session_state.get("curriculum_method", "ADR") if _curriculum_active else ""
    _total_ep = dqn_episodes if algo == "DQN" else (1 if algo == "Policy Iteration" else episodes)

    # ── Observation settings (always computed, even for non-DQN) ───────
    _n_nb_now   = int(st.session_state.get("obs_n_neighbors", 0))
    _use_gd_now = bool(st.session_state.get("obs_use_goal_dist", False))
    if _n_nb_now == 0 and not _use_gd_now:
        _inp_dim_now = rows * cols
    else:
        _inp_dim_now = 2 + ((2 * _n_nb_now + 1) ** 2 - 1 if _n_nb_now > 0 else 0) + (2 if _use_gd_now else 0)

    # ── Grid cell counts ────────────────────────────────────────────────
    _gc: dict = {}
    for _cn, _cv in [("walls", 1), ("slippery", 2), ("goals", 3), ("traps", 4)]:
        _gc[_cn] = int(np.sum(grid == _cv))
    _gc["empty"] = int(np.sum(grid == 0))

    # ── Compact hp_rows (for the status panel captions) ─────────────────
    _hp_rows: list[str] = []
    if algo == "Policy Iteration":
        _hp_rows.append(f"γ={gamma:.2f} · step={step_rew:+.2f} · goal={goal_rew:+.2f} · trap={trap_rew:+.2f}")
    elif algo in ("Q-Learning", "SARSA"):
        _hp_rows.append(f"α={alpha:.4f}  ε₀={epsilon:.3f}  decay={epsilon_decay:.4f}  max_steps={max_steps}")
        _hp_rows.append(f"γ={gamma:.2f} · step={step_rew:+.2f} · goal={goal_rew:+.2f} · trap={trap_rew:+.2f} · slip={slip_prob:.2f}")
    elif algo == "DQN":
        _hp_rows.append(f"lr={dqn_lr:.5f}  batch={dqn_batch_size}  buffer={dqn_buffer_size}  target_update={dqn_target_update}")
        _hp_rows.append(f"hidden={dqn_hidden}  ε₀={dqn_epsilon:.3f}→{dqn_epsilon_min:.3f}×{dqn_epsilon_decay:.4f}")
        _hp_rows.append(f"γ={gamma:.2f} · step={step_rew:+.2f} · goal={goal_rew:+.2f} · trap={trap_rew:+.2f} · slip={slip_prob:.2f}")
        _obs_label = f"obs: n_neighbors={_n_nb_now}"
        if _use_gd_now:
            _obs_label += " + goal_dist"
        _obs_label += f"  →  input_dim={_inp_dim_now}"
        _hp_rows.append(_obs_label)
        if _curriculum_active:
            _hp_rows.append(f"🎓 Curriculum: {_method} · {_n_cur} setups · threshold {st.session_state.get('curriculum_perf_threshold', 0.5):.2f}")

    # ── Comprehensive training summary (persisted in session state) ──────
    _n_test_setups = len(st.session_state.get("curriculum_test_setups", []))
    st.session_state["training_summary"] = {
        # Identity
        "algo":             algo,
        "run_mode":         run_mode,
        # Grid
        "rows":             rows,
        "cols":             cols,
        "total_ep":         _total_ep,
        "cell_counts":      _gc,
        "n_custom_cell_rewards": len(st.session_state.get("cell_rewards", {})),
        # Rewards & dynamics
        "step_rew":         float(step_rew),
        "goal_rew":         float(goal_rew),
        "trap_rew":         float(trap_rew),
        "gamma":            float(gamma),
        "slip_prob":        float(slip_prob),
        # Common training
        "max_steps":        int(max_steps),
        "exploring_starts": bool(exploring_starts),
        # Q / SARSA params
        "alpha":            float(st.session_state.get("alpha", 0.1)),
        "epsilon":          float(st.session_state.get("epsilon", 1.0)),
        "epsilon_decay":    float(st.session_state.get("epsilon_decay", 0.998)),
        "epsilon_min":      0.01,
        # DQN params
        "dqn_lr":           float(st.session_state.get("dqn_lr", 1e-3)),
        "dqn_hidden":       list(dqn_hidden) if algo == "DQN" else None,
        "dqn_batch":        int(st.session_state.get("dqn_batch_size", 64)),
        "dqn_buffer":       int(st.session_state.get("dqn_buffer_size", 10_000)),
        "dqn_target_update":int(st.session_state.get("dqn_target_update", 100)),
        "dqn_epsilon":      float(st.session_state.get("dqn_epsilon", 1.0)),
        "dqn_epsilon_decay":float(st.session_state.get("dqn_epsilon_decay", 0.997)),
        "dqn_epsilon_min":  float(st.session_state.get("dqn_epsilon_min", 0.01)),
        # Observation (DQN)
        "obs_n_neighbors":  _n_nb_now,
        "obs_use_goal_dist":_use_gd_now,
        "obs_input_dim":    _inp_dim_now,
        # Curriculum
        "curriculum_active":     bool(_curriculum_active),
        "curriculum_method":     _method,
        "curriculum_n_train":    int(_n_cur),
        "curriculum_n_test":     int(_n_test_setups),
        "curriculum_threshold":  float(st.session_state.get("curriculum_perf_threshold", 0.5)),
        # Reward shaping
        "potential_shaping":     bool(st.session_state.get("use_potential_shaping", True)),
        "shaping_alpha":         round(1.0 / max(1, rows + cols), 4),
        # Legacy captions for the status panel
        "hp_rows":          _hp_rows,
    }

    # ── Training status panel (shown in the results placeholder) ─────
    # Peek at saved state so we can restore progress on resume reruns
    _saved_tr_peek    = st.session_state.get("train_run") or {}
    _saved_state_peek = _saved_tr_peek.get("state") or {}
    _ep_restored      = len(_saved_state_peek.get("rewards") or [])
    _init_frac        = min(1.0, _ep_restored / max(1, _total_ep)) if _ep_restored > 0 else 0.0
    _init_text        = (
        f"Episode {_ep_restored:,} / {_total_ep:,}" if _ep_restored > 0 else "Initializing…"
    )

    with _train_progress_ph.container():
        with st.status("🏋️ Training in progress…", expanded=True) as _train_status:
            # ── Training plan summary ────────────────────────────────
            _sv1, _sv2, _sv3 = st.columns(3)
            _sv1.metric("Algorithm", algo)
            _sv2.metric("Grid", f"{rows}×{cols}")
            _sv3.metric("Episodes", f"{_total_ep:,}")
            if _hp_rows:
                for _hr in _hp_rows:
                    st.caption(_hr)
            st.divider()
            # ── Live progress area ───────────────────────────────────
            _prog_bar = st.progress(_init_frac, text=_init_text)
            _is_dqn_run = (algo == "DQN")
            if _is_dqn_run:
                _lm1, _lm2, _lm3, _lm4, _lm5 = st.columns(5)
            else:
                _lm1, _lm2, _lm3, _lm4 = st.columns(4)
                _lm5 = None
            _ep_ph   = _lm1.empty()
            _rw_ph   = _lm2.empty()
            _eps_ph  = _lm3.empty()
            _ela_ph  = _lm4.empty()
            _buf_ph  = _lm5.empty() if _lm5 is not None else None
            _cur_ph  = st.empty() if _curriculum_active else None

            # ── Restore live metrics immediately on resume reruns ────
            if _ep_restored > 0:
                _rews_peek = _saved_state_peek.get("rewards") or []
                _avg_r_peek = float(
                    sum(_rews_peek[-20:]) / max(1, len(_rews_peek[-20:]))
                ) if _rews_peek else 0.0
                _eps_peek = float(
                    _saved_state_peek.get("epsilon",
                    _saved_state_peek.get("dqn_epsilon", 0.0))
                )
                _elapsed_peek = time.time() - _saved_tr_peek.get("started_at", time.time())
                _eta_peek = (
                    (_elapsed_peek / _ep_restored) * max(0, _total_ep - _ep_restored)
                ) if _ep_restored > 0 else 0.0
                _ep_ph.metric("Episode", f"{_ep_restored:,}", f"/ {_total_ep:,}")
                _rw_ph.metric("Avg Reward", f"{_avg_r_peek:+.3f}")
                _eps_ph.metric("Epsilon", f"{_eps_peek:.3f}")
                _ela_ph.metric(
                    "Elapsed / ETA",
                    f"{int(_elapsed_peek)}s",
                    f"≈{int(_eta_peek)}s left" if _eta_peek > 0 else None,
                )
                if _buf_ph is not None:
                    _buf_peek = _saved_state_peek.get("buffer")
                    _buf_len  = len(_buf_peek) if _buf_peek is not None else 0
                    _buf_cap  = _saved_tr_peek.get("cfg") and _saved_tr_peek["cfg"].dqn_buffer_size or 1
                    _buf_ph.metric(
                        "Replay buffer",
                        f"{_buf_len:,}",
                        f"/ {_buf_cap:,}",
                        help="Number of transitions stored in the DQN replay buffer.",
                    )
            # ── Stop button lives here so it's always visible ────────
            st.divider()
            if st.button(
                "⏹ Stop Training",
                key="stop_training_btn_main",
                use_container_width=True,
                type="primary",
                help="Stop now and show results collected so far",
            ):
                st.session_state["stop_training_requested"] = True
                st.rerun()

    env = GridWorld(grid_cfg)
    # If step-through mode requested, initialise controller and don't run full training
    if run_mode == "Step-through":
        sr = st.session_state.setdefault("step_run", {})
        sr["algo"] = algo
        sr["cfg"] = algo_cfg
        sr["env"] = env
        sr.setdefault("state", None)
        sr.setdefault("finished", False)
        _train_status.update(label="⏸ Step-through mode ready — use the controls below", state="complete")
        _safe_rerun()

    if algo == "Policy Iteration":
        _prog_bar.progress(0.3, text="Running Policy Iteration…")
        pol, vals, hist = policy_iteration(env, algo_cfg)
        st.session_state.results = dict(
            algo=algo, env=env, policy=pol, values=vals,
            rewards=None, lengths=None, pi_iters=len(hist),
        )
        _prog_bar.progress(1.0, text=f"Complete — {len(hist)} iterations")
    else:
        # Resumable full training loop stored in session_state['train_run']
        if not isinstance(st.session_state.get("train_run"), dict):
            st.session_state.train_run = {}
        tr = st.session_state.train_run
        tr.setdefault("algo", algo)
        tr.setdefault("cfg", algo_cfg)
        tr.setdefault("env", env)
        tr.setdefault("state", None)
        tr.setdefault("finished", False)
        tr.setdefault("last_update", 0.0)
        tr.setdefault("started_at", time.time())

        cfg = tr["cfg"]
        env = tr["env"]

        # ── Potential-based shaping setup ─────────────────────────────────
        _use_shaping = bool(st.session_state.get("use_potential_shaping", True))
        # α = goal_reward / (rows + cols)  — precomputed once for the main env
        _shaping_alpha = 1.0 / max(1, env.rows + env.cols)
        # BFS distances from goal on the main env (used for Q-Learning / SARSA)
        # Stored in tr so they survive batch reruns without recomputation.
        if _use_shaping and "shaping_dist" not in tr:
            tr["shaping_dist"] = _bfs_goal_distances(env)
        _shaping_dist = tr.get("shaping_dist") or []

        # Initialize per-algo state if needed
        state = tr.get("state") or {}
        if state == {}:
            if algo == "Q-Learning" or algo == "SARSA":
                state["Q"] = np.zeros((env.n_states, env.n_actions), dtype=np.float64)
                state["epsilon"] = cfg.epsilon
                state["episode"] = 0
                state["rewards"] = []
                state["lengths"] = []
                state["episode_times"] = []
                state["off_policy_steps"] = []
                state["epsilon_values"] = []
                state["episode_traces"] = []
                state["episode_traces_all"] = []
                state["visits"] = np.zeros(env.n_states, dtype=int)
            elif algo == "DQN":
                _n_nb = int(st.session_state.get("obs_n_neighbors", 0))
                _use_gd = bool(st.session_state.get("obs_use_goal_dist", False))
                _inp_dim = _obs_dim(env, _n_nb, _use_gd)

                # ── Curriculum dimension check ────────────────────
                # The curriculum setups must have been created with the
                # same grid size as the current env; if not, the
                # observation vectors are a different length and the
                # forward/backward pass will silently produce wrong
                # shapes (or crash).  Catch this before building the
                # network.
                _cur_setups_check = st.session_state.get("curriculum_setups") or []
                if _cur_setups_check and st.session_state.get("use_curriculum"):
                    _chk = _cur_setups_check[0]
                    if _chk["rows"] != env.rows or _chk["cols"] != env.cols:
                        st.error(
                            f"⚠️ Curriculum setup grid size "
                            f"({_chk['rows']}×{_chk['cols']}) doesn't match "
                            f"the current grid ({env.rows}×{env.cols}).  "
                            "Please click **Create Setups** again to regenerate "
                            "them for the current grid size."
                        )
                        st.stop()

                device = torch.device("cpu")
                q_net = QNetwork(_inp_dim, env.n_actions, cfg.dqn_hidden).to(device)
                target_net = QNetwork(_inp_dim, env.n_actions, cfg.dqn_hidden).to(device)
                target_net.load_state_dict(q_net.state_dict())
                target_net.eval()
                optimizer = optim.Adam(q_net.parameters(), lr=cfg.dqn_lr)
                buffer = ReplayBuffer(cfg.dqn_buffer_size)
                state.update({
                    "q_net": q_net,
                    "target_net": target_net,
                    "optimizer": optimizer,
                    "buffer": buffer,
                    "epsilon": cfg.dqn_epsilon,
                    "total_steps": 0,
                    "episodes": 0,
                    "rewards": [],
                    "lengths": [],
                    "episode_times": [],
                    "off_policy_steps": [],
                    "epsilon_values": [],
                    "episode_traces": [],
                    "episode_traces_all": [],
                    "visits": np.zeros(env.n_states, dtype=int),
                    "obs_n_neighbors": _n_nb,
                    "obs_use_goal_dist": _use_gd,
                    "adr_difficulty": 0.0,
                    "curriculum_stage": 0,
                    "episode_complexities": [],
                })
                # Pre-build all curriculum environments ONCE so we never call
                # _setup_to_env() (and its _build_transition_table()) per episode.
                _cs_init = st.session_state.get("curriculum_setups") or []
                state["_curric_envs"] = [_setup_to_env(s) for s in _cs_init]
            tr["state"] = state

        now = time.time()

        # Run episodes until we need to update UI or finish
        update_interval = 5.0
        start_time = time.time()
        # loop and perform episodes
        while True:
            if tr.get("finished"):
                break
            episode_start_time = time.time()
            if algo == "Q-Learning":
                Q = state["Q"]
                eps = state["epsilon"]

                # one episode
                if getattr(cfg, "exploring_starts", False):
                    valid_states = [s for s in range(env.n_states) if not env.is_wall(s) and not env.is_terminal(s)]
                    s = int(random.choice(valid_states))
                    env.agent_pos = divmod(s, env.cols)
                    env.done = False
                else:
                    s = env.reset()
                episode_trace = {
                    "states": [s],
                    "actions": [],
                    "rewards": [],
                    "next_states": [],
                    "off_policy_steps": 0,
                    "grid": env.grid.tolist(),
                    "cell_rewards": {f"{r},{c}": float(v) for (r, c), v in env.config.cell_rewards.items()},
                    "start_pos": list(env.start_pos),
                    "complexity": 0.0,
                }
                off_steps = 0
                total_reward = 0.0
                steps = 0
                for _ in range(cfg.max_steps):
                    state["visits"][s] += 1
                    valid = valid_actions(env, s)
                    if np.random.random() < eps:
                        a = int(random.choice(valid))
                        off_steps += 1
                    else:
                        a = int(max(valid, key=lambda act: Q[s, act])) if valid else int(np.argmax(Q[s]))
                    ns, r, done = env.step(a)
                    state["visits"][ns] += 1
                    episode_trace["actions"].append(a)
                    episode_trace["rewards"].append(r)
                    episode_trace["next_states"].append(ns)
                    episode_trace["states"].append(ns)
                    # Potential-based shaping (added to TD target only, not logged)
                    _r_shaped = r + (
                        _potential_shaping(s, ns, _shaping_dist, env.gamma, _shaping_alpha)
                        if _use_shaping and _shaping_dist else 0.0
                    )
                    # Accumulate discounted return: use env.gamma^t * r_t
                    total_reward += (env.gamma ** steps) * r
                    steps += 1
                    best_next = 0.0 if done else float(np.max(Q[ns]))
                    Q[s, a] += cfg.alpha * (_r_shaped + env.gamma * best_next - Q[s, a])
                    s = ns
                    if done:
                        break

                eps = max(cfg.epsilon_min, eps * cfg.epsilon_decay)
                state["epsilon"] = eps
                state["episode"] += 1
                state["rewards"].append(total_reward)
                state["lengths"].append(steps)
                state["episode_times"].append(time.time() - episode_start_time)
                state["off_policy_steps"].append(off_steps)
                state["epsilon_values"].append(eps)
                episode_trace["off_policy_steps"] = off_steps
                if st.session_state.get("record_episodes", False):
                    state["episode_traces_all"].append(episode_trace)
                    limit = st.session_state.get("record_limit", 100)
                    if st.session_state.get("record_strategy", "Most recent") == "Subsample evenly":
                        state["episode_traces"] = _evenly_subsample_traces(state["episode_traces_all"], limit)
                    else:
                        state["episode_traces"] = state["episode_traces_all"][-limit:]

                policy = np.argmax(Q, axis=1)
                st.session_state.results = dict(
                    algo=algo, env=env, policy=policy, values=Q.max(axis=1),
                    rewards=state["rewards"], lengths=state["lengths"], visits=state.get("visits"),
                    episode_times=state.get("episode_times"), off_policy_steps=state.get("off_policy_steps"),
                    epsilon_values=state.get("epsilon_values"), episode_traces=state.get("episode_traces"),
                    episode_complexities=list(state.get("episode_complexities") or []),
                )

            elif algo == "SARSA":
                Q = state["Q"]
                eps = state["epsilon"]

                def eps_greedy(s_):
                    # Return (action, was_random). For terminal states, return dummy action.
                    if env.is_terminal(s_):
                        return 0, False
                    valid = valid_actions(env, s_)
                    if np.random.random() < eps:
                        return int(random.choice(valid)), True
                    if valid:
                        return int(max(valid, key=lambda act: Q[s_, act])), False
                    return int(np.argmax(Q[s_])), False

                if getattr(cfg, "exploring_starts", False):
                    valid_states = [s for s in range(env.n_states) if not env.is_wall(s) and not env.is_terminal(s)]
                    s = int(random.choice(valid_states))
                    env.agent_pos = divmod(s, env.cols)
                    env.done = False
                else:
                    s = env.reset()
                episode_trace = {
                    "states": [s],
                    "actions": [],
                    "rewards": [],
                    "next_states": [],
                    "off_policy_steps": 0,
                    "grid": env.grid.tolist(),
                    "cell_rewards": {f"{r},{c}": float(v) for (r, c), v in env.config.cell_rewards.items()},
                    "start_pos": list(env.start_pos),
                    "complexity": 0.0,
                }
                a, off = eps_greedy(s)
                total_reward = 0.0
                steps = 0
                off_steps = 1 if off else 0
                for _ in range(cfg.max_steps):
                    state["visits"][s] += 1
                    ns, r, done = env.step(a)
                    state["visits"][ns] += 1
                    episode_trace["actions"].append(a)
                    episode_trace["rewards"].append(r)
                    episode_trace["next_states"].append(ns)
                    episode_trace["states"].append(ns)
                    na, next_off = eps_greedy(ns)
                    _r_shaped = r + (
                        _potential_shaping(s, ns, _shaping_dist, env.gamma, _shaping_alpha)
                        if _use_shaping and _shaping_dist else 0.0
                    )
                    total_reward += (env.gamma ** steps) * r
                    steps += 1
                    off_steps += 1 if next_off else 0
                    next_q = 0.0 if done else Q[ns, na]
                    Q[s, a] += cfg.alpha * (_r_shaped + env.gamma * next_q - Q[s, a])
                    s, a = ns, na
                    if done:
                        break

                eps = max(cfg.epsilon_min, eps * cfg.epsilon_decay)
                state["epsilon"] = eps
                state["episode"] += 1
                state["rewards"].append(total_reward)
                state["lengths"].append(steps)
                state["episode_times"].append(time.time() - episode_start_time)
                state["off_policy_steps"].append(off_steps)
                state["epsilon_values"].append(eps)
                episode_trace["off_policy_steps"] = off_steps
                if st.session_state.get("record_episodes", False):
                    state["episode_traces_all"].append(episode_trace)
                    limit = st.session_state.get("record_limit", 100)
                    if st.session_state.get("record_strategy", "Most recent") == "Subsample evenly":
                        state["episode_traces"] = _evenly_subsample_traces(state["episode_traces_all"], limit)
                    else:
                        state["episode_traces"] = state["episode_traces_all"][-limit:]

                policy = np.argmax(Q, axis=1)
                st.session_state.results = dict(
                    algo=algo, env=env, policy=policy, values=Q.max(axis=1),
                    rewards=state["rewards"], lengths=state["lengths"], visits=state.get("visits"),
                    episode_times=state.get("episode_times"), off_policy_steps=state.get("off_policy_steps"),
                    epsilon_values=state.get("epsilon_values"), episode_traces=state.get("episode_traces"),
                    episode_complexities=list(state.get("episode_complexities") or []),
                )

            elif algo == "DQN":
                q_net = state["q_net"]
                target_net = state["target_net"]
                optimizer = state["optimizer"]
                buffer = state["buffer"]
                eps = state["epsilon"]
                total_steps = state.get("total_steps", 0)
                _n_nb = state.get("obs_n_neighbors", 0)
                _use_gd = state.get("obs_use_goal_dist", False)

                # ── Curriculum: pick episode environment ──────────
                _use_curric = st.session_state.get("use_curriculum", False)
                _curric_setups = st.session_state.get("curriculum_setups") or []
                _curric_method = st.session_state.get("curriculum_method", "ADR")
                _curric_thr = float(st.session_state.get("curriculum_perf_threshold", 0.5))

                # Use the pre-cached env list (built once at init) to avoid
                # rebuilding GridWorld + transition table every episode.
                _curric_envs = state.get("_curric_envs", [])

                if _use_curric and _curric_setups and _curric_envs:
                    if _curric_method == "ADR":
                        _adr_diff = state.get("adr_difficulty", 0.0)
                        _tgt = int(_adr_diff * (len(_curric_setups) - 1))
                        # Spread: ±5% of the setup list around the target index
                        _spread = max(1, len(_curric_setups) // 20)
                        _ep_idx = max(0, min(len(_curric_setups) - 1,
                                              _tgt + random.randint(-_spread, _spread)))
                        episode_env = _curric_envs[_ep_idx]
                        if len(state["rewards"]) >= 20:
                            _avg_r = float(np.mean(state["rewards"][-20:]))
                            # Advance rate: reach full difficulty in roughly 50% of
                            # total episodes when consistently above threshold.
                            _adr_step = max(0.002, 2.0 / max(1, cfg.dqn_n_episodes))
                            if _avg_r > _curric_thr:
                                state["adr_difficulty"] = min(1.0, _adr_diff + _adr_step)
                            elif _avg_r < _curric_thr * 0.3:
                                state["adr_difficulty"] = max(0.0, _adr_diff - _adr_step / 2)
                    else:  # Performance Threshold
                        _n_stages = min(10, len(_curric_setups))
                        _stage = state.get("curriculum_stage", 0)
                        _per_stage = max(1, len(_curric_setups) // _n_stages)
                        _s_start = _stage * _per_stage
                        _s_end = min(len(_curric_setups), _s_start + _per_stage)
                        _ep_idx = random.randint(_s_start, _s_end - 1)
                        episode_env = _curric_envs[_ep_idx]
                        if len(state["rewards"]) >= 20:
                            _avg_r = float(np.mean(state["rewards"][-20:]))
                            if _avg_r > _curric_thr and _stage < _n_stages - 1:
                                state["curriculum_stage"] = _stage + 1
                    # Record this episode's complexity (1–10 scale) for live display
                    _this_ep_complexity = float(_curric_setups[_ep_idx].get("complexity", 0.0))
                    state.setdefault("episode_complexities", []).append(_this_ep_complexity)
                else:
                    episode_env = env
                    _this_ep_complexity = 0.0

                def state_vec(s, _e=episode_env):
                    return _build_obs(_e, s, _n_nb, _use_gd)

                # BFS distances for this episode's env (curriculum may use different grids)
                _ep_shaping_dist = (
                    _bfs_goal_distances(episode_env) if _use_shaping else []
                )
                _ep_shaping_alpha = 1.0 / max(1, episode_env.rows + episode_env.cols)

                if getattr(cfg, "exploring_starts", False):
                    valid_states = [s for s in range(episode_env.n_states)
                                    if not episode_env.is_wall(s) and not episode_env.is_terminal(s)]
                    s = int(random.choice(valid_states))
                    episode_env.agent_pos = divmod(s, episode_env.cols)
                    episode_env.done = False
                else:
                    s = episode_env.reset()
                episode_trace = {
                    "states": [s],
                    "actions": [],
                    "rewards": [],
                    "next_states": [],
                    "off_policy_steps": 0,
                    "grid": episode_env.grid.tolist(),
                    "cell_rewards": {f"{r},{c}": float(v) for (r, c), v in episode_env.config.cell_rewards.items()},
                    "start_pos": list(episode_env.start_pos),
                    "complexity": _this_ep_complexity,
                }
                sv = state_vec(s)
                total_rew = 0.0
                steps = 0
                off_steps = 0

                for _ in range(cfg.max_steps):
                    state["visits"][s] += 1
                    valid = valid_actions(episode_env, s)
                    if random.random() < eps:
                        action = int(random.choice(valid))
                        off_steps += 1
                    else:
                        with torch.no_grad():
                            s_t = torch.tensor([sv], dtype=torch.float32)
                            q_values = q_net(s_t).squeeze(0)
                            if valid:
                                action = int(max(valid, key=lambda act: float(q_values[act])))
                            else:
                                action = int(q_values.argmax().item())

                    ns, r, done = episode_env.step(action)
                    state["visits"][ns] += 1
                    episode_trace["actions"].append(action)
                    episode_trace["rewards"].append(r)
                    episode_trace["next_states"].append(ns)
                    episode_trace["states"].append(ns)
                    next_sv = state_vec(ns)
                    _r_shaped = r + (
                        _potential_shaping(s, ns, _ep_shaping_dist, episode_env.gamma, _ep_shaping_alpha)
                        if _use_shaping and _ep_shaping_dist else 0.0
                    )
                    total_rew += (episode_env.gamma ** steps) * r
                    steps += 1
                    total_steps += 1

                    buffer.push(sv, action, float(np.clip(_r_shaped, -10.0, 10.0)), next_sv, 1.0 if done else 0.0)
                    s = ns
                    sv = next_sv

                    if len(buffer) >= cfg.dqn_batch_size:
                        s_b, a_b, r_b, ns_b, d_b = buffer.sample(cfg.dqn_batch_size)
                        s_t = torch.tensor(s_b, dtype=torch.float32)
                        a_t = torch.tensor(a_b, dtype=torch.long)
                        r_t = torch.tensor(r_b, dtype=torch.float32)
                        ns_t = torch.tensor(ns_b, dtype=torch.float32)
                        d_t = torch.tensor(d_b, dtype=torch.float32)

                        current_q = q_net(s_t).gather(1, a_t.unsqueeze(1)).squeeze(1)
                        with torch.no_grad():
                            max_next_q = target_net(ns_t).max(dim=1)[0]
                            target_q = r_t + episode_env.gamma * max_next_q * (1.0 - d_t)

                        loss = nn.functional.mse_loss(current_q, target_q)
                        if not torch.isnan(loss) and not torch.isinf(loss):
                            optimizer.zero_grad()
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=10.0)
                            optimizer.step()

                    if total_steps % cfg.dqn_target_update == 0:
                        target_net.load_state_dict(q_net.state_dict())

                    if done:
                        break

                eps = max(cfg.dqn_epsilon_min, eps * cfg.dqn_epsilon_decay)
                state["epsilon"] = eps
                state["total_steps"] = total_steps
                state["episodes"] += 1
                state["rewards"].append(total_rew)
                state["lengths"].append(steps)
                state["episode_times"].append(time.time() - episode_start_time)
                state["off_policy_steps"].append(off_steps)
                state["epsilon_values"].append(eps)
                episode_trace["off_policy_steps"] = off_steps
                if st.session_state.get("record_episodes", False):
                    state["episode_traces_all"].append(episode_trace)
                    limit = st.session_state.get("record_limit", 100)
                    if st.session_state.get("record_strategy", "Most recent") == "Subsample evenly":
                        state["episode_traces"] = _evenly_subsample_traces(state["episode_traces_all"], limit)
                    else:
                        state["episode_traces"] = state["episode_traces_all"][-limit:]

                # Extract Q-table against main env for visualisation
                all_obs = [_build_obs(env, _si, _n_nb, _use_gd) for _si in range(env.n_states)]
                with torch.no_grad():
                    Q_list = q_net(torch.tensor(all_obs, dtype=torch.float32)).tolist()
                Q_table = np.array(Q_list, dtype=np.float64)
                policy = np.argmax(Q_table, axis=1)
                st.session_state.results = dict(
                    algo=algo, env=env, policy=policy, values=Q_table.max(axis=1),
                    rewards=state["rewards"], lengths=state["lengths"], visits=state.get("visits"),
                    episode_times=state.get("episode_times"), off_policy_steps=state.get("off_policy_steps"),
                    epsilon_values=state.get("epsilon_values"), episode_traces=state.get("episode_traces"),
                    episode_complexities=list(state.get("episode_complexities") or []),
                    # Saved for post-training testing
                    q_net=q_net, obs_n_neighbors=_n_nb, obs_use_goal_dist=_use_gd,
                )

            # persist state
            tr["state"] = state

            # Check for UI update interval
            now = time.time()
            if now - tr.get("last_update", 0.0) >= update_interval:
                tr["last_update"] = now
                # ── Live progress update ─────────────────────────────
                _ep_done  = len(state.get("rewards", []))
                _frac     = min(1.0, _ep_done / max(1, _total_ep))
                _avg_r    = float(np.mean(state["rewards"][-20:])) if state.get("rewards") else 0.0
                _cur_eps  = float(state.get("epsilon", state.get("dqn_epsilon", 0.0)))
                _elapsed  = now - tr.get("started_at", now)
                _eta_s    = (_elapsed / max(1, _ep_done)) * max(0, _total_ep - _ep_done) if _ep_done > 0 else 0.0
                _prog_bar.progress(_frac, text=f"Episode {_ep_done:,} / {_total_ep:,}")
                _ep_ph.metric("Episode", f"{_ep_done:,}", f"/ {_total_ep:,}")
                _rw_ph.metric("Avg Reward", f"{_avg_r:+.3f}")
                _eps_ph.metric("Epsilon", f"{_cur_eps:.3f}")
                _ela_ph.metric(
                    "Elapsed / ETA",
                    f"{int(_elapsed)}s",
                    f"≈{int(_eta_s)}s left" if _eta_s > 0 else None,
                )
                if _buf_ph is not None:
                    _live_buf = state.get("buffer")
                    _live_buf_len = len(_live_buf) if _live_buf is not None else 0
                    _live_buf_cap = cfg.dqn_buffer_size
                    _buf_ph.metric(
                        "Replay buffer",
                        f"{_live_buf_len:,}",
                        f"/ {_live_buf_cap:,}",
                        help="Transitions in replay buffer. Training starts once ≥ batch size.",
                    )
                if _cur_ph is not None:
                    _ep_cx = state.get("episode_complexities", [])
                    _avg_cx = float(np.mean(_ep_cx[-20:])) if _ep_cx else 0.0
                    _prev_cx = float(np.mean(_ep_cx[-40:-20])) if len(_ep_cx) >= 40 else None
                    _cx_delta = f"{_avg_cx - _prev_cx:+.2f}" if _prev_cx is not None else None
                    if _method == "ADR":
                        _cur_ph.metric(
                            "Avg Complexity (20)",
                            f"{_avg_cx:.2f} / 10",
                            _cx_delta,
                            help="Mean complexity of the last 20 curriculum episodes (1 = easiest, 10 = hardest). "
                                 "Delta vs the previous 20-episode window.",
                        )
                    else:
                        _cstage = state.get("curriculum_stage", 0)
                        _cur_ph.metric(
                            "Avg Complexity (20)",
                            f"{_avg_cx:.2f} / 10",
                            f"Stage {_cstage}/{min(10, _n_cur)}",
                            help="Mean complexity of the last 20 curriculum episodes.",
                        )
                # ─────────────────────────────────────────────────────
                # Save state and break out to let Streamlit process events
                # (the st.rerun() at the end of this block will continue training)
                st.session_state.train_run = tr
                break  # yield to Streamlit; next rerun picks up via _tr_resume

            # exit if finished
            if (algo == "DQN" and len(state.get("rewards", [])) >= cfg.dqn_n_episodes) or \
               (algo in ("Q-Learning", "SARSA") and len(state.get("rewards", [])) >= cfg.n_episodes):
                tr["finished"] = True
                break

        # If finished, set final results and clear train_run
        if tr.get("finished"):
            st.session_state.results = st.session_state.results or {}
            # make sure final results are stored
            st.session_state.train_run = tr
            # cleanup controller
            try:
                del st.session_state["train_run"]
            except Exception:
                pass

    # ── Finalise only when the run is actually done ───────────────────
    if tr.get("finished"):
        try:
            _ep_done = len(
                (st.session_state.results or {}).get("rewards") or []
            )
            _prog_bar.progress(1.0, text=f"Complete — {_ep_done:,} episodes")
            _ep_ph.metric("Episode", f"{_ep_done:,}", f"/ {_total_ep:,}")
            _elapsed_final = time.time() - (
                st.session_state.get("train_run", {}).get("started_at", time.time())
            )
            _ela_ph.metric("Elapsed / ETA", f"{int(_elapsed_final)}s", "done ✓")
            _train_status.update(label="✅ Training complete!", state="complete")
        except Exception:
            pass

        # ── Auto-save run to results/ folder ──────────────────────────
        try:
            _saved_path = _auto_save_run(
                st.session_state.get("training_summary", {}),
                st.session_state.results or {},
            )
            if _saved_path:
                st.session_state["last_saved_run_path"] = _saved_path
        except Exception:
            pass

    st.rerun()
