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
import random
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
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea {
    color: #0f172a !important;
    background-color: #ffffff !important;
    caret-color: #0f172a !important;
}
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] {
    padding: 0 0.5rem;
}
[data-testid="stSidebar"] hr { border-color: #334155; margin: 0.8rem 0; }
[data-testid="stSidebar"] label { font-size: 0.8rem !important; color: #94a3b8 !important; }
[data-testid="stSidebar"] h3 {
    font-size: 0.7rem !important; font-weight: 600 !important;
    color: #6366f1 !important; letter-spacing: 0.08em;
    text-transform: uppercase; margin: 0.6rem 0 0.3rem;
}
[data-testid="stSidebar"] .stButton > button {
    background: #1e293b; border: 1px solid #334155;
    color: #e2e8f0 !important; border-radius: 8px;
    font-size: 0.8rem; padding: 0.3rem 0.6rem;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #334155; border-color: #6366f1;
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


# Initialize selector early so it's available when helpers run
if "CLAUDE_MODEL" not in st.session_state:
    _init_claude_model_selector()
CELLS = [
    ("Empty",    ".",  "#F1F5F9", "#CBD5E1", "",    "#334155"),
    ("Wall",     "#",  "#1E293B", "#0F172A", "■",   "#94A3B8"),
    ("Slippery", "~",  "#BAE6FD", "#7DD3FC", "≈",   "#0369A1"),
    ("Goal",     "G",  "#BBF7D0", "#4ADE80", "G",   "#166534"),
    ("Trap",     "X",  "#FECACA", "#F87171", "X",   "#991B1B"),
    ("Start",    "S",  "#FEF08A", "#FACC15", "S",   "#713F12"),
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


def _grid_to_layout(grid, rows, cols):
    return ["".join(CELL_CHAR[grid[r][c]] for c in range(cols))
            for r in range(rows)]


def _layout_to_grid(layout):
    return [[CHAR_TO_IDX[ch] for ch in row] for row in layout]


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

    if pos_factor > 1.1:
        pos_rewards = min(len(reward_spots), pos_rewards + 1)
    if neg_factor > 1.1:
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


def _call_claude(prompt_text):
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
        "max_tokens": 300,
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


def make_editor_fig(grid, rows, cols, trace=None, current_step=None):
    cell = _cell_px(rows, cols)
    fig = go.Figure()

    for r in range(rows):
        for c in range(cols):
            t  = grid[r][c]
            y0 = rows - 1 - r
            fig.add_shape(
                type="rect", layer="below",
                x0=c+0.05, y0=y0+0.05, x1=c+0.95, y1=y0+0.95,
                fillcolor=CELL_FILL[t],
                line=dict(color=CELL_LINE[t], width=1.5),
            )
            lbl = CELL_LBL[t]
            if lbl:
                fig.add_annotation(
                    x=c+0.5, y=y0+0.5, text=f"<b>{lbl}</b>",
                    showarrow=False,
                    font=dict(size=max(13, cell // 4), color=CELL_FG[t]),
                    xanchor="center", yanchor="middle",
                )
            # show custom reward if set for this cell
            try:
                cr = st.session_state.get("cell_rewards", {}).get((r, c), None)
            except Exception:
                cr = None
            if cr is not None:
                # colored badge in bottom-right of cell for custom reward
                badge_color = "#ECFDF5" if float(cr) >= 0 else "#FEF3F2"
                badge_border = "#86EFAC" if float(cr) >= 0 else "#FCA5A5"
                badge_fg = "#065F46" if float(cr) >= 0 else "#9B1C1C"
                fig.add_annotation(
                    x=c+0.78, y=y0+0.22,
                    text=f"<span style='background:{badge_color};border:1px solid {badge_border};padding:2px 6px;border-radius:8px;color:{badge_fg};font-weight:600'>{float(cr):+.2f}</span>",
                    showarrow=False, font=dict(size=9), xanchor="center", yanchor="middle", opacity=0.95,
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

    for r in range(rows):
        for c in range(cols):
            t = int(env.grid[r, c])
            y0 = rows - 1 - r
            fig.add_shape(
                type="rect", layer="below",
                x0=c + 0.05, y0=y0 + 0.05, x1=c + 0.95, y1=y0 + 0.95,
                fillcolor=CELL_FILL[t],
                line=dict(color=CELL_LINE[t], width=1.5),
            )
            lbl = CELL_LBL[t]
            if lbl:
                fig.add_annotation(
                    x=c + 0.5, y=y0 + 0.5, text=f"<b>{lbl}</b>",
                    showarrow=False,
                    font=dict(size=max(13, cell // 4), color=CELL_FG[t]),
                    xanchor="center", yanchor="middle",
                )
            cr = st.session_state.get("cell_rewards", {}).get((r, c), None)
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

    total_reward = sum(rewards)
    fig.update_layout(
        width=cols * cell, height=rows * cell,
        margin=dict(l=0, r=0, t=40, b=0),
        title=dict(text=f"Replay Episode — reward {total_reward:+.2f}", x=0.5, xanchor="center"),
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


# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🐕 GridWorld RL")

    # Grid size
    st.markdown("### Grid Size")
    c1, c2 = st.columns(2)
    n_rows = c1.number_input("Rows", 2, 14, st.session_state.rows,
                             key="inp_rows", label_visibility="visible")
    n_cols = c2.number_input("Cols", 2, 14, st.session_state.cols,
                             key="inp_cols", label_visibility="visible")
    bb1, bb2 = st.columns(2)
    if bb1.button("Resize", use_container_width=True):
        old = st.session_state.grid
        g = _blank_grid(n_rows, n_cols)
        for rr in range(min(n_rows, len(old))):
            for cc in range(min(n_cols, len(old[0]))):
                g[rr][cc] = old[rr][cc]
        st.session_state.grid = g
        st.session_state.rows = n_rows
        st.session_state.cols = n_cols
        st.rerun()
    if bb2.button("Clear", use_container_width=True):
        st.session_state.grid = _blank_grid(
            st.session_state.rows, st.session_state.cols)
        st.rerun()

    st.divider()

    st.markdown("### Layout")
    layout_file = st.file_uploader(
        "Upload Layout", type=["json"],
        help="Load a saved game layout with walls, goals, traps, rewards, and discount factor.",
    )
    if layout_file is not None:
        try:
            content = json.load(layout_file)
            rows = int(content["rows"])
            cols = int(content["cols"])
            layout_lines = content["layout"]
            if len(layout_lines) != rows:
                raise ValueError("Layout rows do not match saved rows.")
            if any(len(row) != cols for row in layout_lines):
                raise ValueError("Layout columns do not match saved cols.")
            st.session_state.grid = _layout_to_grid(layout_lines)
            st.session_state.rows = rows
            st.session_state.cols = cols
            st.session_state.step_rew = float(content.get("step_reward", st.session_state.get("step_rew", 0.0)))
            st.session_state.slip_prob = float(content.get("slip_prob", st.session_state.get("slip_prob", 0.3)))
            st.session_state.gamma = float(content.get("gamma", st.session_state.get("gamma", 0.95)))
            cell_rewards = {}
            for k, v in content.get("cell_rewards", {}).items():
                try:
                    r_s, c_s = k.split(",")
                    cell_rewards[(int(r_s), int(c_s))] = float(v)
                except Exception:
                    continue
            st.session_state.cell_rewards = cell_rewards
            st.success("Layout loaded successfully.")
            _safe_rerun()
        except Exception as exc:
            st.error(f"Unable to load layout: {exc}")

    prompt_text = st.text_area(
        "Generation prompt",
        value=st.session_state.get("layout_prompt", ""),
        help=("Describe layout preferences for the random generator. "
              "Examples: large positive terminal value, more negative rewards, fewer walls."),
        key="layout_prompt",
    )
    use_claude = st.checkbox(
        "Use Claude prompt to influence generation",
        value=st.session_state.get("use_claude_prompt", False),
        key="use_claude_prompt",
        help="If enabled, the prompt is sent to Claude to bias the generated layout.",
    )
    if use_claude and prompt_text.strip() and not _get_claude_api_key():
        st.warning("Claude API key is not configured. Set CLAUDE_API_KEY in the environment or Streamlit secrets.")

    complexity = st.slider(
        "Random layout complexity",
        1, 10, st.session_state.get("layout_complexity", 1),
        help="Choose how complex the randomly generated grid should be.",
        key="layout_complexity",
    )
    if st.button("Generate random layout", use_container_width=True):
        rows = st.session_state.rows
        cols = st.session_state.cols
        cell_rewards = {}
        grid = None

        # ── Claude full-grid path ─────────────────────────────────────
        if use_claude and _get_claude_api_key() and prompt_text.strip():
            with st.spinner("Asking Claude to design the grid…"):
                grid, err = _generate_grid_from_claude(prompt_text, rows, cols, complexity)
            if grid is not None:
                st.success("Claude designed the grid from your prompt.")
            else:
                st.warning(f"Claude grid generation failed ({err}). Falling back to random generator.")
                grid = None

        # ── Fallback: local hints + random generator ──────────────────
        if grid is None:
            hints = None
            if prompt_text.strip():
                hints = _infer_layout_directives_from_prompt(prompt_text, rows, cols)
                if hints:
                    st.info("Using local prompt parsing to bias random layout generation.")
            grid, cell_rewards = _generate_random_layout(complexity, rows, cols, hints=hints)
            if not (use_claude and _get_claude_api_key()):
                st.success("Generated a new random layout.")

        st.session_state.grid = grid
        st.session_state.cell_rewards = cell_rewards
        st.session_state.results = None
        st.session_state.train_run = None
        st.session_state.loaded_training_metadata = None
        st.session_state.play_trace = None
        _safe_rerun()

    training_file = st.file_uploader(
        "Load Training Result", type=["json"],
        help="Load a previously saved training result with metadata, configuration, board, and replay data.",
        key="load_training_file",
    )
    if training_file is not None:
        try:
            payload = json.load(training_file)
            _load_training_payload(payload)
            st.success("Training result loaded successfully.")
        except Exception as exc:
            st.error(f"Unable to load training result: {exc}")

    # Paint tool and rewards in one block
    st.markdown("### Paint Tool & Rewards")
    tool_idx = st.radio(
        "tool", options=list(range(6)),
        format_func=lambda i: f"  {CELL_LBL[i] or '·'}  {CELL_NAME[i]}",
        label_visibility="collapsed",
    )
    st.session_state.tool = tool_idx

    st.markdown("#### Step Reward")
    step_rew = st.number_input("Step",  value=st.session_state.get("step_rew", 0.0), step=0.01, format="%.2f", key="step_rew")
    # Use fixed defaults for Goal and Trap (can be set via per-cell rewards)
    goal_rew = 1.0
    trap_rew = -1.0

    st.markdown("#### Paint-time Cell Reward")
    c1, c2 = st.columns([2, 1])
    paint_reward_value = c1.number_input("Cell reward", value=st.session_state.get("paint_reward_value", 0.0), step=0.01, format="%.2f", key="paint_reward_value")
    paint_apply = c2.checkbox("Apply", value=False, key="paint_apply_reward", help="Apply reward when painting")
    paint_clear = st.checkbox("Clear reward when painting", value=False, key="paint_clear_reward")

    st.divider()
    st.markdown("### Layout")
    

    # Dynamics
    st.markdown("### Dynamics")
    slip_prob = st.slider("Slip probability", 0.0, 1.0, st.session_state.get("slip_prob", 0.3), 0.01,
                          help="Chance a slippery cell deflects the agent sideways",
                          key="slip_prob")
    gamma     = st.slider("Discount γ",       0.5, 1.0, st.session_state.get("gamma", 0.95), 0.01, key="gamma")

    layout_data = {
        "rows": st.session_state.rows,
        "cols": st.session_state.cols,
        "layout": _grid_to_layout(st.session_state.grid, st.session_state.rows, st.session_state.cols),
        "step_reward": float(st.session_state.get("step_rew", 0.0)),
        "goal_reward": 1.0,
        "trap_reward": -1.0,
        "slip_prob": float(st.session_state.get("slip_prob", 0.3)),
        "gamma": float(st.session_state.get("gamma", 0.95)),
        "cell_rewards": { f"{r},{c}": float(v) for (r,c), v in st.session_state.get("cell_rewards", {}).items() },
    }
    json_text = json.dumps(layout_data, indent=2)
    st.download_button(
        "Download Layout",
        json_text,
        file_name=f"gridworld_{layout_data['rows']}x{layout_data['cols']}.json",
        mime="application/json",
    )

    # Play simulation from the Start cell (animate moves)
    st.markdown("### Play From Start")
    play_max_steps = st.number_input("Play max steps", 1, 1000, 100, key="play_max_steps")
    play_delay = st.number_input("Delay per step (s)", 0.01, 2.0, 0.25, step=0.05, format="%.2f", key="play_delay")
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

    # Algorithm
    st.markdown("### Algorithm")
    algo = st.selectbox(
        "algo", ["Policy Iteration", "Q-Learning", "SARSA", "DQN"],
        label_visibility="collapsed",
        key="algo",
    )
    # Run mode: full training or step-through
    run_mode = st.selectbox(
        "Run mode", ["Full", "Step-through"], index=0, label_visibility="collapsed",
        key="run_mode",
    )
    if algo != "Policy Iteration":
        episodes = st.number_input("Episodes", 200, 20_000, 3_000, 200)
        alpha    = st.number_input("Learning rate α", 0.001, 1.0, 0.1, 0.005,
                                   format="%.3f")
        epsilon  = st.slider("Exploration ε", 0.0, 1.0, 1.0, 0.01,
                             help="Initial exploration probability for ε-greedy policies.",
                             key="epsilon")
        epsilon_decay = st.number_input(
            "Epsilon decay (multiplicative)", 0.900, 1.000, 0.998, 0.0005,
            format="%.4f",
            help="Per-episode multiplicative decay for ε (e.g. 0.998)",
            key="epsilon_decay",
        )
        exploring_starts = st.checkbox(
            "Exploring starts (randomize episode start)", value=True,
            help="If enabled, each episode starts in a random non-terminal cell.", key="exploring_starts",
        )
        max_steps = st.number_input("Max steps per episode", 1, 1000, 300, 10,
                                    help="Maximum number of steps before the episode ends.",
                                    key="max_steps")
        # DQN-specific controls
        if algo == "DQN":
            st.markdown("---")
            st.markdown("**DQN Parameters**")
            dqn_hidden_str = st.text_input(
                "Hidden layers (comma-separated)", value=st.session_state.get("dqn_hidden_str", "64,64"),
                help="Sizes for hidden linear layers, e.g. 64,64", key="dqn_hidden_str",
            )
            try:
                dqn_hidden = [int(s.strip()) for s in dqn_hidden_str.split(",") if s.strip()]
                if not dqn_hidden:
                    dqn_hidden = [64, 64]
            except Exception:
                dqn_hidden = [64, 64]

            dqn_lr = st.number_input("DQN learning rate", 1e-6, 1.0, st.session_state.get("dqn_lr", 1e-3), format="%.5f", key="dqn_lr")
            dqn_batch_size = st.number_input("DQN batch size", 8, 4096, st.session_state.get("dqn_batch_size", 64), step=1, key="dqn_batch_size")
            dqn_buffer_size = st.number_input("DQN buffer size", 100, 1_000_000, st.session_state.get("dqn_buffer_size", 10_000), step=100, key="dqn_buffer_size")
            dqn_target_update = st.number_input("DQN target update (steps)", 1, 100_000, st.session_state.get("dqn_target_update", 100), step=1, key="dqn_target_update")
            dqn_episodes = st.number_input("DQN episodes", 1, 50_000, episodes, step=100, key="dqn_episodes")
            dqn_epsilon = st.number_input("DQN epsilon (initial)", 0.0, 1.0, st.session_state.get("dqn_epsilon", 1.0), step=0.01, format="%.3f", key="dqn_epsilon")
            dqn_epsilon_decay = st.number_input("DQN epsilon decay", 0.900, 1.000, st.session_state.get("dqn_epsilon_decay", 0.997), step=0.0005, format="%.4f", key="dqn_epsilon_decay")
            dqn_epsilon_min = st.number_input("DQN epsilon min", 0.0, 1.0, st.session_state.get("dqn_epsilon_min", 0.01), step=0.01, format="%.3f", key="dqn_epsilon_min")
        else:
            # sensible defaults when not using DQN
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
        alpha    = 0.1
        epsilon  = st.session_state.get("epsilon", 1.0)
        epsilon_decay = st.session_state.get("epsilon_decay", 0.998)
        exploring_starts = st.session_state.get("exploring_starts", True)
        max_steps = st.session_state.get("max_steps", 300)

        # default DQN values for non-DQN algorithms so the config object is complete
        dqn_hidden = [64, 64]
        dqn_lr = 1e-3
        dqn_batch_size = 64
        dqn_buffer_size = 10_000
        dqn_target_update = 100
        dqn_episodes = 3_000
        dqn_epsilon = 1.0
        dqn_epsilon_decay = 0.997
        dqn_epsilon_min = 0.01

    record_episodes = st.checkbox(
        "Record episodes for replay",
        value=st.session_state.get("record_episodes", False),
        key="record_episodes",
        help="Store episode traces so you can replay episodes after training.",
    )
    record_strategy = st.selectbox(
        "Replay retention",
        ["Most recent", "Subsample evenly"],
        index=0,
        key="record_strategy",
        help="Most recent keeps the latest episodes; subsample evenly keeps a periodic sample across the full training run.",
    )
    record_limit = st.number_input(
        "Max recorded episodes",
        min_value=1, max_value=1000, value=100, step=1,
        help="Keep this many recorded episodes for replay.",
        key="record_limit",
    )

    st.divider()

    

    # Train button
    st.markdown('<div class="train-btn">', unsafe_allow_html=True)
    train_clicked = st.button("▶  Train Agent", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Step-through controls
    if run_mode == "Step-through":
        st.markdown("---")
        if st.button("Start Step-through", use_container_width=True):
            # Initialize a step-run controller in session state
            st.session_state.step_run = {
                "algo": algo,
                "cfg": None,  # filled when training starts
                "env": None,
                "state": None,
                "finished": False,
            }
            _safe_rerun()


# ── Main layout ───────────────────────────────────────────────────────
st.markdown("# 🐕  GridWorld RL Trainer")
st.markdown(
    "Design your environment, choose an algorithm, and watch the agent learn.",
    help="Click cells in the grid to paint them, then press **Train Agent**.",
)
st.divider()

left_col, right_col = st.columns([1, 1.5], gap="large")

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
    st.markdown('<p class="section-title">📊 Results</p>', unsafe_allow_html=True)

    res = st.session_state.results
    tr = st.session_state.get("train_run")
    if res is not None:
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
    if res is None:
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

            # Episode replay
            episode_traces = res.get("episode_traces")
            if episode_traces:
                st.markdown("<div class=\"section-title\">🎬 Episode Replay</div>", unsafe_allow_html=True)
                episode_index = st.selectbox(
                    "Episode to replay",
                    options=list(range(1, len(episode_traces) + 1)),
                    index=0,
                    format_func=lambda i: f"Episode {i}",
                    key="replay_episode",
                )
                trace = episode_traces[episode_index - 1]
                ep_steps = len(trace.get("actions", []))
                ep_reward = sum(trace.get("rewards", []))
                ep_off_policy = trace.get("off_policy_steps", 0)
                c1, c2, c3, c4 = st.columns([1, 1, 1, 1.4])
                c1.metric("Replay episode", f"{episode_index}")
                c2.metric("Steps", f"{ep_steps}")
                c3.metric("Off-policy steps", f"{ep_off_policy}")
                c4.metric("Reward", f"{ep_reward:+.2f}")
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
                    n_episodes=st.session_state.get("episodes", 3000),
                    dqn_n_episodes=st.session_state.get("episodes", 3000),
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
        else:
            pi_iters = res.get("pi_iters", "?")
            st.success(f"Policy Iteration converged in **{pi_iters}** sweeps.", icon="✅")
            st.info(
                "Policy Iteration state values are expected returns from each state. "
                "If the next action enters the goal, the adjacent state's value can be 1.0 "
                "because the goal reward is received immediately upon reaching it.",
                icon="ℹ️",
            )

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
                        device = torch.device("cpu")
                        q_net = QNetwork(env.n_states, env.n_actions, cfg.dqn_hidden).to(device)
                        target_net = QNetwork(env.n_states, env.n_actions, cfg.dqn_hidden).to(device)
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
                        })

                    q_net = state["q_net"]
                    target_net = state["target_net"]
                    optimizer = state["optimizer"]
                    buffer = state["buffer"]
                    eps = state["epsilon"]
                    total_steps = state["total_steps"]

                    def state_vec(s):
                        v = [0.0] * env.n_states
                        v[s] = 1.0
                        return v

                    # Run one episode
                    s = env.reset()
                    sv = state_vec(s)
                    total_rew = 0.0
                    steps = 0

                    for _ in range(cfg.max_steps):
                        state["visits"][s] += 1
                        if random.random() < eps:
                            action = random.randrange(env.n_actions)
                        else:
                            with torch.no_grad():
                                s_t = torch.tensor([sv], dtype=torch.float32)
                                action = int(q_net(s_t).argmax(dim=1).item())

                        ns, r, done = env.step(action)
                        # count arrival
                        state["visits"][ns] += 1
                        next_sv = state_vec(ns)
                        total_rew += (env.gamma ** steps) * r
                        steps += 1
                        total_steps += 1

                        buffer.push(sv, action, r, next_sv, 1.0 if done else 0.0)
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
                                target_q = r_t + env.gamma * max_next_q * (1.0 - d_t)

                            loss = nn.functional.mse_loss(current_q, target_q)
                            optimizer.zero_grad()
                            loss.backward()
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

                    # Extract Q-table for visualization
                    all_states = [[1.0 if i == s else 0.0 for i in range(env.n_states)]
                                  for s in range(env.n_states)]
                    with torch.no_grad():
                        Q_list = q_net(torch.tensor(all_states, dtype=torch.float32)).tolist()
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


# ── Training logic ────────────────────────────────────────────────────
if train_clicked:
    grid = st.session_state.grid
    rows = st.session_state.rows
    cols = st.session_state.cols

    flat = [grid[r][c] for r in range(rows) for c in range(cols)]
    if ST not in flat:
        st.sidebar.error("❌ Grid needs a Start cell (S).")
        st.stop()
    if GO not in flat:
        st.sidebar.error("❌ Grid needs at least one Goal cell (G).")
        st.stop()

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

    with st.spinner(f"Training with **{algo}** — please wait…"):
        env = GridWorld(grid_cfg)
        # If step-through mode requested, initialise controller and don't run full training
        if run_mode == "Step-through":
            sr = st.session_state.setdefault("step_run", {})
            sr["algo"] = algo
            sr["cfg"] = algo_cfg
            sr["env"] = env
            sr.setdefault("state", None)
            sr.setdefault("finished", False)
            _safe_rerun()

        if algo == "Policy Iteration":
            pol, vals, hist = policy_iteration(env, algo_cfg)
            st.session_state.results = dict(
                algo=algo, env=env, policy=pol, values=vals,
                rewards=None, lengths=None, pi_iters=len(hist),
            )
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
                    device = torch.device("cpu")
                    q_net = QNetwork(env.n_states, env.n_actions, cfg.dqn_hidden).to(device)
                    target_net = QNetwork(env.n_states, env.n_actions, cfg.dqn_hidden).to(device)
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
                    })
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
                        # Accumulate discounted return: use env.gamma^t * r_t
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
                        total_reward += (env.gamma ** steps) * r
                        steps += 1
                        off_steps += 1 if next_off else 0
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
                    )

                elif algo == "DQN":
                    q_net = state["q_net"]
                    target_net = state["target_net"]
                    optimizer = state["optimizer"]
                    buffer = state["buffer"]
                    eps = state["epsilon"]
                    total_steps = state.get("total_steps", 0)

                    def state_vec(s):
                        v = [0.0] * env.n_states
                        v[s] = 1.0
                        return v

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
                    }
                    sv = state_vec(s)
                    total_rew = 0.0
                    steps = 0
                    off_steps = 0

                    for _ in range(cfg.max_steps):
                        state["visits"][s] += 1
                        valid = valid_actions(env, s)
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

                        ns, r, done = env.step(action)
                        state["visits"][ns] += 1
                        episode_trace["actions"].append(action)
                        episode_trace["rewards"].append(r)
                        episode_trace["next_states"].append(ns)
                        episode_trace["states"].append(ns)
                        next_sv = state_vec(ns)
                        total_rew += (env.gamma ** steps) * r
                        steps += 1
                        total_steps += 1

                        buffer.push(sv, action, r, next_sv, 1.0 if done else 0.0)
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
                                target_q = r_t + env.gamma * max_next_q * (1.0 - d_t)

                            loss = nn.functional.mse_loss(current_q, target_q)
                            optimizer.zero_grad()
                            loss.backward()
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

                    # Extract Q-table
                    all_states = [[1.0 if i == s else 0.0 for i in range(env.n_states)] for s in range(env.n_states)]
                    with torch.no_grad():
                        Q_list = q_net(torch.tensor(all_states, dtype=torch.float32)).tolist()
                    Q_table = np.array(Q_list, dtype=np.float64)
                    policy = np.argmax(Q_table, axis=1)
                    st.session_state.results = dict(
                        algo=algo, env=env, policy=policy, values=Q_table.max(axis=1),
                        rewards=state["rewards"], lengths=state["lengths"], visits=state.get("visits"),
                        episode_times=state.get("episode_times"), off_policy_steps=state.get("off_policy_steps"), 
                        epsilon_values=state.get("epsilon_values"), episode_traces=state.get("episode_traces"),
                    )

                # persist state
                tr["state"] = state

                # Check for UI update interval
                now = time.time()
                if now - tr.get("last_update", 0.0) >= update_interval:
                    tr["last_update"] = now
                    st.session_state.train_run = tr
                    _safe_rerun()

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

    st.rerun()
