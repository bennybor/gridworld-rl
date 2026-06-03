# GridWorld RL — Claude Code Project Guide

## What this project is

A reinforcement learning playground built as a **Streamlit web app**.  
A user draws a grid (walls, goals, traps, slippery cells), tunes hyperparameters in the sidebar, trains one of four RL algorithms, and inspects the learned policy interactively.

**Primary entry point:** `streamlit_app.py`  
`main.py` is a standalone matplotlib demo — it is not the web app.

---

## How to run the app

```bash
.venv\Scripts\streamlit run streamlit_app.py
```

The app opens at http://localhost:8501 in the browser.

To stop it: `Ctrl+C` in the terminal.

---

## Project structure

```
gridworld_rl/
├── streamlit_app.py        # ← main web UI (Streamlit + Plotly)
├── main.py                 # standalone matplotlib demo (not the web app)
├── config.py               # GridConfig, AlgorithmConfig dataclasses
├── environment/
│   └── gridworld.py        # GridWorld env (gym-like: reset/step/get_transitions)
├── algorithms/
│   ├── policy_iteration.py # DP, model-based
│   ├── q_learning.py       # off-policy TD
│   ├── sarsa.py            # on-policy TD
│   └── dqn.py              # deep Q-network (PyTorch, CPU)
├── visualization/
│   └── renderer.py         # matplotlib figures (used only by main.py)
├── requirements.txt
└── .venv/                  # virtual environment
```

---

## Architecture

### Environment (`environment/gridworld.py`)
- `GridWorld(GridConfig)` — builds a transition table at construction time
- Cell types: `EMPTY=0 WALL=1 SLIPPERY=2 GOAL=3 TRAP=4`
- `step(action)` → `(next_state, reward, done)`
- `get_transitions(state, action)` → `[(prob, next_state, reward, done)]` for model-based algos
- Slippery cells slip sideways with `slippery_slip_prob`; normal cells use `default_slip_prob`

### Algorithms
All return `(policy, Q_or_V, rewards_list, lengths_list)` except Policy Iteration which returns `(policy, V, history)`.

| File | Algorithm | Model-based? |
|------|-----------|-------------|
| `policy_iteration.py` | Policy Iteration | Yes |
| `q_learning.py` | Q-Learning (ε-greedy) | No |
| `sarsa.py` | SARSA (ε-greedy) | No |
| `dqn.py` | DQN with replay buffer + target net | No (PyTorch) |

### Streamlit app (`streamlit_app.py`)
- **Sidebar**: grid size, paint tool, rewards, algorithm picker, hyperparameters
- **Main area**: interactive Plotly grid editor → train → policy/value heatmap + training curves
- All Plotly figures are built by `make_*_fig()` helper functions in the same file
- Session state keys: `grid`, `rows`, `cols`, `tool`, `results`, `cell_rewards`, `slip_prob`, `gamma`, `algo`, etc.
- Claude API integration: `_call_claude()` uses `CLAUDE_API_KEY` env var or Streamlit secrets to optionally generate grid layouts via prompt

### Config (`config.py`)
- `GridConfig` — grid dimensions, layout string, rewards, slip probs, gamma
- `AlgorithmConfig` — all hyperparameters for all four algorithms

---

## Key CSS notes (in `streamlit_app.py`)

The sidebar has a dark background (`#0f172a → #1e293b`).  
`[data-testid="stSidebar"] * { color: #e2e8f0 }` sets all sidebar text white.  
`input` and `textarea` elements are explicitly overridden to `color: #0f172a` on white background so typed values are readable.

---

## Cell character map

```
'.' = Empty    '#' = Wall    '~' = Slippery
'G' = Goal     'X' = Trap    'S' = Start
```

Start (`S`) is stored as `EMPTY` in the grid array; `start_pos` is tracked separately.

---

## Dependencies

Python 3.11+, all packages in `.venv`:

```
streamlit==1.58.0
plotly==6.7.0
torch==2.12.0+cpu
numpy==2.4.6
matplotlib==3.10.9
```

Install: `.venv\Scripts\pip install -r requirements.txt`

---

## Common tasks

**Add a new algorithm:**
1. Create `algorithms/my_algo.py` returning `(policy, Q, rewards, lengths)`
2. Import and call it in `streamlit_app.py` in the training dispatch block
3. Add its name to the algorithm selectbox options

**Change sidebar appearance:**
- All sidebar CSS lives in the `st.markdown("""<style>...""")` block near the top of `streamlit_app.py`

**Add a new Plotly grid view:**
- Add a `make_xyz_fig(env, data)` function following the pattern of `make_policy_fig` or `make_visit_fig`
- Call it from the results display section of the main area

**Modify cell types:**
- `CELLS` list in `streamlit_app.py` defines name, char, fill color, stroke color, label, and text color
- `GridWorld._CHAR_MAP` in `environment/gridworld.py` maps chars to int cell types
