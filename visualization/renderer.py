from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from environment.gridworld import GridWorld

# ── Visual constants ──────────────────────────────────────────────────

_CELL_COLORS = {
    GridWorld.EMPTY:    "#E8EAF0",
    GridWorld.WALL:     "#2E2E2E",
    GridWorld.SLIPPERY: "#7EC8E3",
    GridWorld.GOAL:     "#52C97A",
    GridWorld.TRAP:     "#E63946",
}

_FIG_BG   = "#F0F2F5"
_AXES_BG  = "#D8DBE3"

_CELL_LABELS = {
    GridWorld.EMPTY:    "",
    GridWorld.WALL:     "",
    GridWorld.SLIPPERY: "~",
    GridWorld.GOAL:     "GOAL",
    GridWorld.TRAP:     "TRAP",
}

# Arrow offsets for each action (in matplotlib coords: UP = +y)
_ARROW_DX = [0.0,   0.22,  0.0,  -0.22]  # UP RIGHT DOWN LEFT
_ARROW_DY = [0.22,  0.0,  -0.22,  0.0]


# ── Core grid renderer ────────────────────────────────────────────────

def render_grid(
    env: GridWorld,
    *,
    values: np.ndarray | None = None,
    policy: np.ndarray | None = None,
    agent_state: int | None = None,
    title: str = "GridWorld",
    ax: plt.Axes | None = None,
    show_start: bool = True,
) -> plt.Axes:
    """Draw the grid with optional value heatmap, policy arrows, and agent marker."""
    rows, cols = env.rows, env.cols

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(4, cols * 1.3), max(4, rows * 1.3)))
        fig.patch.set_facecolor(_FIG_BG)
    ax.set_facecolor(_AXES_BG)

    # Normalise values for heatmap (exclude walls and terminals)
    cmap = norm = None
    if values is not None:
        valid = [
            float(values[s])
            for s in range(env.n_states)
            if not env.is_wall(s) and not env.is_terminal(s)
        ]
        if valid:
            vmin, vmax = min(valid), max(valid)
            if abs(vmax - vmin) < 1e-10:
                vmin -= 1.0; vmax += 1.0
            norm = Normalize(vmin=vmin, vmax=vmax)
            cmap = plt.cm.RdYlGn

    for r in range(rows):
        for c in range(cols):
            s    = r * cols + c
            cell = env.grid[r, c]
            # matplotlib y: row 0 appears at TOP → flip
            y = rows - 1 - r

            # Background
            if cmap and norm and not env.is_wall(s) and not env.is_terminal(s):
                bg = cmap(norm(float(values[s])))
            else:
                bg = _CELL_COLORS[cell]

            rect = patches.Rectangle(
                (c, y), 1, 1,
                facecolor=bg, edgecolor="#707070", linewidth=1.0,
            )
            ax.add_patch(rect)

            # Cell label (centre-top)
            label = _CELL_LABELS.get(cell, "")
            if label:
                ax.text(c + 0.5, y + 0.72, label,
                        ha="center", va="center", fontsize=9,
                        color="white",
                        fontweight="bold")

            # Slippery indicator
            if cell == GridWorld.SLIPPERY:
                ax.text(c + 0.5, y + 0.72, "≈",
                        ha="center", va="center", fontsize=12, color="#003A5C",
                        fontweight="bold")

            # Value (centre-bottom) — white pill so it reads on any cell colour
            if values is not None and not env.is_wall(s):
                ax.text(c + 0.5, y + 0.22, f"{float(values[s]):.2f}",
                        ha="center", va="center", fontsize=8,
                        color="#111111", fontweight="bold",
                        bbox=dict(
                            boxstyle="round,pad=0.12",
                            facecolor="white", alpha=0.80,
                            edgecolor="none",
                        ))

            # Start marker
            if show_start and (r, c) == env.start_pos and cell == GridWorld.EMPTY:
                ax.text(c + 0.16, y + 0.84, "S",
                        ha="center", va="center", fontsize=8,
                        color="#333333", fontweight="bold", fontstyle="italic")

            # Policy arrow
            if policy is not None and not env.is_wall(s) and not env.is_terminal(s):
                a  = int(policy[s])
                cx, cy = c + 0.5, y + 0.5
                dx, dy = _ARROW_DX[a], _ARROW_DY[a]
                ax.annotate(
                    "", xy=(cx + dx, cy + dy), xytext=(cx - dx, cy - dy),
                    arrowprops=dict(
                        arrowstyle="->", color="#1a1aaa",
                        lw=1.6, mutation_scale=14,
                    ),
                )

    # Agent marker
    if agent_state is not None and not env.is_wall(agent_state):
        ar, ac = divmod(agent_state, cols)
        ay = rows - 1 - ar
        ax.text(
            ac + 0.5, ay + 0.5, "A",
            ha="center", va="center", fontsize=16, fontweight="bold", color="white",
            bbox=dict(
                boxstyle="circle,pad=0.18", facecolor="#1155CC",
                edgecolor="#0A3A9A", linewidth=2.0, alpha=0.95,
            ),
        )

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=12, pad=8)

    if cmap and norm:
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, fraction=0.04, pad=0.02, label="State value")

    return ax


# ── Training curves ───────────────────────────────────────────────────

def plot_training_curves(
    episode_rewards: list[float],
    episode_lengths: list[int],
    algo_name: str,
    window: int = 50,
) -> plt.Figure:
    """Plot smoothed reward and episode-length curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor(_FIG_BG)
    fig.suptitle(f"{algo_name} — Training Curves", fontsize=13)

    def smooth(data: list) -> np.ndarray:
        return np.convolve(data, np.ones(window) / window, mode="valid")

    for ax, data, ylabel, color in zip(
        axes,
        [episode_rewards, episode_lengths],
        ["Total Reward", "Episode Length"],
        ["steelblue", "salmon"],
    ):
        ax.plot(data, alpha=0.2, color=color)
        if len(data) >= window:
            ax.plot(range(window - 1, len(data)), smooth(data), color=color, lw=2)
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ── Combined summary figure ───────────────────────────────────────────

def render_summary(
    env: GridWorld,
    policy: np.ndarray,
    values: np.ndarray,
    algo_name: str,
    episode_rewards: list[float] | None = None,
    episode_lengths: list[int]  | None = None,
) -> plt.Figure:
    """Grid (+ optional training curves) in one figure."""
    has_curves = episode_rewards is not None

    if has_curves:
        fig = plt.figure(figsize=(16, 5))
        ax_grid = fig.add_subplot(1, 3, 1)
        ax_r    = fig.add_subplot(1, 3, 2)
        ax_l    = fig.add_subplot(1, 3, 3)
    else:
        fig, ax_grid = plt.subplots(figsize=(max(5, env.cols * 1.3), max(5, env.rows * 1.3)))
    fig.patch.set_facecolor(_FIG_BG)

    render_grid(env, values=values, policy=policy,
                title=f"{algo_name} — Policy & Values", ax=ax_grid)

    if has_curves:
        window = max(1, min(50, len(episode_rewards) // 10))

        def smooth(d: list) -> np.ndarray:
            return np.convolve(d, np.ones(window) / window, mode="valid")

        ax_r.plot(episode_rewards, alpha=0.2, color="steelblue")
        if len(episode_rewards) >= window:
            ax_r.plot(range(window - 1, len(episode_rewards)),
                      smooth(episode_rewards), color="steelblue", lw=2)
        ax_r.set_xlabel("Episode"); ax_r.set_ylabel("Reward")
        ax_r.set_title("Reward per Episode"); ax_r.grid(True, alpha=0.3)

        ax_l.plot(episode_lengths, alpha=0.2, color="salmon")
        if len(episode_lengths) >= window:
            ax_l.plot(range(window - 1, len(episode_lengths)),
                      smooth(episode_lengths), color="salmon", lw=2)
        ax_l.set_xlabel("Episode"); ax_l.set_ylabel("Steps")
        ax_l.set_title("Episode Length"); ax_l.grid(True, alpha=0.3)

    fig.suptitle(f"Algorithm: {algo_name}", fontsize=14, y=1.01)
    plt.tight_layout()
    return fig


# ── Episode animation ─────────────────────────────────────────────────

def animate_episode(
    env: GridWorld,
    policy: np.ndarray,
    algo_name: str,
    max_steps: int = 50,
) -> tuple[animation.FuncAnimation, plt.Figure]:
    """Roll out one episode following *policy* and animate it step by step."""
    state  = env.reset()
    states = [state]
    cumulative: list[float] = [0.0]

    for _ in range(max_steps):
        action              = int(policy[state])
        next_state, r, done = env.step(action)
        states.append(next_state)
        cumulative.append(cumulative[-1] + r)
        state = next_state
        if done:
            break

    fig, ax = plt.subplots(figsize=(max(4, env.cols * 1.3), max(4, env.rows * 1.3)))

    def update(frame: int) -> None:
        ax.clear()
        render_grid(
            env,
            policy=policy,
            agent_state=states[frame],
            title=(
                f"{algo_name}  |  Step {frame}"
                f"  |  Cumulative reward: {cumulative[frame]:.2f}"
            ),
            ax=ax,
        )

    ani = animation.FuncAnimation(
        fig, update, frames=len(states), interval=600, repeat=False
    )
    plt.tight_layout()
    return ani, fig
