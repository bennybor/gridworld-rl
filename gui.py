"""GridWorld RL Trainer — Interactive GUI
Run:  python gui.py   (from d:\\gridworld_rl)
"""
import threading
import queue
import warnings
import tkinter as tk
from tkinter import ttk, messagebox

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

warnings.filterwarnings("ignore", message="Failed to initialize NumPy")

from config import GridConfig, AlgorithmConfig
from environment.gridworld import GridWorld
from algorithms.policy_iteration import policy_iteration
from algorithms.q_learning import q_learning
from algorithms.sarsa import sarsa
from algorithms.dqn import dqn
from visualization.renderer import render_grid


# ── Cell type palette ─────────────────────────────────────────────────
#          name        char   bg          fg         label
CELLS = [
    ("Empty",    ".",  "#E8E8E8", "#333333", ""),
    ("Wall",     "#",  "#2C2C2C", "#FFFFFF", "■"),
    ("Slippery", "~",  "#A8D8EA", "#1a5276", "≈"),
    ("Goal",     "G",  "#52B788", "#FFFFFF", "G"),
    ("Trap",     "X",  "#E63946", "#FFFFFF", "X"),
    ("Start",    "S",  "#FFD166", "#333333", "S"),
]
E, WL, SL, GO, TR, ST = 0, 1, 2, 3, 4, 5


# ── Grid editor ───────────────────────────────────────────────────────

class GridEditor(tk.Frame):
    """Click or drag to paint cells with the selected type."""

    def __init__(self, master, rows: int = 5, cols: int = 5, **kw):
        super().__init__(master, bg="#CCCCCC", **kw)
        self.rows = rows
        self.cols = cols
        self.selected = tk.IntVar(value=E)
        self._data: list[list[int]] = []
        self._btns: list[list[tk.Button]] = []
        self._btn_rc: dict = {}          # button → (r, c)
        self._rebuild()

    def _rebuild(self):
        for w in self.winfo_children():
            w.destroy()
        self._data = [[E] * self.cols for _ in range(self.rows)]
        self._btns = []
        self._btn_rc = {}

        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                b = tk.Button(
                    self, width=4, height=1,
                    relief=tk.GROOVE, bd=1,
                    font=("Helvetica", 10, "bold"),
                    cursor="hand2",
                )
                b.grid(row=r, column=c, padx=1, pady=1)
                b.bind("<Button-1>",  lambda e, r=r, c=c: self._paint(r, c))
                b.bind("<B1-Motion>", lambda e: self._on_drag(e))
                self._btn_rc[b] = (r, c)
                row.append(b)
            self._btns.append(row)

        self._set(0, 0, ST)   # default start at top-left

    def _on_drag(self, event):
        w = event.widget.winfo_containing(event.x_root, event.y_root)
        if w in self._btn_rc:
            r, c = self._btn_rc[w]
            self._paint(r, c)

    def _paint(self, r, c):
        self._set(r, c, self.selected.get())

    def _set(self, r, c, t):
        if t == ST:                          # enforce single start
            for rr in range(self.rows):
                for cc in range(self.cols):
                    if self._data[rr][cc] == ST:
                        self._data[rr][cc] = E
                        nm, ch, bg, fg, lbl = CELLS[E]
                        self._btns[rr][cc].config(bg=bg, fg=fg, text=lbl)
        self._data[r][c] = t
        nm, ch, bg, fg, lbl = CELLS[t]
        self._btns[r][c].config(bg=bg, fg=fg, text=lbl)

    def resize(self, rows: int, cols: int):
        old = [row[:] for row in self._data]
        self.rows, self.cols = rows, cols
        self._rebuild()
        self._set(0, 0, E)  # clear default ST; restored from old data or fallback below
        for r in range(min(rows, len(old))):
            for c in range(min(cols, len(old[0]))):
                if old[r][c] != E:
                    self._set(r, c, old[r][c])
        if not any(self._data[r][c] == ST for r in range(rows) for c in range(cols)):
            self._set(0, 0, ST)

    def clear(self):
        self.resize(self.rows, self.cols)

    def get_layout(self) -> list[str]:
        return [
            "".join(CELLS[self._data[r][c]][1] for c in range(self.cols))
            for r in range(self.rows)
        ]

    def validate(self) -> str | None:
        flat = [self._data[r][c] for r in range(self.rows) for c in range(self.cols)]
        if ST not in flat:
            return "Grid needs a Start cell (S)."
        if GO not in flat:
            return "Grid needs at least one Goal cell (G)."
        return None


# ── Main application ──────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("GridWorld RL Trainer")
        self.configure(bg="#F0F0F0")
        self._queue: queue.Queue = queue.Queue()
        self._build()
        self._poll()

    # ── Layout ────────────────────────────────────────────────────────

    def _build(self):
        # ── top row: editor | settings ─────────────────────────────
        top = tk.Frame(self, bg="#F0F0F0")
        top.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(8, 4))

        self._build_editor(top)
        self._build_settings(top)

        # ── bottom: matplotlib results ─────────────────────────────
        self._build_results()

        # ── status bar ─────────────────────────────────────────────
        self._status = tk.StringVar(value="Design your grid, configure settings, then press Train.")
        tk.Label(self, textvariable=self._status, anchor=tk.W,
                 font=("Helvetica", 9), fg="#555555", bg="#F0F0F0"
                 ).pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(0, 4))

    def _build_editor(self, parent):
        frame = tk.LabelFrame(parent, text="Grid Editor",
                              font=("Helvetica", 10, "bold"),
                              padx=6, pady=6, bg="#F0F0F0")
        frame.pack(side=tk.LEFT, anchor=tk.N, padx=(0, 8))

        # Resize / clear controls
        ctrl = tk.Frame(frame, bg="#F0F0F0")
        ctrl.pack(fill=tk.X, pady=(0, 6))

        tk.Label(ctrl, text="Rows:", bg="#F0F0F0").pack(side=tk.LEFT)
        self._rows_var = tk.IntVar(value=5)
        tk.Spinbox(ctrl, from_=2, to=14, width=3,
                   textvariable=self._rows_var).pack(side=tk.LEFT, padx=2)
        tk.Label(ctrl, text="Cols:", bg="#F0F0F0").pack(side=tk.LEFT, padx=(6, 0))
        self._cols_var = tk.IntVar(value=5)
        tk.Spinbox(ctrl, from_=2, to=14, width=3,
                   textvariable=self._cols_var).pack(side=tk.LEFT, padx=2)
        tk.Button(ctrl, text="Resize", command=self._do_resize,
                  relief=tk.GROOVE).pack(side=tk.LEFT, padx=(8, 2))
        tk.Button(ctrl, text="Clear", command=lambda: self._editor.clear(),
                  relief=tk.GROOVE).pack(side=tk.LEFT)

        self._editor = GridEditor(frame, rows=5, cols=5)
        self._editor.pack()

    def _build_settings(self, parent):
        outer = tk.LabelFrame(parent, text="Settings",
                              font=("Helvetica", 10, "bold"),
                              padx=10, pady=8, bg="#F0F0F0")
        outer.pack(side=tk.LEFT, anchor=tk.N, fill=tk.Y)

        row = 0

        # ── Cell palette ───────────────────────────────────────────
        tk.Label(outer, text="Paint Tool", font=("Helvetica", 9, "bold"),
                 bg="#F0F0F0").grid(row=row, column=0, columnspan=4,
                                    sticky=tk.W, pady=(0, 3))
        row += 1

        for i, (name, ch, bg, fg, lbl) in enumerate(CELLS):
            display = f"  {lbl or '·'}  {name}"
            rb = tk.Radiobutton(
                outer, text=display,
                variable=self._editor.selected, value=i,
                bg=bg, fg=fg, selectcolor=bg,
                activebackground=bg,
                font=("Helvetica", 9),
                relief=tk.GROOVE, padx=4, pady=3,
                indicatoron=False,
            )
            rb.grid(row=row + i // 3, column=i % 3, padx=2, pady=1, sticky=tk.EW)
        row += 2

        ttk.Separator(outer, orient=tk.HORIZONTAL).grid(
            row=row, column=0, columnspan=4, sticky=tk.EW, pady=8)
        row += 1

        # ── Rewards ────────────────────────────────────────────────
        tk.Label(outer, text="Rewards", font=("Helvetica", 9, "bold"),
                 bg="#F0F0F0").grid(row=row, column=0, columnspan=4, sticky=tk.W)
        row += 1

        self._step_rew = self._entry_row(outer, "Step reward:", "-0.04", row); row += 1
        self._goal_rew = self._entry_row(outer, "Goal reward:", "1.0",   row); row += 1
        self._trap_rew = self._entry_row(outer, "Trap reward:", "-1.0",  row); row += 1

        ttk.Separator(outer, orient=tk.HORIZONTAL).grid(
            row=row, column=0, columnspan=4, sticky=tk.EW, pady=8)
        row += 1

        # ── Dynamics ───────────────────────────────────────────────
        tk.Label(outer, text="Dynamics", font=("Helvetica", 9, "bold"),
                 bg="#F0F0F0").grid(row=row, column=0, columnspan=4, sticky=tk.W)
        row += 1

        self._slip_var  = self._scale_row(outer, "Slip prob:", 0.0, 1.0,  0.3,  row); row += 1
        self._gamma_var = self._scale_row(outer, "Gamma:",     0.5, 1.0,  0.95, row); row += 1

        ttk.Separator(outer, orient=tk.HORIZONTAL).grid(
            row=row, column=0, columnspan=4, sticky=tk.EW, pady=8)
        row += 1

        # ── Algorithm ──────────────────────────────────────────────
        tk.Label(outer, text="Algorithm", font=("Helvetica", 9, "bold"),
                 bg="#F0F0F0").grid(row=row, column=0, columnspan=4, sticky=tk.W)
        row += 1

        self._algo_var = tk.StringVar(value="Policy Iteration")
        algo_cb = ttk.Combobox(
            outer, textvariable=self._algo_var, width=22,
            values=["Policy Iteration", "Q-Learning", "SARSA", "DQN"],
            state="readonly",
        )
        algo_cb.grid(row=row, column=0, columnspan=4, sticky=tk.W, pady=2)
        algo_cb.bind("<<ComboboxSelected>>", lambda _: self._refresh_algo_ui())
        row += 1

        # TD-only fields (episodes, learning rate)
        self._td_frame = tk.Frame(outer, bg="#F0F0F0")
        self._td_frame.grid(row=row, column=0, columnspan=4, sticky=tk.W)
        row += 1

        tk.Label(self._td_frame, text="Episodes:", bg="#F0F0F0").pack(side=tk.LEFT)
        self._episodes_var = tk.StringVar(value="3000")
        tk.Entry(self._td_frame, textvariable=self._episodes_var,
                 width=6).pack(side=tk.LEFT, padx=2)
        tk.Label(self._td_frame, text="  LR (α):", bg="#F0F0F0").pack(side=tk.LEFT)
        self._alpha_var = tk.StringVar(value="0.1")
        tk.Entry(self._td_frame, textvariable=self._alpha_var,
                 width=5).pack(side=tk.LEFT, padx=2)

        self._refresh_algo_ui()

        # ── Train button ────────────────────────────────────────────
        self._train_btn = tk.Button(
            outer, text="▶  Train Agent",
            font=("Helvetica", 12, "bold"),
            bg="#2D6A4F", fg="white",
            activebackground="#1B4332", activeforeground="white",
            padx=12, pady=8, cursor="hand2", relief=tk.FLAT,
            command=self._train,
        )
        self._train_btn.grid(row=row, column=0, columnspan=4,
                             sticky=tk.EW, pady=(14, 0))

    def _entry_row(self, parent, label: str, default: str, row: int) -> tk.StringVar:
        tk.Label(parent, text=label, bg="#F0F0F0").grid(
            row=row, column=0, sticky=tk.W, pady=1)
        var = tk.StringVar(value=default)
        tk.Entry(parent, textvariable=var, width=9).grid(
            row=row, column=1, sticky=tk.W, padx=(4, 0))
        return var

    def _scale_row(self, parent, label: str, from_: float, to: float,
                   default: float, row: int) -> tk.DoubleVar:
        tk.Label(parent, text=label, bg="#F0F0F0").grid(
            row=row, column=0, sticky=tk.W, pady=1)
        var = tk.DoubleVar(value=default)
        tk.Scale(parent, variable=var, from_=from_, to=to,
                 resolution=0.01, orient=tk.HORIZONTAL,
                 length=160, showvalue=True, bg="#F0F0F0",
                 ).grid(row=row, column=1, columnspan=3, sticky=tk.W, padx=(4, 0))
        return var

    def _build_results(self):
        frame = tk.LabelFrame(self, text="Results",
                              font=("Helvetica", 10, "bold"),
                              padx=4, pady=4, bg="#F0F0F0")
        frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(0, 4))

        self._fig = Figure(figsize=(13, 3.8), facecolor="#F5F5F5")
        self._canvas = FigureCanvasTkAgg(self._fig, master=frame)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._placeholder()

    def _placeholder(self):
        self._fig.clear()
        ax = self._fig.add_subplot(1, 1, 1)
        ax.text(0.5, 0.5, "Train the agent to see the policy and learning curves here.",
                ha="center", va="center", fontsize=13, color="#AAAAAA",
                transform=ax.transAxes)
        ax.axis("off")
        self._canvas.draw()

    # ── UI helpers ─────────────────────────────────────────────────────

    def _refresh_algo_ui(self):
        is_dp = self._algo_var.get() == "Policy Iteration"
        if is_dp:
            self._td_frame.grid_remove()
        else:
            self._td_frame.grid()

    def _do_resize(self):
        try:
            r, c = int(self._rows_var.get()), int(self._cols_var.get())
            if not (2 <= r <= 14 and 2 <= c <= 14):
                raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Rows and cols must be integers 2–14.")
            return
        self._editor.resize(r, c)

    # ── Training ───────────────────────────────────────────────────────

    def _train(self):
        err = self._editor.validate()
        if err:
            messagebox.showerror("Invalid Grid", err)
            return

        try:
            step_rew = float(self._step_rew.get())
            goal_rew = float(self._goal_rew.get())
            trap_rew = float(self._trap_rew.get())
            slip     = float(self._slip_var.get())
            gamma    = float(self._gamma_var.get())
            algo     = self._algo_var.get()
            episodes = int(self._episodes_var.get()) if algo != "Policy Iteration" else 0
            alpha    = float(self._alpha_var.get())  if algo != "Policy Iteration" else 0.1
        except ValueError as e:
            messagebox.showerror("Invalid Settings", f"Check numeric fields:\n{e}")
            return

        grid_cfg = GridConfig(
            rows=self._editor.rows,
            cols=self._editor.cols,
            layout=self._editor.get_layout(),
            step_reward=step_rew,
            goal_reward=goal_rew,
            trap_reward=trap_rew,
            slippery_slip_prob=slip,
            gamma=gamma,
        )
        algo_cfg = AlgorithmConfig(
            alpha=alpha,
            n_episodes=episodes,
            dqn_n_episodes=episodes,
            max_steps=300,
            epsilon=1.0,
            epsilon_decay=0.998,
            epsilon_min=0.01,
        )

        self._train_btn.config(state=tk.DISABLED, text="⏳  Training…")
        self._status.set(f"Training with {algo}…  please wait.")
        self.update_idletasks()

        def worker():
            try:
                env = GridWorld(grid_cfg)
                if algo == "Policy Iteration":
                    pol, vals, hist = policy_iteration(env, algo_cfg)
                    self._queue.put(("ok", algo, env, pol, vals, None, None))
                elif algo == "Q-Learning":
                    pol, Q, rews, lens = q_learning(env, algo_cfg)
                    self._queue.put(("ok", algo, env, pol, Q.max(axis=1), rews, lens))
                elif algo == "SARSA":
                    pol, Q, rews, lens = sarsa(env, algo_cfg)
                    self._queue.put(("ok", algo, env, pol, Q.max(axis=1), rews, lens))
                elif algo == "DQN":
                    pol, Q, rews, lens = dqn(env, algo_cfg)
                    self._queue.put(("ok", algo, env, pol, Q.max(axis=1), rews, lens))
            except Exception as ex:
                import traceback
                self._queue.put(("err", str(ex), traceback.format_exc()))

        threading.Thread(target=worker, daemon=True).start()

    def _poll(self):
        try:
            msg = self._queue.get_nowait()
            if msg[0] == "ok":
                _, algo, env, pol, vals, rews, lens = msg
                self._show(algo, env, pol, vals, rews, lens)
                self._status.set(f"Done — {algo} finished training.")
            else:
                _, err, tb = msg
                self._status.set(f"Error: {err}")
                messagebox.showerror("Training Error", tb)
            self._train_btn.config(state=tk.NORMAL, text="▶  Train Agent")
        except queue.Empty:
            pass
        self.after(120, self._poll)

    def _show(self, algo, env, policy, values, rewards, lengths):
        self._fig.clear()
        has_curves = rewards is not None

        if has_curves:
            ax_g = self._fig.add_subplot(1, 3, 1)
            ax_r = self._fig.add_subplot(1, 3, 2)
            ax_l = self._fig.add_subplot(1, 3, 3)
        else:
            ax_g = self._fig.add_subplot(1, 1, 1)

        render_grid(env, values=values, policy=policy,
                    title=f"{algo} — Policy & Values", ax=ax_g)

        if has_curves:
            n  = len(rewards)
            w  = max(1, min(50, n // 10))
            sm = lambda d: np.convolve(d, np.ones(w) / w, mode="valid")

            ax_r.plot(rewards, alpha=0.2, color="steelblue")
            if n >= w:
                ax_r.plot(range(w - 1, n), sm(rewards), color="steelblue", lw=2)
            ax_r.set_xlabel("Episode")
            ax_r.set_ylabel("Total Reward")
            ax_r.set_title("Reward per Episode")
            ax_r.grid(True, alpha=0.3)

            ax_l.plot(lengths, alpha=0.2, color="salmon")
            if n >= w:
                ax_l.plot(range(w - 1, n), sm(lengths), color="salmon", lw=2)
            ax_l.set_xlabel("Episode")
            ax_l.set_ylabel("Steps")
            ax_l.set_title("Episode Length")
            ax_l.grid(True, alpha=0.3)

        self._fig.suptitle(f"Algorithm: {algo}", fontsize=12)
        self._fig.tight_layout()
        self._canvas.draw()


if __name__ == "__main__":
    app = App()
    app.mainloop()
