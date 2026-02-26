"""
Interactive weight visualizer for weights.json.

Views (switchable via left-side buttons):
  - Discrete Actor : heatmap  (n_actions × n_features)
  - Aim Actors     : heatmap  (n_aims × n_features)
  - Critic         : heatmap  (1 × n_features)

Interactions:
  - Click any row in the heatmap  → detail panel below updates
  - Absolute |w|                  → show absolute values (yellow colormap)
  - Normalize rows                → each row scaled to [-1, 1] for cross-row comparison
  - Group summary                 → detail panel shows per-group L2 norm instead of per-feature bars

Usage:
    cd 純白之塔_RL
    python tools/visualize_weights.py
    python tools/visualize_weights.py --weights path/to/weights.json
"""

import argparse
import json
import os
import sys
import textwrap

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Button, CheckButtons

# ── Path setup ─────────────────────────────────────────────────────────────
_TOOLS_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_TOOLS_DIR)
sys.path.insert(0, _PROJECT_DIR)

# ── CJK font fallback (for Chinese action labels on Windows) ────────────────
try:
    matplotlib.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei',
                                           'Noto Sans CJK SC', 'sans-serif']
except Exception:
    pass


# ---------------------------------------------------------------------------
# Color palette (Tableau-20 style, 20 distinct hues)
# ---------------------------------------------------------------------------
_PALETTE = [
    '#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f',
    '#edc948', '#b07aa1', '#ff9da7', '#9c755f', '#bab0ac',
    '#499894', '#86bcb6', '#d37295', '#a0cbe8', '#ffbe7d',
    '#8cd17d', '#b6992d', '#f1ce63', '#79706e', '#d4a6c8',
]

def _gc(i: int) -> str:
    """Get group color by index."""
    return _PALETTE[i % len(_PALETTE)]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(path: str) -> dict:
    """Load weights.json and return a structured dict."""
    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    w      = raw['weights']
    layout = raw.get('layout', {})

    actor_discrete = np.array(w['actor_discrete'])      # (n_actions, n_features)
    aim_actors     = np.array(w['aim_actors'])           # (n_aims, n_features)
    critic         = np.array(w['critic'])               # (n_features,)

    # Feature group layout
    feat_layout = layout.get('features', [])
    if not feat_layout:
        n = actor_discrete.shape[1]
        feat_layout = [{'name': f'dim_{i}', 'dims': 1, 'offset': i} for i in range(n)]

    # Discrete action labels  (one per row)
    action_labels = []
    for e in layout.get('discrete_actions', []):
        idx  = e.get('index', len(action_labels))
        name = e.get('name', f'action_{idx}')
        lbl  = e.get('label', '')
        action_labels.append(f"{idx}  {name}" + (f"\n{lbl}" if lbl else ''))
    while len(action_labels) < actor_discrete.shape[0]:
        action_labels.append(f"action_{len(action_labels)}")

    # Aim actor labels  (one per row)
    aim_labels = []
    for e in layout.get('aim_actors', []):
        idx   = e.get('index', len(aim_labels))
        name  = e.get('name', f'aim_{idx}')
        skill = e.get('skill_name', '')
        aim_labels.append(f"{idx}  {name}" + (f"\n({skill})" if skill else ''))
    while len(aim_labels) < aim_actors.shape[0]:
        aim_labels.append(f"aim_{len(aim_labels)}")

    return {
        'actor_discrete': actor_discrete,
        'aim_actors':     aim_actors,
        'critic':         critic,
        'feat_layout':    feat_layout,
        'action_labels':  action_labels,
        'aim_labels':     aim_labels,
    }


# ---------------------------------------------------------------------------
# Visualizer
# ---------------------------------------------------------------------------

class WeightVisualizer:
    _VIEWS = ['Discrete Actor', 'Aim Actors', 'Critic']

    def __init__(self, data: dict):
        self.data = data
        self._view      = 0      # 0=Discrete, 1=Aim, 2=Critic
        self._row       = 0      # selected heatmap row
        self._abs       = False  # show |w|
        self._norm_rows = False  # normalize each row to max-abs=1
        self._grp_sum   = False  # detail: per-group L2 norm

        self._build_figure()
        self._refresh()

    # -----------------------------------------------------------------------
    # Figure layout
    # -----------------------------------------------------------------------

    def _build_figure(self):
        self.fig = plt.figure(figsize=(18, 11), facecolor='#efefef')
        self.fig.suptitle(
            'Weight Visualizer  —  click a row to inspect below',
            fontsize=12, fontweight='bold', y=0.995
        )

        # Main axes
        self.ax_heat   = self.fig.add_axes([0.155, 0.39, 0.795, 0.55])
        self.ax_cbar   = self.fig.add_axes([0.953, 0.39, 0.012, 0.55])
        self.ax_detail = self.fig.add_axes([0.155, 0.065, 0.795, 0.28])

        # Left-strip widget dimensions
        lx, lw, bh = 0.012, 0.13, 0.040

        # ── View buttons ───────────────────────────────────────────────────
        ax_d = self.fig.add_axes([lx, 0.89, lw, bh])
        ax_a = self.fig.add_axes([lx, 0.84, lw, bh])
        ax_c = self.fig.add_axes([lx, 0.79, lw, bh])

        self._btn_d = Button(ax_d, 'Discrete Actor', color='#c8e6c9', hovercolor='#a5d6a7')
        self._btn_a = Button(ax_a, 'Aim Actors',     color='#bbdefb', hovercolor='#90caf9')
        self._btn_c = Button(ax_c, 'Critic',         color='#fce4ec', hovercolor='#f8bbd0')
        self._btn_d.label.set_fontsize(8.5)
        self._btn_a.label.set_fontsize(8.5)
        self._btn_c.label.set_fontsize(8.5)

        self._btn_d.on_clicked(lambda _: self._set_view(0))
        self._btn_a.on_clicked(lambda _: self._set_view(1))
        self._btn_c.on_clicked(lambda _: self._set_view(2))

        # Highlight active view button
        self._view_btns = [self._btn_d, self._btn_a, self._btn_c]
        self._btn_colors = ['#c8e6c9', '#bbdefb', '#fce4ec']
        self._btn_active_colors = ['#388e3c', '#1565c0', '#c62828']

        # ── Option checkboxes ──────────────────────────────────────────────
        ax_chk = self.fig.add_axes([lx, 0.60, lw, 0.14])
        ax_chk.set_facecolor('#e4e4e4')
        self._chk = CheckButtons(
            ax_chk,
            ['Absolute |w|', 'Normalize rows', 'Group summary'],
            [False, False, False]
        )
        for lbl in self._chk.labels:
            lbl.set_fontsize(8)
        self._chk.on_clicked(self._on_toggle)

        # ── Legend hint ────────────────────────────────────────────────────
        self.fig.text(
            lx + lw / 2, 0.565,
            'RdBu_r: red=neg\nblue=pos',
            ha='center', va='top', fontsize=7, color='#666',
            linespacing=1.5
        )

        # ── Click handler ──────────────────────────────────────────────────
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)

    # -----------------------------------------------------------------------
    # Data helpers
    # -----------------------------------------------------------------------

    def _matrix(self):
        """Return (processed_matrix, row_labels) for the current view."""
        if self._view == 0:
            mat    = self.data['actor_discrete'].copy()
            labels = list(self.data['action_labels'])
        elif self._view == 1:
            mat    = self.data['aim_actors'].copy()
            labels = list(self.data['aim_labels'])
        else:
            mat    = self.data['critic'].reshape(1, -1).copy()
            labels = ['Critic']

        if self._abs:
            mat = np.abs(mat)
        if self._norm_rows:
            mx = np.abs(mat).max(axis=1, keepdims=True)
            mx[mx == 0] = 1.0
            mat = mat / mx
        return mat, labels

    # -----------------------------------------------------------------------
    # Drawing
    # -----------------------------------------------------------------------

    def _refresh(self):
        self._update_btn_highlight()
        self._draw_heatmap()
        self._draw_detail()

    def _update_btn_highlight(self):
        """Bold-border the active view button."""
        for i, btn in enumerate(self._view_btns):
            if i == self._view:
                btn.ax.set_facecolor(self._btn_active_colors[i])
                btn.label.set_color('white')
                btn.label.set_fontweight('bold')
            else:
                btn.ax.set_facecolor(self._btn_colors[i])
                btn.label.set_color('black')
                btn.label.set_fontweight('normal')

    def _draw_heatmap(self):
        self.ax_heat.clear()
        self.ax_cbar.clear()

        mat, labels = self._matrix()
        n_rows, n_cols = mat.shape

        # Colormap
        if self._abs:
            cmap, vmin, vmax = 'YlOrRd', 0.0, max(float(np.abs(mat).max()), 1e-9)
        else:
            lim = max(float(np.abs(mat).max()), 1e-9)
            cmap, vmin, vmax = 'RdBu_r', -lim, lim

        im = self.ax_heat.imshow(
            mat, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax,
            interpolation='nearest'
        )
        cb = plt.colorbar(im, cax=self.ax_cbar)
        cb.ax.tick_params(labelsize=7)

        # ── Feature group decorations ─────────────────────────────────────
        feat_layout = self.data['feat_layout']
        xtick_pos, xtick_lbl = [], []
        for gi, g in enumerate(feat_layout):
            off, dim = g['offset'], g['dims']
            center   = off + dim / 2 - 0.5
            xtick_pos.append(center)
            xtick_lbl.append(f"{g['name']}\n({dim}D)")

            if off > 0:
                self.ax_heat.axvline(x=off - 0.5, color='white', lw=1.2, alpha=0.55)
            self.ax_heat.axvspan(
                off - 0.5, off + dim - 0.5,
                alpha=0.07, color=_gc(gi), zorder=0
            )

        self.ax_heat.set_xticks(xtick_pos)
        self.ax_heat.set_xticklabels(xtick_lbl, rotation=40, ha='right', fontsize=7)
        self.ax_heat.set_yticks(range(n_rows))
        self.ax_heat.set_yticklabels(labels, fontsize=9)
        self.ax_heat.tick_params(axis='x', pad=2)

        # Selected-row highlight
        if n_rows > 1:
            row = max(0, min(self._row, n_rows - 1))
            self.ax_heat.axhspan(row - 0.5, row + 0.5, color='gold', alpha=0.35, zorder=3)

        # Title
        opts = ('  |w|' if self._abs else '') + ('  row-norm' if self._norm_rows else '')
        self.ax_heat.set_title(
            f'{self._VIEWS[self._view]} Weights  '
            f'({n_rows} rows × {n_cols} features){opts}',
            fontsize=9.5, pad=5
        )

        self.fig.canvas.draw_idle()

    def _draw_detail(self):
        self.ax_detail.clear()

        mat, labels = self._matrix()
        n_rows = mat.shape[0]
        row    = max(0, min(self._row, n_rows - 1))
        w      = mat[row]

        feat_layout = self.data['feat_layout']

        if self._grp_sum:
            # ── Per-group L2-norm bars ─────────────────────────────────────
            grp_names  = [g['name'] for g in feat_layout]
            grp_norms  = [
                float(np.linalg.norm(w[g['offset']: g['offset'] + g['dims']]))
                for g in feat_layout
            ]
            x = list(range(len(grp_names)))
            self.ax_detail.bar(
                x, grp_norms,
                color=[_gc(i) for i in range(len(grp_names))],
                width=0.72, edgecolor='white', linewidth=0.5
            )
            self.ax_detail.set_xticks(x)
            self.ax_detail.set_xticklabels(grp_names, rotation=40, ha='right', fontsize=7.5)
            self.ax_detail.set_ylabel('L2 norm', fontsize=8)
            title_mode = 'Group L2-norm'

        else:
            # ── Per-feature bars ───────────────────────────────────────────
            n_feat    = len(w)
            dim_cols  = []
            for gi, g in enumerate(feat_layout):
                dim_cols.extend([_gc(gi)] * g['dims'])
            dim_cols = dim_cols[:n_feat]

            bars = self.ax_detail.bar(
                range(n_feat), w,
                color=dim_cols, width=0.85, edgecolor='none'
            )
            if not self._abs:
                for bar, val in zip(bars, w):
                    if val < 0:
                        bar.set_alpha(0.55)

            # Group separators + x-tick per group
            xtick_pos, xtick_lbl = [], []
            for gi, g in enumerate(feat_layout):
                off, dim = g['offset'], g['dims']
                if off > 0:
                    self.ax_detail.axvline(x=off - 0.5, color='#999', lw=0.8, alpha=0.5)
                xtick_pos.append(off + dim / 2 - 0.5)
                xtick_lbl.append(g['name'])

            self.ax_detail.set_xticks(xtick_pos)
            self.ax_detail.set_xticklabels(xtick_lbl, rotation=40, ha='right', fontsize=7)
            self.ax_detail.axhline(0, color='#777', lw=0.8)
            self.ax_detail.set_xlim(-0.5, n_feat - 0.5)
            self.ax_detail.set_ylabel('Weight', fontsize=8)
            title_mode = 'per-feature  (color = group,  faded = negative)'

        # ── Colour legend (group patches) ─────────────────────────────────
        patches = [
            mpatches.Patch(color=_gc(i), label=g['name'])
            for i, g in enumerate(feat_layout)
        ]
        # Only show legend when it won't be too crowded
        if len(patches) <= 12:
            self.ax_detail.legend(
                handles=patches, loc='upper right',
                fontsize=6.5, ncol=2, framealpha=0.7,
                title='Feature groups', title_fontsize=7
            )

        lbl_clean = labels[row].replace('\n', '  ')
        self.ax_detail.set_title(
            f'Detail — {self._VIEWS[self._view]}  ›  row {row}: {lbl_clean}   [{title_mode}]',
            fontsize=8.5
        )
        self.ax_detail.grid(True, axis='y', linestyle=':', alpha=0.4)

        self.fig.canvas.draw_idle()

    # -----------------------------------------------------------------------
    # Callbacks
    # -----------------------------------------------------------------------

    def _set_view(self, idx: int):
        self._view = idx
        self._row  = 0
        self._refresh()

    def _on_toggle(self, label: str):
        if label == 'Absolute |w|':
            self._abs = not self._abs
        elif label == 'Normalize rows':
            self._norm_rows = not self._norm_rows
        elif label == 'Group summary':
            self._grp_sum = not self._grp_sum
        self._refresh()

    def _on_click(self, event):
        if event.inaxes is not self.ax_heat:
            return
        if event.ydata is None:
            return
        mat, _ = self._matrix()
        row = int(round(float(event.ydata)))
        row = max(0, min(row, mat.shape[0] - 1))
        if row == self._row:
            return  # no change
        self._row = row
        self._draw_heatmap()
        self._draw_detail()

    def show(self):
        plt.tight_layout(rect=[0.15, 0.04, 0.97, 0.97])
        plt.show()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Interactive visualizer for weights.json',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples:
              python tools/visualize_weights.py
              python tools/visualize_weights.py --weights path/to/weights.json
        """)
    )
    parser.add_argument(
        '--weights', default='weights.json',
        help='Path to weights.json (default: weights.json in project root)'
    )
    args = parser.parse_args()

    weights_path = args.weights
    if not os.path.isabs(weights_path) and not os.path.exists(weights_path):
        weights_path = os.path.join(_PROJECT_DIR, weights_path)

    if not os.path.exists(weights_path):
        print(f"Error: weights file not found: {args.weights}")
        print("Run training first to generate weights.json, or specify with --weights")
        sys.exit(1)

    print(f"Loading weights from: {weights_path}")
    data = load_data(weights_path)

    n_act  = data['actor_discrete'].shape[0]
    n_aim  = data['aim_actors'].shape[0]
    n_feat = data['actor_discrete'].shape[1]
    n_grp  = len(data['feat_layout'])
    print(f"  Discrete actor : {n_act} actions  ×  {n_feat} features")
    print(f"  Aim actors     : {n_aim} actors   ×  {n_feat} features")
    print(f"  Feature groups : {n_grp} groups")
    print()
    print("Controls:")
    print("  Left buttons  — switch view (Discrete Actor / Aim Actors / Critic)")
    print("  Checkboxes    — Absolute |w| / Normalize rows / Group summary")
    print("  Click row     — inspect feature weights in detail panel")
    print()

    viz = WeightVisualizer(data)
    viz.show()
