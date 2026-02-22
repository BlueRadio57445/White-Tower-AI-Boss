"""
FSS result visualizer.

Reads the JSONL log produced by tools/fss.py and shows two plots:

  Plot 1 – Score Progression
      X: selection step (0 = baseline, 1 = first group added, ...)
      Y: score
      Each point shows which group was added at that step.

  Plot 2 – Candidate Scores per Step (heatmap)
      X: FSS step
      Y: candidate group name
      Color: score at that trial
      The chosen group at each step is marked with a white border.
      Groups that had already been selected are shown in gray.

Usage:
    python tools/visualize_fss.py --log fss_results_missile.jsonl
    python tools/visualize_fss.py --log fss_results_missile.jsonl --save fss_plot.png
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_TOOLS_DIR)
sys.path.insert(0, _PROJECT_DIR)


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

def load_log(path: str) -> list:
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def parse_log(records: list) -> dict:
    """
    Extract structured data from JSONL records.

    Returns dict with:
        baseline_score      float
        steps               list of dicts (step, candidate scores, added group)
        final_selected      list of str
        final_score         float
    """
    baseline_score = None
    # step_data[step] = {"scores": {candidate: score}, "added": str}
    step_data = defaultdict(lambda: {"scores": {}, "added": None})
    final_selected = []
    final_score = None

    for r in records:
        ev = r.get("event")
        if ev == "baseline":
            baseline_score = r["score"]
        elif ev == "eval":
            step_data[r["step"]]["scores"][r["candidate"]] = r["score"]
        elif ev == "added":
            step_data[r["step"]]["added"] = r["group"]
        elif ev == "done":
            final_selected = r["selected_groups"]
            final_score = r["final_score"]

    return {
        "baseline_score": baseline_score,
        "step_data": dict(step_data),
        "final_selected": final_selected,
        "final_score": final_score,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def visualize(log_path: str, save_path: str = None, show: bool = True) -> None:
    records = load_log(log_path)
    if not records:
        print("Log file is empty.")
        return

    parsed = parse_log(records)
    baseline = parsed["baseline_score"]
    step_data = parsed["step_data"]
    final_selected = parsed["final_selected"]
    final_score = parsed["final_score"]

    if baseline is None:
        print("No baseline record found in log.")
        return

    n_steps = len(step_data)
    if n_steps == 0:
        print("No evaluation steps found in log.")
        return

    # Collect all candidate names in order of first appearance
    all_candidates = []
    seen = set()
    for s in sorted(step_data.keys()):
        for c in step_data[s]["scores"]:
            if c not in seen:
                all_candidates.append(c)
                seen.add(c)

    # Build score progression (x=step number, y=score after adding group)
    progression_x = [0]
    progression_y = [baseline]
    progression_labels = ["(baseline)"]
    current = baseline
    for s in sorted(step_data.keys()):
        added = step_data[s]["added"]
        if added and added in step_data[s]["scores"]:
            current = step_data[s]["scores"][added]
        progression_x.append(s)
        progression_y.append(current)
        progression_labels.append(added or "?")

    # --- Figure ---
    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(f"FSS Results  —  {os.path.basename(log_path)}", fontsize=13, fontweight='bold')

    gs = fig.add_gridspec(2, 1, hspace=0.45, top=0.91, bottom=0.08)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # ---- Plot 1: Score Progression ----
    ax1.plot(progression_x, progression_y, 'o-', color='royalblue',
             linewidth=2, markersize=8, zorder=3)
    ax1.axhline(baseline, color='gray', linestyle='--', linewidth=1, alpha=0.6, label='baseline')

    for x, y, label in zip(progression_x, progression_y, progression_labels):
        ax1.annotate(label, (x, y),
                     textcoords="offset points", xytext=(0, 10),
                     ha='center', fontsize=8, color='#222222')

    ax1.set_title("Score Progression (higher = better)")
    ax1.set_xlabel("Selection Step")
    ax1.set_ylabel("Score")
    ax1.set_xticks(progression_x)
    ax1.legend(fontsize=8)
    ax1.grid(True, linestyle=':', alpha=0.5)

    # ---- Plot 2: Heatmap of candidate scores per step ----
    steps_sorted = sorted(step_data.keys())
    n_cands = len(all_candidates)
    cand_idx = {c: i for i, c in enumerate(all_candidates)}

    # Build matrix: rows = candidates, cols = steps
    mat = np.full((n_cands, len(steps_sorted)), np.nan)
    already_selected = set()

    for col_i, s in enumerate(steps_sorted):
        for cand, score in step_data[s]["scores"].items():
            row_i = cand_idx[cand]
            mat[row_i, col_i] = score
        added = step_data[s]["added"]
        if added:
            already_selected.add(added)

    # Mask NaN cells (already-selected groups not re-evaluated)
    masked = np.ma.masked_invalid(mat)

    # Choose colormap: diverging around 0 or just sequential
    vmin = np.nanmin(mat)
    vmax = np.nanmax(mat)

    cmap = matplotlib.colormaps.get_cmap('RdYlGn').copy()
    cmap.set_bad(color='#cccccc')  # gray for already-selected (NaN)

    im = ax2.imshow(masked, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax,
                    extent=[-0.5, len(steps_sorted) - 0.5,
                            -0.5, n_cands - 0.5],
                    origin='lower')

    # Mark cells that beat the baseline with black borders,
    # merging adjacent rows in the same column into one large rectangle.
    for col_i, s in enumerate(steps_sorted):
        beating_rows = sorted(
            cand_idx[cand]
            for cand, score in step_data[s]["scores"].items()
            if score > baseline and cand in cand_idx
        )
        # Find contiguous runs and draw one rectangle per run
        if beating_rows:
            run_start = beating_rows[0]
            run_end = beating_rows[0]
            for row_i in beating_rows[1:]:
                if row_i == run_end + 1:
                    run_end = row_i
                else:
                    rect = matplotlib.patches.Rectangle(
                        (col_i - 0.5, run_start - 0.5), 1, run_end - run_start + 1,
                        linewidth=2.0, edgecolor='black', facecolor='none', zorder=4
                    )
                    ax2.add_patch(rect)
                    run_start = row_i
                    run_end = row_i
            rect = matplotlib.patches.Rectangle(
                (col_i - 0.5, run_start - 0.5), 1, run_end - run_start + 1,
                linewidth=2.0, edgecolor='black', facecolor='none', zorder=4
            )
            ax2.add_patch(rect)

    # Mark the chosen group at each step with a white border (on top)
    for col_i, s in enumerate(steps_sorted):
        added = step_data[s]["added"]
        if added and added in cand_idx:
            row_i = cand_idx[added]
            rect = matplotlib.patches.Rectangle(
                (col_i - 0.5, row_i - 0.5), 1, 1,
                linewidth=2.5, edgecolor='white', facecolor='none', zorder=5
            )
            ax2.add_patch(rect)

    # Annotate each cell with its score
    for row_i in range(n_cands):
        for col_i in range(len(steps_sorted)):
            val = mat[row_i, col_i]
            if not np.isnan(val):
                ax2.text(col_i, row_i, f"{val:.1f}",
                         ha='center', va='center', fontsize=7,
                         color='black', fontweight='bold')

    ax2.set_yticks(range(n_cands))
    ax2.set_yticklabels(all_candidates, fontsize=8)
    ax2.set_xticks(range(len(steps_sorted)))
    ax2.set_xticklabels([f"Step {s}" for s in steps_sorted], fontsize=8)
    ax2.set_title("Candidate Scores per Step  (white box = selected, black dashed = beats baseline, gray = already chosen)")
    plt.colorbar(im, ax=ax2, fraction=0.03, pad=0.02, label='score')

    # Summary text
    if final_selected:
        summary = f"Final: {final_selected}  |  dims: {len(final_selected)}+  |  score: {final_score}"
        fig.text(0.5, 0.01, summary, ha='center', fontsize=9,
                 color='#444444', style='italic')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {save_path}")
    if show:
        plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize FSS log")
    parser.add_argument("--log", required=True,
                        help="Path to FSS JSONL log (e.g. fss_results_missile.jsonl)")
    parser.add_argument("--save", default=None,
                        help="Save plot to file instead of showing")
    args = parser.parse_args()

    log_path = args.log
    if not os.path.isabs(log_path) and not os.path.exists(log_path):
        log_path = os.path.join(_PROJECT_DIR, log_path)

    if not os.path.exists(log_path):
        print(f"Error: Log file not found: {args.log}")
        sys.exit(1)

    visualize(log_path, save_path=args.save, show=args.save is None)
