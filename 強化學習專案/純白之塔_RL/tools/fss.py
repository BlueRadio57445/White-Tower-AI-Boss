"""
Forward Stepwise Selection (FSS) for feature groups.

Greedy algorithm: starting from a fixed set of groups, iteratively adds the
candidate group that most improves agent performance. Stops when no candidate
improves the current score.

Usage:
    python tools/fss.py --config tools/fss_missile.json

Config JSON format:
    {
        "fixed_groups":     ["bias"],
        "candidate_groups": ["monster_dist", "monster_dx_dy", ...],
        "curriculum":       "curricula/missile_training.json",
        "eval_epochs":      300,
        "steps_per_epoch":  50,
        "eval_window":      50,
        "tolerance_count":  1,
        "output":           "fss_results.json"
    }

Fields:
    fixed_groups     Groups always included (never removed). Usually ["bias"].
    candidate_groups Groups to evaluate. FSS picks from these greedily.
    curriculum       Path to curriculum JSON (relative to project root).
    eval_epochs      Training epochs per candidate evaluation.
    steps_per_epoch  Steps per epoch (default: 50).
    eval_window      How many final epochs to average for scoring (default: 50).
    tolerance_count  Steps allowed without beating the all-time best score.
                     In each such step the best candidate is still added.
                     Does not reset on improvement (default: 0).
    output           Path to save results JSON (default: fss_results.json).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Resolve project root so this script can be run from any directory.
_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_TOOLS_DIR)
sys.path.insert(0, _PROJECT_DIR)

from ai.feature_registry import SelectiveFeatureExtractor, list_all_groups
from training.curriculum import CurriculumManager
from training.trainer import Trainer, TrainingConfig


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def _evaluate(
    group_names: list,
    curriculum: CurriculumManager,
    cfg: dict,
) -> float:
    """
    Train a fresh agent with the given feature groups and return its score.

    Score = mean reward over the last `eval_window` epochs.
    """
    extractor = SelectiveFeatureExtractor(group_names, world_size=10.0)

    config = TrainingConfig(
        epochs=cfg["eval_epochs"],
        steps_per_epoch=cfg.get("steps_per_epoch", 50),
        use_pygame=False,
        export_weights=False,
        # Hyperparameters: inherit from config or use defaults
        gamma=cfg.get("gamma", 0.99),
        lmbda=cfg.get("lmbda", 0.95),
        epsilon=cfg.get("epsilon", 0.2),
        sigma_init=cfg.get("sigma_init", 0.6),
        sigma_min=cfg.get("sigma_min", 0.15),
        sigma_decay=cfg.get("sigma_decay", 0.9998),
        lr_actor_discrete=cfg.get("lr_actor_discrete", 0.003),
        lr_actor_continuous=cfg.get("lr_actor_continuous", 0.002),
        lr_critic=cfg.get("lr_critic", 0.007),
    )

    trainer = Trainer(config, curriculum=curriculum, feature_extractor=extractor)
    history = trainer.train(render=False)

    window = min(cfg.get("eval_window", 50), len(history))
    return float(np.mean(history[-window:]))


# ---------------------------------------------------------------------------
# JSONL logger
# ---------------------------------------------------------------------------

class _FSSLog:
    """Append-only JSONL writer for FSS progress."""

    def __init__(self, path: str):
        self._f = open(path, 'w', encoding='utf-8')
        print(f"FSS log: {path}")

    def write(self, event: str, **kwargs) -> None:
        record = {"event": event, "ts": round(time.time(), 2), **kwargs}
        self._f.write(json.dumps(record, ensure_ascii=False) + '\n')
        self._f.flush()

    def close(self) -> None:
        self._f.close()


# ---------------------------------------------------------------------------
# FSS algorithm
# ---------------------------------------------------------------------------

def run_fss(config_path: str) -> dict:
    """
    Run Forward Stepwise Selection.

    Returns dict with keys:
        selected_groups  Final ordered list of selected group names.
        steps            List of {step, added, score, n_features} dicts.
        final_score      Score after all additions.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Resolve curriculum path relative to project root
    curriculum_raw = cfg["curriculum"]
    if not os.path.isabs(curriculum_raw):
        curriculum_raw = os.path.join(_PROJECT_DIR, curriculum_raw)

    curriculum = CurriculumManager(
        curriculum_raw,
        levels_dir=os.path.join(_PROJECT_DIR, "levels"),
        masks_dir=os.path.join(_PROJECT_DIR, "masks"),
    )

    fixed = list(cfg.get("fixed_groups", ["bias"]))
    candidates = list(cfg["candidate_groups"])
    remaining = [c for c in candidates if c not in fixed]

    # Derive log path from output path (replace .json → .jsonl)
    output_path = cfg.get("output", "fss_results.json")
    if not os.path.isabs(output_path):
        output_path = os.path.join(_PROJECT_DIR, output_path)
    log_path = str(Path(output_path).with_suffix(".jsonl"))
    log = _FSSLog(log_path)

    print("\n" + "=" * 60)
    print("Forward Stepwise Selection")
    print("=" * 60)
    print(curriculum.describe())
    tolerance_count = cfg.get("tolerance_count", 0)

    print(f"\nFixed groups   : {fixed}")
    print(f"Candidates     : {remaining}")
    print(f"Eval epochs    : {cfg['eval_epochs']}")
    print(f"Eval window    : {cfg.get('eval_window', 50)}")
    print(f"Tolerance count: {tolerance_count}")
    print()

    log.write("start",
              fixed_groups=fixed,
              candidates=remaining,
              eval_epochs=cfg["eval_epochs"],
              eval_window=cfg.get("eval_window", 50),
              tolerance_count=tolerance_count,
              curriculum=cfg["curriculum"])

    steps = []

    # Initial baseline score
    print("Computing baseline score...")
    t0 = time.time()
    current_score = _evaluate(fixed, curriculum, cfg)
    elapsed = time.time() - t0
    print(f"Baseline  fixed={fixed}  score={current_score:.3f}  ({elapsed:.1f}s)")
    log.write("baseline", groups=fixed, score=round(current_score, 4),
              n_features=SelectiveFeatureExtractor(fixed).n_features,
              elapsed_s=round(elapsed, 1))

    best_ever_score = current_score
    tolerance_left = tolerance_count

    step = 0
    while remaining:
        step += 1
        print(f"\n--- Step {step} ---")
        print(f"Current groups : {fixed}  (score={current_score:.3f}, best={best_ever_score:.3f}, "
              f"tolerance={tolerance_left}/{tolerance_count})")

        step_best_candidate = None
        step_best_score = -float("inf")

        for candidate in remaining:
            trial = fixed + [candidate]
            t0 = time.time()
            score = _evaluate(trial, curriculum, cfg)
            elapsed = time.time() - t0
            marker = " *" if score > step_best_score else ""
            print(f"  + {candidate:<30} score={score:.3f}  ({elapsed:.1f}s){marker}")
            log.write("eval",
                      step=step,
                      candidate=candidate,
                      trial_groups=trial,
                      score=round(score, 4),
                      n_features=SelectiveFeatureExtractor(trial).n_features,
                      elapsed_s=round(elapsed, 1))
            if score > step_best_score:
                step_best_score = score
                step_best_candidate = candidate

        improved = step_best_score > best_ever_score
        if improved:
            best_ever_score = step_best_score
        elif tolerance_left > 0:
            tolerance_left -= 1
            used = tolerance_count - tolerance_left
            print(f"\nNo improvement over best ({best_ever_score:.3f}). "
                  f"Tolerance {used}/{tolerance_count}: adding '{step_best_candidate}' "
                  f"(score={step_best_score:.3f})")
            log.write("tolerance",
                      step=step,
                      candidate=step_best_candidate,
                      step_best_score=round(step_best_score, 4),
                      best_ever_score=round(best_ever_score, 4),
                      tolerance_used=used,
                      tolerance_count=tolerance_count)
        else:
            print(f"\nNo candidate improved best score ({best_ever_score:.3f}) and "
                  f"tolerance exhausted. Stopping.")
            log.write("no_improvement", step=step, best_ever_score=round(best_ever_score, 4))
            break

        fixed.append(step_best_candidate)
        remaining.remove(step_best_candidate)
        extractor_tmp = SelectiveFeatureExtractor(fixed)
        steps.append({
            "step": step,
            "added": step_best_candidate,
            "score": round(step_best_score, 4),
            "n_features": extractor_tmp.n_features,
            "selected_so_far": list(fixed),
            "tolerance_used": not improved,
        })
        current_score = step_best_score
        log.write("added",
                  step=step,
                  group=step_best_candidate,
                  score=round(current_score, 4),
                  n_features=extractor_tmp.n_features,
                  selected=list(fixed),
                  tolerance_used=not improved)
        print(f"\n→ Added: '{step_best_candidate}'  "
              f"new score={current_score:.3f}  "
              f"total dims={extractor_tmp.n_features}")

    # Summary
    final_extractor = SelectiveFeatureExtractor(fixed)
    print("\n" + "=" * 60)
    print("FSS Complete")
    print("=" * 60)
    print(f"Selected groups : {fixed}")
    print(f"Total dims      : {final_extractor.n_features}")
    print(f"Final score     : {current_score:.3f}")
    print()
    print(final_extractor.describe())

    result = {
        "selected_groups": fixed,
        "steps": steps,
        "final_score": round(current_score, 4),
        "total_dims": final_extractor.n_features,
    }

    log.write("done",
              selected_groups=fixed,
              final_score=round(current_score, 4),
              total_dims=final_extractor.n_features)
    log.close()

    # Save results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nResults  : {output_path}")
    print(f"Log      : {log_path}")
    print(f"Visualize: python tools/visualize_fss.py --log {log_path}")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Forward Stepwise Selection for feature groups"
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to FSS config JSON (e.g. tools/fss_missile.json)"
    )
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path) and not os.path.exists(config_path):
        config_path = os.path.join(_PROJECT_DIR, config_path)
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    run_fss(config_path)
