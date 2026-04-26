"""Generate final demo plots for ForgeArena hackathon submission.

Produces 5 plots in plots_final/:
  1. before_after_eval.png   — 2×3 baseline vs trained comparison
  2. training_dynamics.png   — 2×2 loss/entropy/length/reward-std
  3. corruption_radar.png    — Spider chart: detection by corruption type
  4. episode_waterfall.png   — Sorted per-episode reward bars
  5. trained_eval.png        — 6-panel dashboard on trained results

Usage:
    python scripts/plot_final.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH  = ROOT / "results.json"
TRAINED_PATH   = ROOT / "results_phase3.json"
P1_LOG_PATH    = ROOT / "outputs" / "overseer-grpo" / "phase1_log_history.json"
P3_LOG_PATH    = ROOT / "outputs" / "overseer-grpo-phase2" / "phase3_log_history.json"
OUT_DIR        = ROOT / "plots_final"

# ── style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0d0e1a", "axes.facecolor": "#12132a",
    "axes.edgecolor": "#2a2d50",   "axes.labelcolor": "#c0c4e0",
    "xtick.color": "#9aa3c2",      "ytick.color": "#9aa3c2",
    "text.color": "#e0e0ff",       "grid.color": "#1e2040",
    "grid.linestyle": "--",        "grid.alpha": 0.6,
    "font.family": "monospace",    "font.size": 10,
    "legend.facecolor": "#12132a", "legend.edgecolor": "#2a2d50",
})

ACCENT = "#5b6bff"
GREEN  = "#4ade80"
RED    = "#f87171"
YELLOW = "#fbbf24"
PURPLE = "#a78bfa"
TEAL   = "#2dd4bf"

CORRUPTION_COLORS = {
    "TEMPORAL_SHIFT":        TEAL,
    "FACTUAL_OMISSION":      YELLOW,
    "AUTHORITY_FABRICATION":  RED,
    "BIAS_INJECTION":        PURPLE,
    "INSTRUCTION_OVERRIDE":  ACCENT,
}
DOMAIN_COLORS = {
    "customer_support":       ACCENT,
    "legal_summarisation":    TEAL,
    "code_review":            GREEN,
    "product_recommendation": YELLOW,
    "mixed":                  RED,
}

REWARD_KEYS = ["rewards/arena_reward/mean", "reward"]


# ── helpers ────────────────────────────────────────────────────────────────────
def load_records(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    return [r for r in data["records"]
            if r.get("error") in (None, "") and r.get("reward") is not None]


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text())["summary"]


def smooth(xs, ys, w=8):
    if len(ys) < w:
        return list(xs), list(ys)
    k = np.ones(w) / w
    s = np.convolve(ys, k, mode="valid")
    h = w // 2
    return list(xs[h:h+len(s)]), list(s)


def extract_series(log: list[dict], key: str) -> tuple[list, list]:
    """Pull (steps, values) for a given key from log history."""
    steps, vals = [], []
    for e in log:
        s = e.get("step")
        v = e.get(key)
        if s and v is not None:
            steps.append(s)
            vals.append(v)
    return steps, vals


def extract_reward_series(log: list[dict]) -> tuple[list, list]:
    steps, vals = [], []
    for e in log:
        s = e.get("step")
        r = next((e[k] for k in REWARD_KEYS if k in e), None)
        if s and r is not None:
            steps.append(s)
            vals.append(r)
    return steps, vals


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 1: Before/After Evaluation Dashboard  (2×3)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_before_after(baseline: list[dict], trained: list[dict], out: Path):
    fig, axes = plt.subplots(2, 3, figsize=(19, 11))
    fig.suptitle("Forge + Arena — Baseline  vs  GRPO-Trained Overseer",
                 fontsize=15, y=0.98, color="#e0e0ff", fontweight="bold")
    plt.subplots_adjust(hspace=0.50, wspace=0.38)

    # ── 1a: Component score comparison ─────────────────────────────────────
    ax = axes[0, 0]
    comps = ["detection_score", "explanation_score", "correction_score",
             "calibration_score", "reward"]
    labels = ["Detection\n(×0.40)", "Explanation\n(×0.30)", "Correction\n(×0.20)",
              "Calibration\n(×0.10)", "Composite\nReward"]
    base_means = [np.mean([r[c] for r in baseline]) for c in comps]
    train_means = [np.mean([r[c] for r in trained]) for c in comps]
    x = np.arange(len(comps))
    w = 0.35
    b1 = ax.bar(x - w/2, base_means, w, label=f"Baseline (n={len(baseline)})",
                color=RED, alpha=0.75, edgecolor="#0d0e1a")
    b2 = ax.bar(x + w/2, train_means, w, label=f"Trained (n={len(trained)})",
                color=GREEN, alpha=0.85, edgecolor="#0d0e1a")
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.008, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=7.5, color="#e0e0ff")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Score")
    ax.set_title("Component Score Comparison", pad=8)
    ax.legend(fontsize=8, framealpha=0.3); ax.grid(True, axis="y")

    # ── 1b: Detection rate by corruption type ─────────────────────────────
    ax = axes[0, 1]
    all_types = sorted(set(
        r["corruption_type"] for r in baseline + trained
        if r["corruption_present"] and r.get("corruption_type")
    ))
    def det_rates(recs):
        by_t = defaultdict(list)
        for r in recs:
            if r["corruption_present"] and r.get("corruption_type"):
                by_t[r["corruption_type"]].append(r["detection_score"])
        return {t: np.mean(by_t[t]) if by_t[t] else 0 for t in all_types}

    base_det = det_rates(baseline)
    train_det = det_rates(trained)
    x = np.arange(len(all_types))
    b1 = ax.bar(x - w/2, [base_det[t] for t in all_types], w,
                label="Baseline", color=RED, alpha=0.7, edgecolor="#0d0e1a")
    b2 = ax.bar(x + w/2, [train_det[t] for t in all_types], w,
                label="Trained", color=GREEN, alpha=0.85, edgecolor="#0d0e1a")
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.015, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=7, color="#e0e0ff")
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("_", "\n") for t in all_types], fontsize=7)
    ax.set_ylim(0, 1.15); ax.set_ylabel("Detection Rate")
    ax.set_title("Detection by Corruption Type", pad=8)
    ax.axhline(0.5, color="#606880", lw=1, ls=":", alpha=0.5)
    ax.legend(fontsize=8, framealpha=0.3); ax.grid(True, axis="y")

    # ── 1c: Confusion matrices side-by-side ────────────────────────────────
    ax = axes[0, 2]
    def confusion(recs):
        tp = fp = tn = fn = 0
        for r in recs:
            det = r["detection_score"] > 0.5
            cor = bool(r["corruption_present"])
            if det and cor: tp += 1
            elif det and not cor: fp += 1
            elif not det and cor: fn += 1
            else: tn += 1
        return tp, fp, tn, fn

    tp_b, fp_b, tn_b, fn_b = confusion(baseline)
    tp_t, fp_t, tn_t, fn_t = confusion(trained)

    # Simple text-based comparison
    ax.axis("off")
    ax.set_title("Detection Confusion Matrix", pad=8)
    headers = "           Baseline    Trained"
    rows = [
        f"  TP       {tp_b:>4d}        {tp_t:>4d}",
        f"  FP       {fp_b:>4d}        {fp_t:>4d}",
        f"  TN       {tn_b:>4d}        {tn_t:>4d}",
        f"  FN       {fn_b:>4d}        {fn_t:>4d}",
        "",
        f"  Prec     {tp_b/(tp_b+fp_b):.2f}" + f"        {tp_t/(tp_t+fp_t):.2f}" if (tp_b+fp_b) and (tp_t+fp_t) else "",
        f"  Recall   {tp_b/(tp_b+fn_b):.2f}" + f"        {tp_t/(tp_t+fn_t):.2f}" if (tp_b+fn_b) and (tp_t+fn_t) else "",
    ]
    prec_b = tp_b/(tp_b+fp_b) if (tp_b+fp_b) else 0
    rec_b = tp_b/(tp_b+fn_b) if (tp_b+fn_b) else 0
    f1_b = 2*prec_b*rec_b/(prec_b+rec_b) if (prec_b+rec_b) else 0
    prec_t = tp_t/(tp_t+fp_t) if (tp_t+fp_t) else 0
    rec_t = tp_t/(tp_t+fn_t) if (tp_t+fn_t) else 0
    f1_t = 2*prec_t*rec_t/(prec_t+rec_t) if (prec_t+rec_t) else 0

    cell_text = [
        ["TP", str(tp_b), str(tp_t)],
        ["FP", str(fp_b), str(fp_t)],
        ["TN", str(tn_b), str(tn_t)],
        ["FN", str(fn_b), str(fn_t)],
        ["", "", ""],
        ["Precision", f"{prec_b:.2f}", f"{prec_t:.2f}"],
        ["Recall", f"{rec_b:.2f}", f"{rec_t:.2f}"],
        ["F1", f"{f1_b:.2f}", f"{f1_t:.2f}"],
        ["Accuracy", f"{(tp_b+tn_b)/(tp_b+tn_b+fp_b+fn_b):.2f}",
                     f"{(tp_t+tn_t)/(tp_t+tn_t+fp_t+fn_t):.2f}"],
    ]
    table = ax.table(cellText=cell_text,
                     colLabels=["Metric", "Baseline", "Trained"],
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#2a2d50")
        cell.set_text_props(color="#e0e0ff")
        if row == 0:
            cell.set_facecolor("#1e2050")
            cell.set_text_props(fontweight="bold", color="#e0e0ff")
        elif row == 5:  # separator
            cell.set_facecolor("#0d0e1a")
            cell.set_height(0.02)
        else:
            cell.set_facecolor("#12132a")
        # Highlight improvements
        if col == 2 and row > 0 and row not in (2, 5):  # Trained column
            cell.set_facecolor("#0f2a1a")

    # ── 1d: Reward distribution shift ──────────────────────────────────────
    ax = axes[1, 0]
    bins = np.linspace(0, 1, 21)
    ax.hist([r["reward"] for r in baseline], bins=bins, alpha=0.6,
            color=RED, edgecolor="#0d0e1a", label=f"Baseline (μ={np.mean([r['reward'] for r in baseline]):.3f})")
    ax.hist([r["reward"] for r in trained], bins=bins, alpha=0.65,
            color=GREEN, edgecolor="#0d0e1a", label=f"Trained (μ={np.mean([r['reward'] for r in trained]):.3f})")
    ax.set_title("Reward Distribution Shift", pad=8)
    ax.set_xlabel("Composite Reward"); ax.set_ylabel("Episodes")
    ax.legend(fontsize=8, framealpha=0.3); ax.grid(True, axis="y")

    # ── 1e: Per-domain reward comparison ───────────────────────────────────
    ax = axes[1, 1]
    def by_domain(recs):
        d = defaultdict(list)
        for r in recs:
            d[r["domain"]].append(r["reward"])
        return d
    bd_base = by_domain(baseline)
    bd_train = by_domain(trained)
    all_doms = sorted(set(list(bd_base.keys()) + list(bd_train.keys())))
    x = np.arange(len(all_doms))
    b1 = ax.bar(x - w/2, [np.mean(bd_base.get(d, [0])) for d in all_doms], w,
                label="Baseline", color=RED, alpha=0.7, edgecolor="#0d0e1a")
    b2 = ax.bar(x + w/2, [np.mean(bd_train.get(d, [0])) for d in all_doms], w,
                label="Trained", color=GREEN, alpha=0.85, edgecolor="#0d0e1a")
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.008, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=7, color="#e0e0ff")
    ax.set_xticks(x)
    ax.set_xticklabels([d.replace("_", "\n") for d in all_doms], fontsize=8)
    ax.set_ylim(0, max(0.7, max(np.mean(bd_train.get(d, [0])) for d in all_doms) * 1.25))
    ax.set_title("Mean Reward by Domain", pad=8)
    ax.set_ylabel("Mean Reward")
    ax.legend(fontsize=8, framealpha=0.3); ax.grid(True, axis="y")

    # ── 1f: Clean vs Corrupted breakdown ───────────────────────────────────
    ax = axes[1, 2]
    comps_short = ["detection_score", "explanation_score", "correction_score", "reward"]
    short_labels = ["Detect", "Explain", "Correct", "Reward"]
    clean_t  = [r for r in trained if not r["corruption_present"]]
    dirty_t  = [r for r in trained if r["corruption_present"]]
    clean_b  = [r for r in baseline if not r["corruption_present"]]
    dirty_b  = [r for r in baseline if r["corruption_present"]]

    x = np.arange(len(comps_short))
    bw = 0.2
    ax.bar(x - 1.5*bw, [np.mean([r[c] for r in clean_b]) for c in comps_short], bw,
           label=f"Base Clean (n={len(clean_b)})", color=TEAL, alpha=0.6, edgecolor="#0d0e1a")
    ax.bar(x - 0.5*bw, [np.mean([r[c] for r in dirty_b]) for c in comps_short], bw,
           label=f"Base Corrupt (n={len(dirty_b)})", color=RED, alpha=0.5, edgecolor="#0d0e1a")
    ax.bar(x + 0.5*bw, [np.mean([r[c] for r in clean_t]) for c in comps_short], bw,
           label=f"Train Clean (n={len(clean_t)})", color=GREEN, alpha=0.8, edgecolor="#0d0e1a")
    ax.bar(x + 1.5*bw, [np.mean([r[c] for r in dirty_t]) for c in comps_short], bw,
           label=f"Train Corrupt (n={len(dirty_t)})", color=YELLOW, alpha=0.8, edgecolor="#0d0e1a")
    ax.set_xticks(x); ax.set_xticklabels(short_labels)
    ax.set_ylim(0, 1.1); ax.set_title("Clean vs Corrupted: Baseline & Trained", pad=8)
    ax.set_ylabel("Mean Score")
    ax.legend(fontsize=7, framealpha=0.3, ncol=2); ax.grid(True, axis="y")

    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="#0d0e1a")
    plt.close(fig)
    print(f"  Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 2: Training Dynamics Panel  (2×2)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_training_dynamics(p1_log: list[dict], p3_log: list[dict], out: Path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Training Dynamics — Phase 1 + Phase 3 GRPO",
                 fontsize=14, y=0.98, color="#e0e0ff", fontweight="bold")
    plt.subplots_adjust(hspace=0.40, wspace=0.30)

    p1_final_step = max((e.get("step", 0) for e in p1_log), default=0)

    def offset_p3(steps):
        return [s + p1_final_step for s in steps]

    # ── 2a: Loss curve ────────────────────────────────────────────────────
    ax = axes[0, 0]
    s1, v1 = extract_series(p1_log, "loss")
    s3, v3 = extract_series(p3_log, "loss")
    if s1:
        ax.plot(s1, v1, color=ACCENT, alpha=0.3, lw=1)
        sx, sy = smooth(np.array(s1), v1, w=6)
        ax.plot(sx, sy, color=ACCENT, lw=2, label="Phase 1")
    if s3:
        s3o = offset_p3(s3)
        ax.plot(s3o, v3, color=GREEN, alpha=0.3, lw=1)
        sx, sy = smooth(np.array(s3o), v3, w=6)
        ax.plot(sx, sy, color=GREEN, lw=2, label="Phase 3")
    ax.axvline(p1_final_step, color=YELLOW, lw=1.5, ls="--", alpha=0.7, label="Forge activated")
    ax.set_title("GRPO Loss", pad=8); ax.set_xlabel("Step"); ax.set_ylabel("Loss")
    ax.legend(fontsize=8, framealpha=0.3); ax.grid(True)

    # ── 2b: Entropy evolution ─────────────────────────────────────────────
    ax = axes[0, 1]
    s1, v1 = extract_series(p1_log, "entropy")
    s3, v3 = extract_series(p3_log, "entropy")
    if s1:
        ax.plot(s1, v1, color=ACCENT, alpha=0.3, lw=1)
        sx, sy = smooth(np.array(s1), v1, w=6)
        ax.plot(sx, sy, color=ACCENT, lw=2, label="Phase 1")
    if s3:
        s3o = offset_p3(s3)
        ax.plot(s3o, v3, color=GREEN, alpha=0.3, lw=1)
        sx, sy = smooth(np.array(s3o), v3, w=6)
        ax.plot(sx, sy, color=GREEN, lw=2, label="Phase 3")
    ax.axvline(p1_final_step, color=YELLOW, lw=1.5, ls="--", alpha=0.7, label="Forge activated")
    ax.set_title("Policy Entropy", pad=8); ax.set_xlabel("Step"); ax.set_ylabel("Entropy")
    ax.legend(fontsize=8, framealpha=0.3); ax.grid(True)

    # ── 2c: Completion length trend ───────────────────────────────────────
    ax = axes[1, 0]
    s1, v1 = extract_series(p1_log, "completions/mean_length")
    s3, v3 = extract_series(p3_log, "completions/mean_length")
    if s1:
        ax.plot(s1, v1, color=ACCENT, alpha=0.3, lw=1)
        sx, sy = smooth(np.array(s1), v1, w=6)
        ax.plot(sx, sy, color=ACCENT, lw=2, label="Phase 1")
    if s3:
        s3o = offset_p3(s3)
        ax.plot(s3o, v3, color=GREEN, alpha=0.3, lw=1)
        sx, sy = smooth(np.array(s3o), v3, w=6)
        ax.plot(sx, sy, color=GREEN, lw=2, label="Phase 3")
    ax.axvline(p1_final_step, color=YELLOW, lw=1.5, ls="--", alpha=0.7, label="Forge activated")
    ax.set_title("Mean Completion Length (tokens)", pad=8)
    ax.set_xlabel("Step"); ax.set_ylabel("Tokens")
    ax.legend(fontsize=8, framealpha=0.3); ax.grid(True)

    # ── 2d: Reward std (exploration signal) ───────────────────────────────
    ax = axes[1, 1]
    s1, v1 = extract_series(p1_log, "reward_std")
    s3, v3 = extract_series(p3_log, "reward_std")
    if s1:
        ax.fill_between(s1, 0, v1, color=ACCENT, alpha=0.15)
        ax.plot(s1, v1, color=ACCENT, lw=1.5, label="Phase 1")
    if s3:
        s3o = offset_p3(s3)
        ax.fill_between(s3o, 0, v3, color=GREEN, alpha=0.15)
        ax.plot(s3o, v3, color=GREEN, lw=1.5, label="Phase 3")
    ax.axvline(p1_final_step, color=YELLOW, lw=1.5, ls="--", alpha=0.7, label="Forge activated")
    ax.set_title("Reward Std (Exploration Signal)", pad=8)
    ax.set_xlabel("Step"); ax.set_ylabel("σ(reward)")
    ax.legend(fontsize=8, framealpha=0.3); ax.grid(True)

    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="#0d0e1a")
    plt.close(fig)
    print(f"  Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 3: Corruption-Type Radar/Spider Chart
# ═══════════════════════════════════════════════════════════════════════════════
def plot_corruption_radar(baseline: list[dict], trained: list[dict], out: Path):
    all_types = sorted(CORRUPTION_COLORS.keys())
    pretty = [t.replace("_", " ").title() for t in all_types]

    def rates(recs):
        by_t = defaultdict(list)
        for r in recs:
            if r["corruption_present"] and r.get("corruption_type"):
                by_t[r["corruption_type"]].append(r["detection_score"])
        return [np.mean(by_t[t]) if by_t[t] else 0.0 for t in all_types]

    base_r = rates(baseline)
    train_r = rates(trained)

    N = len(all_types)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    base_r += base_r[:1]
    train_r += train_r[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("#0d0e1a")
    ax.set_facecolor("#12132a")

    ax.plot(angles, base_r, "o-", color=RED, lw=2, alpha=0.7, label="Baseline", markersize=7)
    ax.fill(angles, base_r, color=RED, alpha=0.1)
    ax.plot(angles, train_r, "o-", color=GREEN, lw=2.5, alpha=0.9, label="GRPO-Trained", markersize=8)
    ax.fill(angles, train_r, color=GREEN, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(pretty, fontsize=10, color="#e0e0ff")
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=8, color="#9aa3c2")
    ax.yaxis.grid(True, color="#1e2040", linestyle="--", alpha=0.6)
    ax.xaxis.grid(True, color="#2a2d50", linestyle="-", alpha=0.4)
    ax.spines["polar"].set_color("#2a2d50")

    ax.set_title("Corruption Detection Rate by Type\nBaseline vs GRPO-Trained",
                 pad=25, fontsize=13, color="#e0e0ff", fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=10, framealpha=0.3)

    # Add delta annotations
    for i, t in enumerate(all_types):
        delta = train_r[i] - base_r[i]
        sign = "+" if delta >= 0 else ""
        color = GREEN if delta > 0 else (RED if delta < 0 else "#9aa3c2")
        ax.annotate(f"{sign}{delta:.2f}", xy=(angles[i], max(base_r[i], train_r[i])),
                    xytext=(8, 8), textcoords="offset points",
                    fontsize=8, color=color, fontweight="bold")

    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="#0d0e1a")
    plt.close(fig)
    print(f"  Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 4: Per-Episode Reward Waterfall (sorted bar chart)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_episode_waterfall(trained: list[dict], out: Path):
    # Sort by reward
    sorted_recs = sorted(trained, key=lambda r: r["reward"])

    rewards = [r["reward"] for r in sorted_recs]
    domains = [r["domain"] for r in sorted_recs]
    detected_correct = [
        (r["detection_score"] > 0.5) == bool(r["corruption_present"])
        for r in sorted_recs
    ]

    fig, ax = plt.subplots(figsize=(16, 5))
    x = np.arange(len(rewards))

    colors = [DOMAIN_COLORS.get(d, ACCENT) for d in domains]
    edge_colors = [GREEN if c else RED for c in detected_correct]

    bars = ax.bar(x, rewards, color=colors, edgecolor=edge_colors,
                  linewidth=1.2, alpha=0.85)

    # Mean line
    mean_r = np.mean(rewards)
    ax.axhline(mean_r, color=YELLOW, lw=1.5, ls="--", alpha=0.8,
               label=f"Mean reward = {mean_r:.3f}")

    ax.set_title("Per-Episode Reward — Trained Overseer (sorted)",
                 fontsize=13, pad=10, fontweight="bold")
    ax.set_xlabel("Episode (sorted by reward)")
    ax.set_ylabel("Composite Reward")
    ax.set_xlim(-0.5, len(rewards) - 0.5)
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y")

    # Legend for domains + detection correctness
    dom_patches = [mpatches.Patch(color=DOMAIN_COLORS.get(d, ACCENT), label=d.replace("_", " ").title())
                   for d in sorted(set(domains))]
    edge_patches = [
        mpatches.Patch(edgecolor=GREEN, facecolor="none", linewidth=2, label="Detection ✓"),
        mpatches.Patch(edgecolor=RED, facecolor="none", linewidth=2, label="Detection ✗"),
    ]
    ax.legend(handles=dom_patches + edge_patches + [
        plt.Line2D([0], [0], color=YELLOW, lw=1.5, ls="--", label=f"Mean={mean_r:.3f}")
    ], fontsize=7.5, framealpha=0.3, ncol=4, loc="upper left")

    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="#0d0e1a")
    plt.close(fig)
    print(f"  Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 5: Trained Eval Dashboard (reuse plot_results.py logic on trained data)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_trained_dashboard(records: list[dict], out: Path):
    """Reuse the 6-panel analysis from plot_results.py on trained results."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Forge + Arena — GRPO-Trained Overseer Evaluation",
                 fontsize=14, y=1.01, color="#e0e0ff", fontweight="bold")
    plt.subplots_adjust(hspace=0.45, wspace=0.35)

    # 5a: Reward distribution
    ax = axes[0, 0]
    rewards = [r["reward"] for r in records]
    bins = np.linspace(0, 1, 21)
    ax.hist(rewards, bins=bins, color=GREEN, edgecolor="#0d0e1a", lw=0.5, alpha=0.85)
    ax.axvline(np.mean(rewards), color=YELLOW, lw=1.5, ls="--",
               label=f"mean={np.mean(rewards):.3f}")
    ax.set_title(f"Reward Distribution (n={len(records)})", pad=8)
    ax.set_xlabel("Composite Reward"); ax.set_ylabel("Episodes")
    ax.legend(framealpha=0.3); ax.grid(True, axis="y")

    # 5b: Component means
    ax = axes[0, 1]
    comps = ["detection_score", "explanation_score", "correction_score",
             "calibration_score", "reward"]
    labels = ["Detection\n(×0.40)", "Explanation\n(×0.30)", "Correction\n(×0.20)",
              "Calibration\n(×0.10)", "Composite"]
    means = [np.mean([r[c] for r in records]) for c in comps]
    colors_c = [ACCENT, TEAL, GREEN, PURPLE, YELLOW]
    bars = ax.bar(labels, means, color=colors_c, edgecolor="#0d0e1a", lw=0.5, alpha=0.85)
    for bar, v in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.01, f"{v:.3f}",
                ha="center", va="bottom", fontsize=9, color="#e0e0ff")
    ax.set_ylim(0, 1.05); ax.set_title("Mean Score by Component", pad=8)
    ax.set_ylabel("Score [0–1]"); ax.grid(True, axis="y")

    # 5c: Detection by corruption type
    ax = axes[0, 2]
    corrupted = [r for r in records if r["corruption_present"]]
    by_type = defaultdict(list)
    for r in corrupted:
        by_type[r["corruption_type"]].append(r["detection_score"])
    types = sorted(by_type)
    rates = [np.mean(by_type[t]) for t in types]
    counts = [len(by_type[t]) for t in types]
    colors_t = [CORRUPTION_COLORS.get(t, ACCENT) for t in types]
    bars = ax.bar(types, rates, color=colors_t, edgecolor="#0d0e1a", lw=0.5, alpha=0.85)
    for bar, v, n in zip(bars, rates, counts):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.015,
                f"{v:.2f}\n(n={n})", ha="center", va="bottom", fontsize=8.5, color="#e0e0ff")
    ax.set_ylim(0, 1.2); ax.set_title("Detection Rate by Corruption Type", pad=8)
    ax.set_ylabel("Detection Score (mean)")
    ax.set_xticklabels([t.replace("_", "\n") for t in types], fontsize=8)
    ax.axhline(0.5, color="#606880", lw=1, ls=":", label="chance")
    ax.legend(framealpha=0.3); ax.grid(True, axis="y")

    # 5d: Reward by domain
    ax = axes[1, 0]
    by_dom = defaultdict(list)
    for r in records:
        by_dom[r["domain"]].append(r["reward"])
    doms = sorted(by_dom)
    dom_means = [np.mean(by_dom[d]) for d in doms]
    dom_counts = [len(by_dom[d]) for d in doms]
    colors_d = [DOMAIN_COLORS.get(d, ACCENT) for d in doms]
    bars = ax.bar(doms, dom_means, color=colors_d, edgecolor="#0d0e1a", lw=0.5, alpha=0.85)
    for bar, v, n in zip(bars, dom_means, dom_counts):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.01,
                f"{v:.3f}\n(n={n})", ha="center", va="bottom", fontsize=8.5, color="#e0e0ff")
    ax.set_ylim(0, max(dom_means) * 1.3 + 0.05)
    ax.set_title("Mean Reward by Domain", pad=8); ax.set_ylabel("Mean Reward")
    ax.set_xticklabels([d.replace("_", "\n") for d in doms], fontsize=9)
    ax.grid(True, axis="y")

    # 5e: Confusion matrix
    ax = axes[1, 1]
    tp = fp = tn = fn = 0
    for r in records:
        det = r["detection_score"] > 0.5
        cor = bool(r["corruption_present"])
        if det and cor: tp += 1
        elif det and not cor: fp += 1
        elif not det and cor: fn += 1
        else: tn += 1
    mat = np.array([[tp, fn], [fp, tn]])
    labels_m = [["TP", "FN"], ["FP", "TN"]]
    colors_m = np.array([[GREEN, RED], [YELLOW, TEAL]])
    for i in range(2):
        for j in range(2):
            rect = mpatches.FancyBboxPatch((j + 0.05, 1 - i + 0.05), 0.9, 0.9,
                                           boxstyle="round,pad=0.02", lw=1,
                                           edgecolor="#2a2d50",
                                           facecolor=colors_m[i][j], alpha=0.35)
            ax.add_patch(rect)
            ax.text(j + 0.5, 1 - i + 0.5, f"{labels_m[i][j]}\n{mat[i, j]}",
                    ha="center", va="center", fontsize=14, fontweight="bold", color="#e0e0ff")
    ax.set_xlim(0, 2); ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5]); ax.set_yticks([0.5, 1.5])
    ax.set_xticklabels(["Predicted\nCorrupted", "Predicted\nClean"], fontsize=9)
    ax.set_yticklabels(["Actual\nClean", "Actual\nCorrupted"], fontsize=9)
    ax.set_title("Detection Confusion Matrix", pad=8)
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec  = tp / (tp + fn) if (tp + fn) else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    ax.text(1.0, -0.18, f"Precision={prec:.2f}  Recall={rec:.2f}  F1={f1:.2f}",
            ha="center", transform=ax.transAxes, fontsize=8.5, color="#9aa3c2")

    # 5f: Clean vs Corrupted
    ax = axes[1, 2]
    clean = [r for r in records if not r["corruption_present"]]
    dirty = [r for r in records if r["corruption_present"]]
    comps_s = ["detection_score", "explanation_score", "correction_score",
               "calibration_score", "reward"]
    short = ["Detect", "Explain", "Correct", "Calibrate", "Reward"]
    x = np.arange(len(comps_s))
    w = 0.35
    cl_m = [np.mean([r[c] for r in clean]) if clean else 0 for c in comps_s]
    di_m = [np.mean([r[c] for r in dirty]) if dirty else 0 for c in comps_s]
    ax.bar(x - w/2, cl_m, w, label=f"Clean (n={len(clean)})", color=GREEN, alpha=0.8, edgecolor="#0d0e1a")
    ax.bar(x + w/2, di_m, w, label=f"Corrupted (n={len(dirty)})", color=RED, alpha=0.8, edgecolor="#0d0e1a")
    ax.set_xticks(x); ax.set_xticklabels(short)
    ax.set_ylim(0, 1.1); ax.set_title("Score Breakdown: Clean vs Corrupted", pad=8)
    ax.set_ylabel("Mean Score"); ax.legend(framealpha=0.3); ax.grid(True, axis="y")

    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d0e1a")
    plt.close(fig)
    print(f"  Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 6 (BONUS): Copy double-rise curve into plots_final for one-stop access
# ═══════════════════════════════════════════════════════════════════════════════
def copy_double_rise(out_dir: Path):
    src = ROOT / "outputs" / "overseer-grpo-phase2" / "plots" / "double_rise_reward_curve.png"
    dst = out_dir / "double_rise_reward_curve.png"
    if src.exists():
        import shutil
        shutil.copy2(src, dst)
        print(f"  Copied: {dst}")
    else:
        print(f"  WARNING: {src} not found, skipping copy")


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE (printed to console + saved as text)
# ═══════════════════════════════════════════════════════════════════════════════
def print_summary(baseline_sum: dict, trained_sum: dict, out: Path):
    lines = []
    lines.append("=" * 62)
    lines.append("  FORGE + ARENA — Evaluation Comparison")
    lines.append("=" * 62)
    lines.append(f"  {'Metric':<24} {'Baseline':>10} {'Trained':>10} {'Δ':>10}")
    lines.append("-" * 62)

    metrics = [
        ("Mean Reward",       "mean_reward"),
        ("Detection Accuracy", "detection_accuracy"),
        ("Mean Detection",    "mean_detection"),
        ("Mean Explanation",  "mean_explanation"),
        ("Mean Correction",   "mean_correction"),
    ]
    for label, key in metrics:
        b = baseline_sum.get(key, 0)
        t = trained_sum.get(key, 0)
        d = t - b
        sign = "+" if d >= 0 else ""
        lines.append(f"  {label:<24} {b:>10.4f} {t:>10.4f} {sign}{d:>9.4f}")

    lines.append("-" * 62)
    lines.append(f"  Episodes:  Baseline={baseline_sum.get('episodes', '?')}  "
                 f"Trained={trained_sum.get('episodes', '?')}")
    lines.append("=" * 62)

    text = "\n".join(lines)
    print(text)
    (out.parent / "summary_comparison.txt").write_text(text)
    print(f"\n  Saved: {out.parent / 'summary_comparison.txt'}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    baseline = load_records(BASELINE_PATH)
    trained  = load_records(TRAINED_PATH)
    baseline_sum = load_summary(BASELINE_PATH)
    trained_sum  = load_summary(TRAINED_PATH)
    p1_log = json.loads(P1_LOG_PATH.read_text())
    p3_log = json.loads(P3_LOG_PATH.read_text())
    # Filter out summary entries (no "step" or step=None)
    p1_log = [e for e in p1_log if e.get("step")]
    p3_log = [e for e in p3_log if e.get("step")]

    print(f"  Baseline records: {len(baseline)}")
    print(f"  Trained records:  {len(trained)}")
    print(f"  Phase 1 log:      {len(p1_log)} entries")
    print(f"  Phase 3 log:      {len(p3_log)} entries")
    print()

    print("Generating plots...")
    plot_before_after(baseline, trained, OUT_DIR / "before_after_eval.png")
    plot_training_dynamics(p1_log, p3_log, OUT_DIR / "training_dynamics.png")
    plot_corruption_radar(baseline, trained, OUT_DIR / "corruption_radar.png")
    plot_episode_waterfall(trained, OUT_DIR / "episode_waterfall.png")
    plot_trained_dashboard(trained, OUT_DIR / "trained_eval.png")
    copy_double_rise(OUT_DIR)
    print()

    print_summary(baseline_sum, trained_sum, OUT_DIR / "summary.txt")
    print("\nAll plots saved to plots_final/")


if __name__ == "__main__":
    main()
