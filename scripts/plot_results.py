"""Generate analysis plots from eval results.json.

Usage:
    python scripts/plot_results.py --input results.json --output plots/
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0d0e1a",
    "axes.facecolor":   "#12132a",
    "axes.edgecolor":   "#2a2d50",
    "axes.labelcolor":  "#c0c4e0",
    "xtick.color":      "#9aa3c2",
    "ytick.color":      "#9aa3c2",
    "text.color":       "#e0e0ff",
    "grid.color":       "#1e2040",
    "grid.linestyle":   "--",
    "grid.alpha":       0.6,
    "font.family":      "monospace",
    "font.size":        10,
})

ACCENT   = "#5b6bff"
GREEN    = "#4ade80"
RED      = "#f87171"
YELLOW   = "#fbbf24"
PURPLE   = "#a78bfa"
TEAL     = "#2dd4bf"
PALETTE  = [ACCENT, GREEN, RED, YELLOW, PURPLE, TEAL]

CORRUPTION_COLORS = {
    "TEMPORAL_SHIFT":      "#2dd4bf",
    "FACTUAL_OMISSION":    "#fbbf24",
    "AUTHORITY_FABRICATION": "#f87171",
    "BIAS_INJECTION":      "#a78bfa",
    "INSTRUCTION_OVERRIDE": "#5b6bff",
}
DOMAIN_COLORS = {
    "customer_support":      "#5b6bff",
    "legal_summarisation":   "#2dd4bf",
    "code_review":           "#4ade80",
    "product_recommendation":"#fbbf24",
    "mixed":                 "#f87171",
}


def load(path: str) -> list[dict]:
    data = json.loads(Path(path).read_text())
    return [r for r in data["records"] if r["error"] in (None, "") and r["reward"] is not None]


# ── 1. Reward distribution ─────────────────────────────────────────────────────
def plot_reward_distribution(records: list[dict], ax: plt.Axes) -> None:
    rewards = [r["reward"] for r in records]
    bins = np.linspace(0, 1, 21)
    ax.hist(rewards, bins=bins, color=ACCENT, edgecolor="#0d0e1a", linewidth=0.5, alpha=0.85)
    ax.axvline(np.mean(rewards), color=YELLOW, linewidth=1.5, linestyle="--", label=f"mean={np.mean(rewards):.3f}")
    ax.set_title("Reward Distribution  (n=50)", pad=8)
    ax.set_xlabel("Composite Reward")
    ax.set_ylabel("Episodes")
    ax.legend(framealpha=0.3, edgecolor="#2a2d50")
    ax.grid(True, axis="y")


# ── 2. Component score means ───────────────────────────────────────────────────
def plot_component_means(records: list[dict], ax: plt.Axes) -> None:
    components = ["detection", "explanation", "correction", "calibration", "reward"]
    labels     = ["Detection\n(×0.40)", "Explanation\n(×0.30)", "Correction\n(×0.20)", "Calibration\n(×0.10)", "Composite\nReward"]
    means      = [np.mean([r[f"{c}_score"] if c != "reward" else r["reward"] for r in records]) for c in components]
    colors     = [ACCENT, TEAL, GREEN, PURPLE, YELLOW]
    bars = ax.bar(labels, means, color=colors, edgecolor="#0d0e1a", linewidth=0.5, alpha=0.85)
    for bar, v in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}",
                ha="center", va="bottom", fontsize=9, color="#e0e0ff")
    ax.set_ylim(0, 1.05)
    ax.set_title("Mean Score by Component", pad=8)
    ax.set_ylabel("Score [0–1]")
    ax.grid(True, axis="y")


# ── 3. Detection by corruption type ───────────────────────────────────────────
def plot_detection_by_corruption(records: list[dict], ax: plt.Axes) -> None:
    corrupted = [r for r in records if r["corruption_present"]]
    by_type: dict[str, list[float]] = defaultdict(list)
    for r in corrupted:
        by_type[r["corruption_type"]].append(r["detection_score"])

    types  = sorted(by_type)
    rates  = [np.mean(by_type[t]) for t in types]
    counts = [len(by_type[t]) for t in types]
    colors = [CORRUPTION_COLORS.get(t, ACCENT) for t in types]

    bars = ax.bar(types, rates, color=colors, edgecolor="#0d0e1a", linewidth=0.5, alpha=0.85)
    for bar, v, n in zip(bars, rates, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015,
                f"{v:.2f}\n(n={n})", ha="center", va="bottom", fontsize=8.5, color="#e0e0ff")
    ax.set_ylim(0, 1.2)
    ax.set_title("Corruption Detection Rate by Type", pad=8)
    ax.set_ylabel("Detection Score (mean)")
    ax.set_xticklabels([t.replace("_", "\n") for t in types], fontsize=8)
    ax.axhline(0.5, color="#606880", linewidth=1, linestyle=":", label="chance")
    ax.legend(framealpha=0.3, edgecolor="#2a2d50")
    ax.grid(True, axis="y")


# ── 4. Mean reward by domain ──────────────────────────────────────────────────
def plot_reward_by_domain(records: list[dict], ax: plt.Axes) -> None:
    by_domain: dict[str, list[float]] = defaultdict(list)
    for r in records:
        by_domain[r["domain"]].append(r["reward"])

    domains = sorted(by_domain)
    means   = [np.mean(by_domain[d]) for d in domains]
    counts  = [len(by_domain[d]) for d in domains]
    colors  = [DOMAIN_COLORS.get(d, ACCENT) for d in domains]

    bars = ax.bar(domains, means, color=colors, edgecolor="#0d0e1a", linewidth=0.5, alpha=0.85)
    for bar, v, n in zip(bars, means, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                f"{v:.3f}\n(n={n})", ha="center", va="bottom", fontsize=8.5, color="#e0e0ff")
    ax.set_ylim(0, max(means) * 1.3 + 0.05)
    ax.set_title("Mean Reward by Domain", pad=8)
    ax.set_ylabel("Mean Composite Reward")
    ax.set_xticklabels([d.replace("_", "\n") for d in domains], fontsize=9)
    ax.grid(True, axis="y")


# ── 5. Confusion matrix (detection) ───────────────────────────────────────────
def plot_confusion(records: list[dict], ax: plt.Axes) -> None:
    tp = fp = tn = fn = 0
    for r in records:
        detected   = r["detection_score"] > 0.5
        corrupted  = bool(r["corruption_present"])
        if detected and corrupted:     tp += 1
        elif detected and not corrupted: fp += 1
        elif not detected and corrupted: fn += 1
        else:                            tn += 1

    mat = np.array([[tp, fn], [fp, tn]])
    labels = [["TP", "FN"], ["FP", "TN"]]
    colors_mat = np.array([[GREEN, RED], [YELLOW, TEAL]])

    for i in range(2):
        for j in range(2):
            rect = mpatches.FancyBboxPatch((j + 0.05, 1 - i + 0.05), 0.9, 0.9,
                                           boxstyle="round,pad=0.02",
                                           linewidth=1, edgecolor="#2a2d50",
                                           facecolor=colors_mat[i][j], alpha=0.35)
            ax.add_patch(rect)
            ax.text(j + 0.5, 1 - i + 0.5, f"{labels[i][j]}\n{mat[i, j]}",
                    ha="center", va="center", fontsize=14, fontweight="bold", color="#e0e0ff")

    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5])
    ax.set_yticks([0.5, 1.5])
    ax.set_xticklabels(["Predicted\nCorrupted", "Predicted\nClean"], fontsize=9)
    ax.set_yticklabels(["Actual\nClean", "Actual\nCorrupted"], fontsize=9)
    ax.set_title("Detection Confusion Matrix", pad=8)
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec  = tp / (tp + fn) if (tp + fn) else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    ax.text(1.0, -0.18, f"Precision={prec:.2f}  Recall={rec:.2f}  F1={f1:.2f}",
            ha="center", transform=ax.transAxes, fontsize=8.5, color="#9aa3c2")


# ── 6. Score breakdown: corrupted vs clean ────────────────────────────────────
def plot_clean_vs_corrupted(records: list[dict], ax: plt.Axes) -> None:
    clean  = [r for r in records if not r["corruption_present"]]
    dirty  = [r for r in records if r["corruption_present"]]

    comps = ["detection_score", "explanation_score", "correction_score", "calibration_score", "reward"]
    short = ["Detect", "Explain", "Correct", "Calibrate", "Reward"]

    x   = np.arange(len(comps))
    w   = 0.35
    clean_means = [np.mean([r[c] if c != "reward" else r["reward"] for r in clean]) for c in comps]
    dirty_means = [np.mean([r[c] if c != "reward" else r["reward"] for r in dirty]) for c in comps]

    ax.bar(x - w/2, clean_means, width=w, label=f"Clean (n={len(clean)})", color=GREEN, alpha=0.8, edgecolor="#0d0e1a")
    ax.bar(x + w/2, dirty_means, width=w, label=f"Corrupted (n={len(dirty)})", color=RED, alpha=0.8, edgecolor="#0d0e1a")
    ax.set_xticks(x)
    ax.set_xticklabels(short)
    ax.set_ylim(0, 1.1)
    ax.set_title("Score Breakdown: Clean vs Corrupted Episodes", pad=8)
    ax.set_ylabel("Mean Score")
    ax.legend(framealpha=0.3, edgecolor="#2a2d50")
    ax.grid(True, axis="y")


# ── main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  default="results.json")
    ap.add_argument("--output", default="plots/")
    args = ap.parse_args()

    records = load(args.input)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Forge + Arena — Baseline Evaluation  (Qwen2.5-7B, 50 episodes)",
                 fontsize=14, y=1.01, color="#e0e0ff")
    fig.patch.set_facecolor("#0d0e1a")
    plt.subplots_adjust(hspace=0.45, wspace=0.35)

    plot_reward_distribution(records,      axes[0, 0])
    plot_component_means(records,          axes[0, 1])
    plot_detection_by_corruption(records,  axes[0, 2])
    plot_reward_by_domain(records,         axes[1, 0])
    plot_confusion(records,                axes[1, 1])
    plot_clean_vs_corrupted(records,       axes[1, 2])

    path = out / "baseline_eval.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d0e1a")
    print(f"Saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
