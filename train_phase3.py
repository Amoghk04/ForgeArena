"""
Forge + Arena — Phase 3: Resume GRPO Training on Harder Tasks + Double-Rise Plot

Loads the Phase 1 checkpoint, trains on the harder Phase 2 dataset,
and generates the double-rise reward curve plot.

Usage:
    # Requires Arena server running, Phase 1 + Phase 2 complete:
    #   /home/abhay/miniconda3/envs/motioncanvas/bin/python train_phase3.py
"""
from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoTokenizer, BitsAndBytesConfig, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("train_phase3")

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

SERVER_URL           = "http://localhost:8000"
HF_TOKEN             = os.environ.get("HF_TOKEN", "")
OVERSEER_LOCAL_DIR   = "models/overseer"

# Phase 1 outputs (input to Phase 3)
PHASE1_OUTPUT_DIR    = "outputs/overseer-grpo"
PHASE2_DATASET_PATH  = "datasets/overseer-episodes-phase2"

# Phase 3 outputs
PHASE3_OUTPUT_DIR    = "outputs/overseer-grpo-phase2"
PHASE3_MAX_STEPS     = 200
PHASE3_LEARNING_RATE = 1e-4

# Same training config as Phase 1
PER_DEVICE_BATCH_SIZE = 16
GRADIENT_ACCUMULATION = 1
GRPO_NUM_GENERATIONS  = 16
TEMPERATURE           = 1.0
MAX_NEW_TOKENS        = 512
WARMUP_STEPS          = 20
GRPO_BETA             = 0.04
GRPO_LOSS_TYPE        = "grpo"
SCALE_REWARDS         = "group"
FORMAT_BONUS          = 0.0

# QLoRA (same as Phase 1)
LORA_R               = 16
LORA_ALPHA            = 32
LORA_DROPOUT          = 0.05
LORA_TARGET_MODULES   = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

MODEL_PATH = OVERSEER_LOCAL_DIR if Path(OVERSEER_LOCAL_DIR).exists() else "Qwen/Qwen2.5-1.5B-Instruct"

SYSTEM_PROMPT = (
    "You are an AI Overseer. You will receive a task description and the Worker's "
    "chain-of-thought and output. You must decide whether the Worker's output "
    "contains a corruption, explain your reasoning, and if necessary provide a "
    "corrected output.\n\n"
    "Respond in JSON with the following keys:\n"
    "  corruption_detected (bool)\n"
    "  confidence (float 0-1)\n"
    "  explanation (str)\n"
    "  correction (str or null)\n"
)

REWARD_KEYS = ["rewards/arena_reward/mean", "rewards/arena_reward", "reward", "arena_reward"]

# ═══════════════════════════════════════════════════════════════════════════════
# Reward Function (same as Phase 1)
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_completion_text(completion) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        if not completion:
            return ""
        last = completion[-1]
        if isinstance(last, dict):
            content = last.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return " ".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
            return str(content)
        return str(last) if isinstance(last, str) else str(last)
    return str(completion)


def _parse_completion(text: str) -> dict[str, Any]:
    if not isinstance(text, str) or not text.strip():
        return {}
    try:
        stripped = text.strip()
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            stripped = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        parsed = json.loads(stripped)
        if isinstance(parsed, list):
            return parsed[0] if parsed and isinstance(parsed[0], dict) else {}
        if isinstance(parsed, dict):
            return parsed
        return {}
    except (json.JSONDecodeError, ValueError, IndexError):
        return {}


class ArenaRewardFunction:
    def __init__(self, server_url: str, format_bonus: float = 0.15) -> None:
        self._base = server_url.rstrip("/")
        self._client = httpx.Client(timeout=30.0)
        self._format_bonus = format_bonus
        self.__name__ = "arena_reward"

    def __call__(self, prompts, completions, episode_id, corruption_present, corruption_type,
                 ground_truth_output, worker_output, domains=None, **kwargs) -> list[float]:
        _domains = domains or ["customer_support"] * len(completions)
        def _grade_one(i):
            text = _extract_completion_text(completions[i])
            action = _parse_completion(text)
            has_valid_json = bool(action) and "corruption_detected" in action
            bonus = self._format_bonus if has_valid_json else 0.0
            payload = {
                "episode_id": episode_id[i], "domain": _domains[i],
                "corruption_present": corruption_present[i], "corruption_type": corruption_type[i],
                "ground_truth_output": ground_truth_output[i],
                "overseer_detection": action.get("corruption_detected", False),
                "overseer_confidence": action.get("confidence", 0.5),
                "overseer_explanation": action.get("explanation", ""),
                "overseer_correction": action.get("correction") or "",
            }
            try:
                resp = self._client.post(f"{self._base}/grader", json=payload)
                resp.raise_for_status()
                return float(resp.json()["composite"]) + bonus
            except Exception:
                return bonus
        with ThreadPoolExecutor(max_workers=min(len(completions), 8)) as pool:
            return list(pool.map(_grade_one, range(len(completions))))

    def close(self):
        self._client.close()


class ProgressCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        step = state.global_step
        reward = logs.get("rewards/arena_reward/mean", logs.get("reward", "---"))
        loss = logs.get("loss", "---")
        lr = logs.get("learning_rate", "---")
        if isinstance(reward, float): reward = f"{reward:.4f}"
        if isinstance(loss, float): loss = f"{loss:.4f}"
        if isinstance(lr, float): lr = f"{lr:.2e}"
        pct = 100 * step / args.max_steps
        print(f"  [{step:>5}/{args.max_steps}] ({pct:5.1f}%)  reward={reward}  loss={loss}  lr={lr}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Phase 3 — Resume GRPO on Harder Forge Tasks")
    print("=" * 60)

    # ── Load Phase 1 log history ──────────────────────────────────────────────
    p1_log_path = Path(PHASE1_OUTPUT_DIR) / "phase1_log_history.json"
    if p1_log_path.exists():
        with open(p1_log_path) as f:
            phase1_log_history = json.load(f)
        phase1_rewards = [next((e[k] for k in REWARD_KEYS if k in e), None)
                          for e in phase1_log_history if e.get("step")]
        phase1_rewards = [r for r in phase1_rewards if r is not None]
        phase1_ceiling = max(phase1_rewards[-10:]) if len(phase1_rewards) >= 10 else (max(phase1_rewards) if phase1_rewards else 0.0)
        print(f"  Phase 1 ceiling: {phase1_ceiling:.4f}")
    else:
        phase1_log_history = []
        phase1_ceiling = 0.4
        print(f"  Phase 1 log not found — using default ceiling {phase1_ceiling}")

    # ── Load Phase 2 dataset ──────────────────────────────────────────────────
    phase2_dataset = Dataset.load_from_disk(PHASE2_DATASET_PATH)
    if isinstance(phase2_dataset[0]["prompt"], str):
        def _to_conv(row):
            row["prompt"] = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": row["prompt"]}]
            return row
        phase2_dataset = phase2_dataset.map(_to_conv)
    print(f"  Phase 2 dataset: {len(phase2_dataset)} rows")

    # ── Tokenizer + model kwargs ──────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(OVERSEER_LOCAL_DIR, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "quantization_config": quantization_config,
        "device_map": "auto",
    }
    peft_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES, task_type=TaskType.CAUSAL_LM, bias="none",
    )

    # ── GRPOConfig for Phase 3 ───────────────────────────────────────────────
    grpo_config = GRPOConfig(
        output_dir=PHASE3_OUTPUT_DIR,
        max_steps=PHASE3_MAX_STEPS,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=PHASE3_LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        beta=GRPO_BETA,
        loss_type=GRPO_LOSS_TYPE,
        scale_rewards=SCALE_REWARDS,
        num_generations=GRPO_NUM_GENERATIONS,
        max_completion_length=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        logging_steps=5,
        save_steps=50,
        bf16=True,
        report_to="none",
        model_init_kwargs=model_kwargs,
        log_completions=True,
    )

    reward_fn = ArenaRewardFunction(SERVER_URL, format_bonus=FORMAT_BONUS)

    # GRPOTrainer needs the base model path (not LoRA adapter dir).
    # It will create fresh LoRA adapters via peft_config, then we load
    # Phase 1 adapter weights on top.
    trainer = GRPOTrainer(
        model=MODEL_PATH,           # base model, NOT Phase 1 adapter dir
        args=grpo_config,
        processing_class=tokenizer,
        train_dataset=phase2_dataset,
        reward_funcs=[reward_fn],
        peft_config=peft_config,
    )

    # Load Phase 1 LoRA weights into the fresh adapters
    import safetensors.torch
    p1_weights_path = Path(PHASE1_OUTPUT_DIR) / "adapter_model.safetensors"
    if p1_weights_path.exists():
        p1_state = safetensors.torch.load_file(str(p1_weights_path))
        # Remap key names: saved has "lora_A.weight" but model expects "lora_A.default.weight"
        remapped = {}
        for k, v in p1_state.items():
            new_key = k.replace("lora_A.weight", "lora_A.default.weight").replace("lora_B.weight", "lora_B.default.weight")
            remapped[new_key] = v
        missing, unexpected = trainer.model.load_state_dict(remapped, strict=False)
        loaded = len(remapped) - len(unexpected)
        print(f"  Loaded {loaded} Phase 1 LoRA weights from {p1_weights_path}")
        if unexpected:
            print(f"  (unexpected keys: {len(unexpected)})")
    else:
        print(f"  WARNING: No Phase 1 weights at {p1_weights_path} — training from base model")

    trainer.add_callback(ProgressCallback())

    print(f"  Loaded Phase 1 checkpoint: {PHASE1_OUTPUT_DIR}")
    print(f"  Phase 3 LR: {PHASE3_LEARNING_RATE}")
    print(f"  Phase 3 steps: {PHASE3_MAX_STEPS}")
    print()

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer.train()
    trainer.save_model(PHASE3_OUTPUT_DIR)
    tokenizer.save_pretrained(PHASE3_OUTPUT_DIR)
    reward_fn.close()

    # ── Save Phase 3 log history ──────────────────────────────────────────────
    phase3_log_history = list(trainer.state.log_history)
    with open(Path(PHASE3_OUTPUT_DIR) / "phase3_log_history.json", "w") as f:
        json.dump(phase3_log_history, f)

    phase3_rewards = [next((e[k] for k in REWARD_KEYS if k in e), None)
                      for e in phase3_log_history if e.get("step")]
    phase3_rewards = [r for r in phase3_rewards if r is not None]
    phase3_final = max(phase3_rewards[-10:]) if len(phase3_rewards) >= 10 else (max(phase3_rewards) if phase3_rewards else 0.0)

    print(f"\n=== Phase 3 complete ===")
    print(f"  Phase 1 ceiling  : {phase1_ceiling:.4f}")
    print(f"  Phase 3 final    : {phase3_final:.4f}  ({phase3_final - phase1_ceiling:+.4f})")
    if phase3_final > phase1_ceiling:
        print(f"  >> DOUBLE-RISE ACHIEVED")

    # ═══════════════════════════════════════════════════════════════════════════
    # Double-Rise Plot
    # ═══════════════════════════════════════════════════════════════════════════
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": "#0d0e1a", "axes.facecolor": "#12132a",
        "axes.edgecolor": "#2a2d50", "axes.labelcolor": "#c0c4e0",
        "xtick.color": "#9aa3c2", "ytick.color": "#9aa3c2",
        "text.color": "#e0e0ff", "grid.color": "#1e2040",
        "grid.linestyle": "--", "grid.alpha": 0.6,
        "font.family": "monospace", "font.size": 10,
        "legend.facecolor": "#12132a", "legend.edgecolor": "#2a2d50",
    })
    ACCENT = "#5b6bff"; GREEN = "#4ade80"; RED = "#f87171"; YELLOW = "#fbbf24"

    # Extract Phase 1 steps/rewards
    p1s, p1r = [], []
    for e in phase1_log_history:
        s = e.get("step")
        r = next((e[k] for k in REWARD_KEYS if k in e), None)
        if s and r is not None:
            p1s.append(s); p1r.append(r)
    p1_final_step = max(p1s) if p1s else 0

    # Extract Phase 3 steps/rewards (offset by Phase 1 final step)
    p3s, p3r = [], []
    for e in phase3_log_history:
        s = e.get("step")
        r = next((e[k] for k in REWARD_KEYS if k in e), None)
        if s and r is not None:
            p3s.append(p1_final_step + s); p3r.append(r)

    def sm(xs, ys, w=8):
        if len(ys) < w:
            return xs, ys
        k = np.ones(w) / w
        s = np.convolve(ys, k, mode="valid")
        h = w // 2
        return xs[h:h+len(s)], list(s)

    plots_dir = Path(PHASE3_OUTPUT_DIR) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    if p1r:
        ax.plot(p1s, p1r, color=ACCENT, lw=1, alpha=0.3)
    if p3r:
        ax.plot(p3s, p3r, color=GREEN, lw=1, alpha=0.3)
    if p1r:
        sx, sy = sm(p1s, p1r)
        ax.plot(sx, sy, color=ACCENT, lw=2.5, label="Phase 1 (static)")
    if p3r:
        sx, sy = sm(p3s, p3r)
        ax.plot(sx, sy, color=GREEN, lw=2.5, label="Phase 3 (Forge harder)")
    ax.axvline(p1_final_step, color=YELLOW, lw=1.5, ls="--", alpha=0.8, label="Forge activated")
    ax.axhline(phase1_ceiling, color=RED, lw=1, ls=":", alpha=0.5, label=f"Phase 1 ceiling ({phase1_ceiling:.3f})")
    ax.set_title("Forge + Arena — Double-Rise Reward Curve", fontsize=14, pad=10)
    ax.set_xlabel("Step")
    ax.set_ylabel("Composite Reward")
    ax.legend(loc="lower right", framealpha=0.4)
    ax.grid(True)
    fig.tight_layout()
    plot_path = plots_dir / "double_rise_reward_curve.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot saved: {plot_path}")

    print("\n" + "=" * 60)
    print("All 3 phases complete!")
    print(f"  Phase 1 adapters : {PHASE1_OUTPUT_DIR}")
    print(f"  Phase 3 adapters : {PHASE3_OUTPUT_DIR}")
    print(f"  Double-rise plot : {plot_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
