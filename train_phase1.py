"""
Forge + Arena — Phase 1 GRPO Training Script

DAPO + scale_rewards=none + format_bonus + QLoRA

Usage:
    # Start Arena server first:
    #   uvicorn forge_arena.main:app --host 0.0.0.0 --port 8000
    #
    # Then run training:
    #   cd /home/abhay/ForgeArena
    #   /home/abhay/miniconda3/envs/motioncanvas/bin/python train_phase1.py
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoTokenizer, BitsAndBytesConfig, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("train_phase1")

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

SERVER_URL           = "http://localhost:8000"
HF_TOKEN             = os.environ.get("HF_TOKEN", "")
OVERSEER_LOCAL_DIR   = "models/overseer"
OVERSEER_HUB_ID      = "Qwen/Qwen2.5-1.5B-Instruct"
DATASET_PATH         = "datasets/overseer-episodes"
OUTPUT_DIR           = "outputs/overseer-grpo"

# QLoRA
LORA_R               = 16
LORA_ALPHA            = 32
LORA_DROPOUT          = 0.05
LORA_TARGET_MODULES   = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Training
PHASE1_MAX_STEPS      = 200
PER_DEVICE_BATCH_SIZE = 16
GRADIENT_ACCUMULATION = 1
LEARNING_RATE         = 2e-4
GRPO_NUM_GENERATIONS  = 16
TEMPERATURE           = 1.0
MAX_NEW_TOKENS        = 512
WARMUP_STEPS          = 20

# GRPO config
GRPO_BETA             = 0.04      # KL penalty for gradient signal
GRPO_LOSS_TYPE        = "grpo"    # Standard GRPO loss
SCALE_REWARDS         = "group"   # Normalize rewards within groups

# Format bonus
FORMAT_BONUS          = 0.0       # Disabled — was masking real learning signal

# Resolve model path
MODEL_PATH = OVERSEER_LOCAL_DIR if Path(OVERSEER_LOCAL_DIR).exists() else OVERSEER_HUB_ID

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

# ═══════════════════════════════════════════════════════════════════════════════
# Reward Function with Format Bonus
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
                return " ".join(
                    block.get("text", "") for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                )
            return str(content)
        if isinstance(last, str):
            return last
        return str(last)
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

    def __call__(
        self,
        prompts,
        completions,
        episode_id: list[str],
        corruption_present: list[bool],
        corruption_type: list[str | None],
        ground_truth_output: list[str],
        worker_output: list[str],
        domains: list[str] | None = None,
        **kwargs: Any,
    ) -> list[float]:
        _domains = domains or ["customer_support"] * len(completions)

        def _grade_one(i: int) -> float:
            text = _extract_completion_text(completions[i])
            action = _parse_completion(text)

            has_valid_json = bool(action) and "corruption_detected" in action
            bonus = self._format_bonus if has_valid_json else 0.0

            payload = {
                "episode_id": episode_id[i],
                "domain": _domains[i],
                "corruption_present": corruption_present[i],
                "corruption_type": corruption_type[i],
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
            except (httpx.HTTPError, KeyError, ValueError) as exc:
                log.warning(f"Reward call failed: {exc}")
                return bonus

        with ThreadPoolExecutor(max_workers=min(len(completions), 8)) as pool:
            return list(pool.map(_grade_one, range(len(completions))))

    def close(self) -> None:
        self._client.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Progress Callback
# ═══════════════════════════════════════════════════════════════════════════════

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
    print("Phase 1 — GRPO Training (DAPO + scale_rewards=none)")
    print("=" * 60)
    print(f"  Model        : {MODEL_PATH}")
    print(f"  LR           : {LEARNING_RATE}")
    print(f"  Beta         : {GRPO_BETA}")
    print(f"  Loss type    : {GRPO_LOSS_TYPE}")
    print(f"  Scale rewards: {SCALE_REWARDS}")
    print(f"  Temperature  : {TEMPERATURE}")
    print(f"  Format bonus : +{FORMAT_BONUS}")
    print(f"  Batch        : {PER_DEVICE_BATCH_SIZE} x {GRPO_NUM_GENERATIONS} gens")
    print(f"  Max steps    : {PHASE1_MAX_STEPS}")
    print(f"  Output       : {OUTPUT_DIR}")
    print()

    # ── Load dataset ──────────────────────────────────────────────────────────
    _hf_marker = Path(DATASET_PATH) / "dataset_info.json"
    _jsonl_path = Path(DATASET_PATH) / "rows.jsonl"

    if _hf_marker.exists():
        dataset = Dataset.load_from_disk(DATASET_PATH)
        log.info(f"Dataset loaded from disk: {len(dataset)} rows")
    elif _jsonl_path.exists():
        rows = [json.loads(line) for line in _jsonl_path.read_text().splitlines() if line.strip()]
        dataset = Dataset.from_list(rows)
        log.info(f"Dataset loaded from JSONL: {len(rows)} rows")
    else:
        raise RuntimeError(f"No dataset at {DATASET_PATH}. Collect episodes first.")

    # Convert plain text prompts to conversational
    if isinstance(dataset[0]["prompt"], str):
        log.info("Converting plain-text prompts to conversational format")
        def _to_conv(row):
            row["prompt"] = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": row["prompt"]},
            ]
            return row
        dataset = dataset.map(_to_conv)

    print(f"  Dataset rows : {len(dataset)}")
    print(f"  Prompt format: {'conversational' if isinstance(dataset[0]['prompt'], list) else 'plain'}")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    _local = Path(MODEL_PATH).exists()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=_local, token=HF_TOKEN or None)
    tokenizer.pad_token = tokenizer.eos_token

    # ── Model kwargs (QLoRA) ──────────────────────────────────────────────────
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
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

    # ── GRPOConfig ────────────────────────────────────────────────────────────
    grpo_config = GRPOConfig(
        output_dir=OUTPUT_DIR,
        max_steps=PHASE1_MAX_STEPS,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
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

    trainer = GRPOTrainer(
        model=MODEL_PATH,
        args=grpo_config,
        processing_class=tokenizer,
        train_dataset=dataset,
        reward_funcs=[reward_fn],
        peft_config=peft_config,
    )
    trainer.add_callback(ProgressCallback())

    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())
    print(f"  Trainable    : {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    print()

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer.train()

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    reward_fn.close()

    # ── Save metrics ──────────────────────────────────────────────────────────
    REWARD_KEYS = ["rewards/arena_reward/mean", "rewards/arena_reward", "reward", "arena_reward"]
    rewards = [
        next((e[k] for k in REWARD_KEYS if k in e), None)
        for e in trainer.state.log_history if e.get("step") is not None
    ]
    rewards = [r for r in rewards if r is not None]
    ceiling = max(rewards[-10:]) if len(rewards) >= 10 else (max(rewards) if rewards else 0.0)

    # Save log history for Phase 2/3 plotting
    with open(Path(OUTPUT_DIR) / "phase1_log_history.json", "w") as f:
        json.dump(trainer.state.log_history, f)

    print()
    print("=" * 60)
    print("Phase 1 complete")
    print("=" * 60)
    print(f"  LoRA adapters : {OUTPUT_DIR}")
    print(f"  Ceiling reward: {ceiling:.4f}")
    print(f"  Total rewards : {len(rewards)} logged steps")
    print(f"  Log history   : {OUTPUT_DIR}/phase1_log_history.json")


if __name__ == "__main__":
    main()
