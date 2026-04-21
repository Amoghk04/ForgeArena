"""Pydantic Settings and openenv.yaml configuration loader."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DifficultyThresholds(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    too_easy: float = 0.85
    too_hard: float = 0.20

    @model_validator(mode="after")
    def thresholds_are_ordered(self) -> "DifficultyThresholds":
        if self.too_hard >= self.too_easy:
            raise ValueError(
                f"too_hard ({self.too_hard}) must be strictly less than too_easy ({self.too_easy})"
            )
        return self


class GeneratorConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    max_variants_per_seed: int = 20
    min_acceptance_rate: float = 0.30


class ForgeConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    estimation_k: int = 8
    estimation_n_samples: int = 32  # must be >= estimation_k; n >> k for non-binary pass@k
    queue_replenishment_threshold: int = 10
    batch_reestimation_interval: int = 50
    difficulty_thresholds: DifficultyThresholds = Field(default_factory=DifficultyThresholds)
    generator: GeneratorConfig = Field(default_factory=GeneratorConfig)


class WorkerModelConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    repo_id: str = "Qwen/Qwen2.5-7B-Instruct"
    inference_api: bool = True
    max_new_tokens: int = 1024
    temperature: float = 0.7


class RewardWeights(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    detection_weight: float = 0.40
    explanation_weight: float = 0.30
    correction_weight: float = 0.20
    calibration_weight: float = 0.10

    @model_validator(mode="after")
    def weights_sum_to_one(self) -> "RewardWeights":
        total = (
            self.detection_weight
            + self.explanation_weight
            + self.correction_weight
            + self.calibration_weight
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Reward weights must sum to 1.0, got {total:.4f}")
        return self


class CorrectionThresholds(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    full_reward_threshold: float = 0.80
    zero_reward_threshold: float = 0.50
    neutral_score_when_clean: float = 0.50


class Settings(BaseSettings):
    """Application-wide settings. Values are read from environment variables
    and/or overridden by values loaded from openenv.yaml at startup."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="FORGE_",
        extra="ignore",
    )

    # HuggingFace Inference API key — accepts HF_TOKEN (standard) or HF_API_KEY
    hf_api_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("HF_TOKEN", "HF_API_KEY"),
    )

    # Path to the openenv.yaml relative to the working directory
    openenv_yaml_path: str = "openenv.yaml"
    seed_bank_path: str = "tasks/seed_tasks.json"

    # Sub-configs loaded from YAML
    forge: ForgeConfig = Field(default_factory=ForgeConfig)
    worker_model: WorkerModelConfig = Field(default_factory=WorkerModelConfig)
    reward: RewardWeights = Field(default_factory=RewardWeights)
    correction: CorrectionThresholds = Field(default_factory=CorrectionThresholds)


def _load_yaml(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open() as f:
        return yaml.safe_load(f) or {}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load and return cached application settings.

    Merges environment variables with values from openenv.yaml.
    openenv.yaml values that correspond to sub-config fields are
    injected before Pydantic validation.
    """
    raw = _load_yaml("openenv.yaml")

    # Build nested configs from YAML
    forge_raw = raw.get("forge", {})
    worker_raw = raw.get("models", {}).get("worker", {})
    reward_raw = raw.get("reward", {})
    correction_raw = raw.get("correction", {})

    return Settings(
        forge=ForgeConfig(**forge_raw),
        worker_model=WorkerModelConfig(**worker_raw),
        reward=RewardWeights(**reward_raw),
        correction=CorrectionThresholds(**correction_raw),
        seed_bank_path=raw.get("tasks", {}).get("seed_bank_path", "tasks/seed_tasks.json"),
    )
