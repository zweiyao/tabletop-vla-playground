"""Validated, immutable hybrid settings; paths resolve relative to the project."""
import os
import tomllib
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

ROOT = Path(__file__).resolve().parents[2]


class HybridConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    review_steps: Literal[10] = 10
    execute_steps: Literal[5] = 5
    default_max_steps: int = Field(default=250, ge=5, le=1000)
    image_size: int = Field(default=384, ge=128, le=512)
    translation_limit_m: float = Field(default=0.01, gt=0, le=0.05)
    rotation_limit_rad: float = Field(default=0.05, gt=0, le=0.35)
    timeout_seconds: float = Field(default=30, gt=0, le=120)
    max_output_tokens: int = Field(default=1024, ge=256, le=4096)
    temperature: float = Field(default=0, ge=0, le=1)
    enforce_budget: bool = True
    budget_usd: float = Field(default=1, gt=0, le=1)
    stop_spend_usd: float = Field(default=0.9, gt=0)
    ledger_path: str = "runs/openrouter-budget.json"
    prompt_path: str = "prompts/hybrid_review.md"
    default_model: str = "qwen/qwen3.7-flash"
    default_weight: Literal["pi05", "pi05_lora"] = "pi05_lora"

    @model_validator(mode="after")
    def check_limits(self):
        if self.default_max_steps % self.execute_steps or self.stop_spend_usd >= self.budget_usd:
            raise ValueError("步数必须整除执行块，停止费用阈值必须低于总预算")
        return self


class ReviewerModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    id: str
    label: str
    input_price: float = Field(ge=0)
    output_price: float = Field(ge=0)
    image_price: float = Field(default=0, ge=0)
    budget_input_price: float = Field(ge=0)
    budget_output_price: float = Field(ge=0)
    context_tokens: int = Field(gt=0)
    response_format: Literal["json_schema", "json_object"]
    reasoning: Literal["off", "minimal", "none"]
    disabled_reason: str | None = None

    @model_validator(mode="after")
    def check_budget_prices(self):
        if self.budget_input_price < self.input_price or self.budget_output_price < self.output_price:
            raise ValueError("预算预留单价不能低于基础单价")
        return self


def load_config(directory=None):
    directory = Path(directory or os.getenv("TABLETOP_CONFIG_DIR", ROOT / "configs"))
    config = HybridConfig.model_validate(tomllib.loads((directory / "hybrid.toml").read_text()))
    values = [ReviewerModel.model_validate(v) for v in tomllib.loads((directory / "models.toml").read_text())["models"]]
    models = {m.id: m for m in values}
    if len(models) != len(values) or config.default_model not in models or models[config.default_model].disabled_reason:
        raise ValueError("审查模型 ID 重复或默认模型不存在")
    return config, models
