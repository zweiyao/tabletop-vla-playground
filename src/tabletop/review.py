"""Pure review contract and bounded action edits, independent of HTTP and MuJoCo."""
from typing import Literal
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictStr


class Correction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    translation_m: list[StrictFloat] = Field(min_length=3, max_length=3)
    rotation_rad: list[StrictFloat] = Field(min_length=3, max_length=3)
    gripper: Literal["keep", "open", "close"]


class ReviewDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    proposal_id: StrictStr
    previous_assessment: str = Field(min_length=1, max_length=400)
    next_intent: str = Field(min_length=1, max_length=400)
    decision: Literal["accept", "correct"]
    reason: str = Field(min_length=1, max_length=400)
    corrections: list[Correction] | None


def validate_proposal(proposal, config):
    actions = np.asarray(proposal, dtype=float)
    if actions.shape != (config.review_steps, 7) or not np.isfinite(actions).all() or np.any(np.abs(actions) > 1):
        raise ValueError("π0.5 提案必须是有限、范围内的 10×7 动作")
    return actions


def apply_review(proposal, decision, proposal_id, config):
    actions = validate_proposal(proposal, config)
    if decision.proposal_id != proposal_id:
        raise ValueError("审查结果对应的提案已失效")
    result = actions[:config.execute_steps].copy()
    if decision.decision == "accept":
        if decision.corrections is not None:
            raise ValueError("放行决策不能携带纠正动作")
        return result
    if decision.corrections is None or len(decision.corrections) != config.execute_steps:
        raise ValueError("纠正决策必须包含恰好五步修正")
    for action, correction in zip(result, decision.corrections):
        xyz, rotation = np.asarray(correction.translation_m), np.asarray(correction.rotation_rad)
        if not np.isfinite(xyz).all() or not np.isfinite(rotation).all():
            raise ValueError("修正包含非有限数值")
        if np.linalg.norm(xyz) > config.translation_limit_m + 1e-9 or np.linalg.norm(rotation) > config.rotation_limit_rad + 1e-9:
            raise ValueError("修正超出单步位置或旋转幅度")
        action[:3] += xyz / 0.05
        action[3:6] += rotation / 0.5
        if correction.gripper != "keep":
            action[6] = {"open": -1.0, "close": 1.0}[correction.gripper]
    if np.any(np.abs(result) > 1):
        raise ValueError("修正后动作超出控制器范围，拒绝执行")
    return result
