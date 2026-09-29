"""Build robot-only review observations and audit each bounded intervention."""
import hashlib
import json
import math
from pathlib import Path
import uuid
import numpy as np
from PIL import Image
from .hybrid_config import ROOT
from .openrouter import OpenRouterReviewer, SpendLedger
from .review import apply_review, validate_proposal


def robot_snapshot(env):
    """Use the actual OSC origin, without reading object bodies or rewards."""
    robot = env.robots[0]
    controller = robot.part_controllers["right"]
    if controller.input_ref_frame != "base" or controller.input_type != "delta":
        raise ValueError("混合模式要求当前 Panda 的基座系 OSC 增量控制器")
    origin = np.asarray(controller.origin_pos)
    rotation = np.asarray(controller.origin_ori)
    site = robot.eef_site_id["right"]
    obs = env._get_observations(force_update=True)
    result = {"frame": "controller_base", "eef_position_m": (rotation.T @ (obs["robot0_eef_pos"] - origin)).tolist(),
              "eef_rotation_matrix": (rotation.T @ env.sim.data.site_xmat[site].reshape(3, 3)).tolist(),
              "finger_joint_positions_m": obs["robot0_gripper_qpos"].tolist(), "cameras": {}}
    for name, camera in (("front", "frontview"), ("wrist", "robot0_eye_in_hand")):
        index = env.sim.model.camera_name2id(camera)
        result["cameras"][name] = {"position_base_m": (rotation.T @ (env.sim.data.cam_xpos[index] - origin)).tolist(),
                                   "rotation_camera_to_base": (rotation.T @ env.sim.data.cam_xmat[index].reshape(3, 3)).tolist(),
                                   "vertical_fov_degrees": float(env.sim.model.cam_fovy[index])}
    return result


class HybridSession:
    def __init__(self, config, model, reviewer=None):
        self.config, self.model = config, model
        self.prompt = (ROOT / config.prompt_path).read_text()
        self.reviewer = reviewer or OpenRouterReviewer(config, model, self.prompt, SpendLedger(ROOT / config.ledger_path, config))
        self.records = []
        self.previous_images = None
        self.current_path = None

    def snapshot(self):
        return {"config": self.config.model_dump(), "reviewer": self.model.model_dump(),
                "prompt_sha256": hashlib.sha256(self.prompt.encode()).hexdigest()}

    def preflight(self, stop):
        self.reviewer.preflight(stop)

    def choose(self, env, proposal, instruction, english, initial_robot, scene_id, step, out, stop):
        proposal = validate_proposal(proposal, self.config)
        identifier = f"{scene_id}:{step}:{uuid.uuid4().hex}"
        images = env.images(self.config.image_size)
        directory = Path(out) / f"review-{len(self.records):04d}"
        directory.mkdir()
        for name, pixels in images.items():
            Image.fromarray(pixels).save(directory / f"{name}.png")
        previous = self.records[-1] if self.records else None
        context = {"proposal_id": identifier, "scene_id": scene_id, "step": step,
                   "task": instruction, "policy_task": english, "robot": robot_snapshot(env),
                   "initial_robot": initial_robot, "control_frequency_hz": env.control_freq,
                   "proposal_normalized": proposal.tolist(),
                   "proposal_gripper_labels": ["CLOSE" if a[6] > 0 else "OPEN" if a[6] < 0 else "HOLD" for a in proposal],
                   "proposal_metric": (proposal * [0.05, 0.05, 0.05, 0.5, 0.5, 0.5, 1]).tolist(),
                   "translation_correction_limit_m": self.config.translation_limit_m,
                   "rotation_correction_limit_rad": self.config.rotation_limit_rad,
                   "previous_execution": None if previous is None else {
                       "executed_actions": previous["executed_actions"], "decision": previous.get("decision")}}
        attachments = [(f"当前观察 / {name}", pixels) for name, pixels in images.items()]
        if self.previous_images is not None:
            attachments += [(f"上一轮执行前观察 / {name}", pixels) for name, pixels in self.previous_images.items()]
        record = {"context": context, "directory": str(directory), "executed_actions": []}
        self.records.append(record)
        self.current_path = directory / "review.json"
        try:
            decision = self.reviewer.review(context, attachments, stop)
            if stop.is_set():
                raise RuntimeError("用户已停止")
            selected = apply_review(proposal, decision, identifier, self.config)
            record["decision"] = decision.model_dump()
            record["selected_actions"] = selected.tolist()
            self.previous_images = images
            return selected
        except (ValueError, RuntimeError) as exc:
            record["error"] = str(exc)
            raise
        finally:
            record["response"] = self.reviewer.last_response
            self.current_path.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False))

    def record_execution(self, actions):
        self.records[-1]["executed_actions"] = actions
        self.current_path.write_text(json.dumps(self.records[-1], ensure_ascii=False, indent=2, allow_nan=False))

    def status(self):
        if not self.records or "decision" not in self.records[-1]:
            return "等待审查"
        decision = self.records[-1]["decision"]
        label = "原样执行" if decision["decision"] == "accept" else "已纠正"
        return f"{label}：{decision['reason']}"

    def metrics(self):
        responses = [r.get("response") or {} for r in self.records]
        costs = [r.get("accounted_cost_usd") for r in responses]
        valid_costs = [c for c in costs if type(c) in (int, float) and math.isfinite(c) and c >= 0]
        return {"reviews": len(self.records), "corrections": sum(r.get("decision", {}).get("decision") == "correct" for r in self.records),
                "api_cost_usd": sum(valid_costs), "unconfirmed_costs": len(costs) - len(valid_costs),
                "review_seconds": sum(r.get("seconds", 0) for r in responses)}
