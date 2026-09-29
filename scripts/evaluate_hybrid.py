"""Small same-seed pi0.5/hybrid comparison; scoring never enters policy inputs."""
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from tabletop.runtime import check_gpu_idle, configure_gpu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--model", default=None)
    parser.add_argument("--tasks", nargs="+", default=["red", "green", "blue", "home"])
    args = parser.parse_args()
    check_gpu_idle()
    configure_gpu()
    from tabletop.engine import Engine
    from tabletop.hybrid import HybridSession, robot_snapshot
    from tabletop.hybrid_config import ROOT
    os.environ["OPENROUTER_API_KEY"] = (ROOT / "openrouter_key").read_text().strip()
    engine = Engine(load_model=False)
    config, models = engine.hybrid_settings
    model = models[args.model or config.default_model]
    output = ROOT / "runs" / ("hybrid-comparison-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    output.mkdir(exist_ok=True)
    results = []
    try:
        engine.start_pi05("pi05_lora")
        adapter = engine.pi05.adapter
        for task in args.tasks:
            seed = 42000 + ["red", "green", "blue", "home"].index(task)
            instruction = "return the empty gripper to its initial position and orientation and keep it open" if task == "home" else f"pick up the {task} cube"
            for mode in ("pi05", "hybrid"):
                engine.reset(seed)
                if task == "home":
                    for _ in range(20): engine.env.step([.2, -.2, .1, 0, 0, 0, -1])
                session = HybridSession(config, model) if mode == "hybrid" else None
                log = engine.run_pi05(instruction, lambda *a: None, args.steps, adapter, session)
                measured = robot_snapshot(engine.env)
                if task == "home":
                    position_error = float(np.linalg.norm(np.asarray(measured["eef_position_m"]) - engine.initial_robot["eef_position_m"]))
                    rotation_error = float(np.arccos(np.clip((np.trace(np.asarray(measured["eef_rotation_matrix"]).T @ np.asarray(engine.initial_robot["eef_rotation_matrix"])) - 1) / 2, -1, 1)))
                    score = {"position_error_m": position_error, "rotation_error_rad": rotation_error,
                             "final_task_condition": bool(position_error < .03 and rotation_error < .15 and measured["finger_joint_positions_m"][0] > .025)}
                else:
                    held = bool(engine.env._check_grasp(engine.env.robots[0].gripper["right"], engine.env.cubes[task]))
                    z = float(engine.env.object_pos(task)[2])
                    score = {"held": held, "cube_z_m": z, "final_task_condition": held and z > .90}
                result = {"task": task, "seed": seed, "mode": mode, "steps": log["steps"],
                          "termination": log["termination"], "error": log.get("error"), "started": log["started"],
                          "seconds": log["seconds"], "metrics": log.get("hybrid_metrics"), "final_snapshot_score": score,
                          "note": "short integration trial; final snapshot only, not a success-rate benchmark"}
                results.append(result)
                (output / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
                print(json.dumps(result, ensure_ascii=False), flush=True)
                if session:
                    ledger = ROOT / config.ledger_path
                    if ledger.exists() and json.loads(ledger.read_text()).get("pending"):
                        raise RuntimeError("Unconfirmed API charge; comparison stopped")
    finally:
        engine.close_pi05()
        engine.env.close()
        print("Results:", output, flush=True)


if __name__ == "__main__":
    main()
