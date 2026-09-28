"""Replay saved policy actions and independently check physical gripper response."""
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from tabletop.runtime import configure_gpu

configure_gpu()
from tabletop.scene import TabletopEnv
from tabletop.skills import Skills

out = Path("runs") / ("gripper-diagnostic-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
out.mkdir(parents=True)
env = TabletopEnv()
report = {"air_cycles": [], "replay": []}


def snapshot():
    obs = env._get_observations(force_update=True)
    qpos = np.asarray(obs["robot0_gripper_qpos"])
    return {"finger_qpos": qpos.tolist(), "joint_spread_m": float(qpos[0] - qpos[1]),
            "eef": np.asarray(obs["robot0_eef_pos"]).tolist(),
            "red": env.object_pos("red").tolist(),
            "red_grasped": bool(env._check_grasp(env.robots[0].gripper["right"], env.cubes["red"]))}


try:
    env.reset_seed(0)
    for cycle in range(3):
        record = {"cycle": cycle}
        for label, command in (("open", -1), ("close", 1), ("reopen", -1)):
            for _ in range(25):
                env.step([0] * 6 + [command])
            record[label] = snapshot()
        report["air_cycles"].append(record)

    source = Path("runs/pi05-probe/result.json")
    data = json.loads(source.read_text())
    env.reset_seed(data["seed"])
    for _ in range(15):
        env.step([0] * 6 + [-1])
    report["replay_source"] = str(source)
    report["replay"].append({"step": 0, **snapshot()})
    for step, action in enumerate(data["actions"], 1):
        env.step(action)
        if step % 5 == 0:
            report["replay"].append({"step": step, "command": action[-1], **snapshot()})
    # Counterfactual only: do not modify the deployed policy or claim this is model behavior.
    for _ in range(25):
        env.step([0] * 6 + [1])
    report["forced_close_at_policy_endpoint"] = snapshot()
    env.reset_seed(0)
    skills = Skills(env)
    skills.wait(15)
    report["scripted_pick_control"] = skills.execute("pick", "red")
    report["scripted_pick_final"] = snapshot()
finally:
    (out / "result.json").write_text(json.dumps(report, indent=2))
    env.close()
    print(out, flush=True)
    print(json.dumps({k: v for k, v in report.items() if k != "replay"}, indent=2), flush=True)
