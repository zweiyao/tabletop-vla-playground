"""Closed-loop held-out tasks; physical success must persist for ten control steps."""
import argparse
import json
import threading
import time
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
from tabletop.runtime import configure_gpu

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="data/tabletop-four-v1")
parser.add_argument("--adapter")
parser.add_argument("--split", choices=["val", "test"], default="test")
parser.add_argument("--per-task", type=int, default=20)
parser.add_argument("--steps", type=int, default=250)
parser.add_argument("--output", required=True)
args = parser.parse_args()
configure_gpu()
from tabletop.pi05 import Pi05
from tabletop.scene import TabletopEnv
from tabletop.demonstrations import TASKS, prepare, success

out = Path(args.output)
out.mkdir(parents=True, exist_ok=True)
manifest = json.loads((Path(args.data) / "manifest.json").read_text())
episodes = [e for e in manifest["episodes"] if e["split"] == args.split and e["index"] < args.per_task]
if len(episodes) != 4 * args.per_task:
    raise ValueError("Evaluation split is not complete")
stop = threading.Event()
env = TabletopEnv()
policy = Pi05(stop, args.adapter)
results = []
try:
    for episode in episodes:
        task, seed = episode["task"], episode["seed"]
        home = prepare(env, seed, task)
        held, frames, actions = 0, [], []
        started = time.monotonic()
        error = None
        try:
            for step in range(0, args.steps, 5):
                chunk, _ = policy.infer(env.pi05_observation(), TASKS[task], stop)
                for action in chunk[:min(5, args.steps-step)]:
                    env.step(action)
                    actions.append(action.tolist())
                    held = held + 1 if success(env, task, home) else 0
                    if held >= 10:
                        break
                if episode["index"] < 2:
                    frames.append(env.images(384)["front"])
                if held >= 10:
                    break
        except (ValueError, RuntimeError) as exc:
            error = str(exc)
        row = {"task": task, "seed": seed, "index": episode["index"], "success": held >= 10,
               "steps": len(actions), "seconds": time.monotonic() - started, "error": error,
               "close_commands": sum(a[-1] > 0 for a in actions), "final_state": env.pi05_observation()["state"].tolist()}
        results.append(row)
        (out / f"{task}-{episode['index']:02d}-actions.json").write_text(json.dumps(actions))
        if frames:
            imageio.mimsave(out / f"{task}-{episode['index']:02d}.mp4", frames, fps=4)
        summary = {task: {"success": sum(r["success"] for r in results if r["task"] == task),
                          "total": sum(r["task"] == task for r in results)} for task in TASKS}
        (out / "result.json").write_text(json.dumps({"config": vars(args), "results": results, "summary": summary}, indent=2))
        print(json.dumps(row), flush=True)
finally:
    policy.close()
    env.close()
