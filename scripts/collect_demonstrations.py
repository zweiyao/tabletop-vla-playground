"""Collect aligned pre-action RGB/state and normalized OSC action trajectories."""
import argparse
import io
import json
import shutil
import time
import zipfile
from pathlib import Path
import numpy as np
from PIL import Image
from tabletop.runtime import check_gpu_idle, configure_gpu

parser = argparse.ArgumentParser()
parser.add_argument("--output", default="data/tabletop-four-v1")
parser.add_argument("--train", type=int, default=80)
parser.add_argument("--val", type=int, default=12)
parser.add_argument("--test", type=int, default=20)
parser.add_argument("--share-with-app", action="store_true", help="Share the inspected GPU with our existing playground")
args = parser.parse_args()
if not args.share_with_app:
    check_gpu_idle()
configure_gpu()
from tabletop.scene import TabletopEnv
from tabletop.demonstrations import TASKS, prepare, demonstrate, success

root = Path(args.output)
root.mkdir(parents=True, exist_ok=True)
manifest_path = root / "manifest.json"
manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"episodes": [], "failures": []}
done = {(x["split"], x["task"], x["index"]) for x in manifest["episodes"]}
env = TabletopEnv()
original_step = env.step
started = time.monotonic()
previous_seconds = manifest.get("seconds", 0.0)
try:
    for split, count, base in (("train", args.train, 10000), ("val", args.val, 20000), ("test", args.test, 30000)):
        for task_index, task in enumerate(TASKS):
            for index in range(count):
                if (split, task, index) in done:
                    continue
                if shutil.disk_usage(root).free < 8.5 * 1024**3:
                    raise RuntimeError("Stopping collection to preserve 8 GiB free space")
                for retry in range(10):
                    seed = base + task_index * 1000 + index * 10 + retry
                    path = root / f"{split}-{task}-{index:03d}.zip"
                    states, actions = [], []
                    try:
                        home = prepare(env, seed, task)
                        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
                            def record(action):
                                obs = env.pi05_observation()
                                for key in ("image", "wrist_image"):
                                    stream = io.BytesIO()
                                    Image.fromarray(obs[key]).save(stream, format="JPEG", quality=90)
                                    archive.writestr(f"{key}/{len(actions):04d}.jpg", stream.getvalue())
                                states.append(obs["state"])
                                actions.append(np.asarray(action, dtype=np.float32))
                                return original_step(action)
                            env.step = record
                            demonstrate(env, task, home)
                            env.step = original_step
                            if not success(env, task, home):
                                raise RuntimeError("Demonstration failed physical acceptance")
                            arrays = io.BytesIO()
                            np.savez_compressed(arrays, states=np.asarray(states), actions=np.asarray(actions))
                            archive.writestr("arrays.npz", arrays.getvalue())
                            meta = {"split": split, "task": task, "index": index, "seed": seed,
                                    "instruction": TASKS[task], "length": len(actions), "file": path.name,
                                    "home": {k: v.tolist() for k, v in home.items()}}
                            archive.writestr("meta.json", json.dumps(meta))
                        manifest["episodes"].append(meta)
                        break
                    except Exception as exc:
                        env.step = original_step
                        # Keep failed captures separate from the accepted training manifest.
                        if path.exists():
                            path.rename(path.with_suffix(f".failed-{retry}.zip"))
                        manifest["failures"].append({"split": split, "task": task, "seed": seed, "error": str(exc)})
                        print("RETRY", manifest["failures"][-1], flush=True)
                else:
                    raise RuntimeError(f"Unable to collect {split}/{task}/{index}")
                manifest["seconds"] = previous_seconds + time.monotonic() - started
                manifest_path.write_text(json.dumps(manifest, indent=2))
                print(json.dumps({"episodes": len(manifest["episodes"]), **meta, "seconds": manifest["seconds"]}), flush=True)
finally:
    env.step = original_step
    env.close()
