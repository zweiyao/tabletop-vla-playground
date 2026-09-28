"""Direct-action smoke evaluation; no scripted grasp skills are invoked."""
import json
import threading
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
from PIL import Image
from tabletop.runtime import configure_gpu
configure_gpu()
from tabletop.pi05 import Pi05, MODEL_ID, MODEL_REVISION
from tabletop.scene import TabletopEnv

out = Path("runs/pi05-probe")
out.mkdir(parents=True, exist_ok=True)
env = TabletopEnv()
env.reset_seed(0)
for _ in range(15):
    env.step([0]*6+[-1])
stop = threading.Event()
client = Pi05(stop)
frames, actions, timings = [], [], []
max_height = env.object_pos("red")[2]
try:
    for step in range(0, 300, 5):
        observation = env.pi05_observation()
        if step == 0:
            Image.fromarray(observation["image"]).save(out / "input-agent.png")
            Image.fromarray(observation["wrist_image"]).save(out / "input-wrist.png")
            print("STATE", observation["state"].tolist(), flush=True)
        chunk, timing = client.infer(observation, "pick up the red cube", stop)
        timings.append(timing)
        for action in chunk[:5]:
            env.step(action)
            actions.append(action.tolist())
            max_height = max(max_height, env.object_pos("red")[2])
        frames.append(env.images(384)["front"])
        if step % 25 == 0:
            print(json.dumps({"step":step+5, "action":chunk[0].tolist(), "red":env.object_pos("red").tolist(), "timing":timing}), flush=True)
    held = bool(env._check_grasp(env.robots[0].gripper["right"], env.cubes["red"]))
    result = {"model":MODEL_ID, "revision":MODEL_REVISION, "instruction":"pick up the red cube", "seed":0,
              "steps":len(actions), "max_red_height":max_height, "final_red_position":env.object_pos("red").tolist(),
              "success":bool(held and env.object_pos("red")[2] > 0.90), "actions":actions, "timings":timings}
    (out / "result.json").write_text(json.dumps(result, indent=2))
    imageio.mimsave(out / "video.mp4", frames, fps=4)
    print("RESULT", {k:v for k,v in result.items() if k not in ("actions","timings")}, flush=True)
finally:
    client.close()
    env.close()
