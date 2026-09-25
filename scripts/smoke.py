from tabletop.runtime import check_gpu_idle, configure_gpu
check_gpu_idle()
configure_gpu()
from tabletop.scene import TabletopEnv
from tabletop.skills import Skills
from PIL import Image
import json
from pathlib import Path

env = TabletopEnv()
env.reset_seed(0)
s = Skills(env)
s.wait(15)
Path("runs/smoke").mkdir(parents=True, exist_ok=True)
Image.fromarray(env.images()["front"]).save("runs/smoke/before.png")
print("POSITIONS", {n:env.object_pos(n).tolist() for n in env.cubes}, "EEF", s.eef, flush=True)
print("ORIENTATION", s.orientation, flush=True)
try:
    print(s.execute("stack", "red", "blue"), flush=True)
finally:
    Image.fromarray(env.images()["front"]).save("runs/smoke/after.png")
    print("FINAL", {n:env.object_pos(n).tolist() for n in env.cubes}, "EEF", s.eef, flush=True)
    env.close()
