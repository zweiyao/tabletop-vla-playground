"""Remote acceptance tests. Seeds 0..19, each skill tested from a fresh reset."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import imageio.v2 as imageio
from tabletop.runtime import check_gpu_idle, configure_gpu
check_gpu_idle()
configure_gpu()
from tabletop.scene import TabletopEnv, COLORS
from tabletop.skills import Skills, SkillError
from tabletop.policy import Observation, HoldPolicy, to_controller_actions

out = Path("runs/physics")
out.mkdir(parents=True, exist_ok=True)
env = TabletopEnv()
results = []

def reset(seed):
    env.reset_seed(seed)
    skills = Skills(env)
    skills.wait(15)
    return skills

for seed in range(20):
    s = reset(seed)
    pos = {n: env.object_pos(n) for n in COLORS}
    images = env.images(256)
    valid = all(im.shape == (256, 256, 3) and im.std() > 10 for im in images.values())
    valid &= all(abs(p[2]-0.82) < 0.005 for p in pos.values())
    valid &= all(np.max(np.abs(pos[a][:2]-pos[b][:2])) >= 0.039 for a in pos for b in pos if a < b)
    reset(seed)
    valid &= all(np.allclose(pos[n], env.object_pos(n), atol=1e-7) for n in pos)
    results.append({"test": "environment", "seed": seed, "success": bool(valid)})
    for skill in ("pick", "place", "stack"):
        s = reset(seed)
        obj = list(COLORS)[seed % 3]
        target = None if skill == "pick" else ("left" if skill == "place" else list(COLORS)[(seed+1)%3])
        frames = []
        if seed == 0 and skill == "stack":
            s.on_step = lambda step: frames.append(env.images(384)["front"]) if step % 5 == 0 else None
        try:
            result = s.execute(skill, obj, target)
        except SkillError as exc:
            result = {"success": False, "error": str(exc), "steps": s.steps}
            Image.fromarray(env.images()["front"]).save(out / f"failure-{skill}-{seed}.png")
        result.update(test=skill, seed=seed, object=obj, target=target)
        results.append(result)
        print(json.dumps(result), flush=True)
        if frames:
            imageio.mimsave(out / "stack-demo.mp4", frames, fps=4)
    (out / "results.json").write_text(json.dumps(results, indent=2))

# Stops and timeouts must interrupt BEFORE another simulator step.
s = reset(100)
for name, setup, call in [
    ("stop", lambda: s.stop.set(), lambda: s.execute("pick", "red")),
    ("timeout", lambda: (s.stop.clear(), setattr(s, "max_steps", 0)), lambda: s.execute("pick", "red")),
    ("missing_object", lambda: setattr(s, "max_steps", 650), lambda: s.execute("pick", "yellow")),
]:
    setup()
    before = env.sim.data.time
    try:
        call()
        valid = False
    except SkillError:
        valid = env.sim.data.time == before
    results.append({"test": name, "success": bool(valid)})

s = reset(100)
obs = env._get_observations()
observation = Observation(env.images(128), obs["robot0_proprio-state"], "保持")
for action in to_controller_actions(HoldPolicy().predict(observation)):
    env.step(action)
results.append({"test": "vla_interface", "success": bool(np.isfinite(env.sim.data.qpos).all())})
summary = {name: {"passed": sum(r["success"] for r in results if r["test"]==name),
                  "total": sum(r["test"]==name for r in results)} for name in sorted({r["test"] for r in results})}
(out / "results.json").write_text(json.dumps(results, indent=2))
Path("reports/physics.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2), flush=True)
env.close()
if any(v["passed"] < (18 if k in ("pick", "place", "stack") else v["total"]) for k,v in summary.items()):
    raise SystemExit(1)
