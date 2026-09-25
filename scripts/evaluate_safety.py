"""Verify cancellation and a step budget interrupt ongoing physical motion."""
import json
from pathlib import Path
from tabletop.runtime import check_gpu_idle, configure_gpu
check_gpu_idle()
configure_gpu()
from tabletop.scene import TabletopEnv
from tabletop.skills import Skills, SkillError

env = TabletopEnv()
results = []
for name in ("mid_motion_stop", "mid_motion_timeout"):
    env.reset_seed(42)
    s = Skills(env)
    if name == "mid_motion_stop":
        s.on_step = lambda step: s.stop.set() if step == 5 else None
    else:
        s.max_steps = 5
    try:
        s.execute("pick", "red")
        success = False
    except SkillError as exc:
        success = s.steps == 5
        results.append({"test": name, "success": success, "steps": s.steps, "message": str(exc)})
    if not success:
        raise AssertionError(f"{name} did not stop at step 5")
env.close()
Path("reports/safety.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
print(json.dumps(results, ensure_ascii=False), flush=True)
