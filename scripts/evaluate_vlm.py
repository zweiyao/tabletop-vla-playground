"""30 image questions and 20 closed-loop Chinese commands on fixed test seeds."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from tabletop.runtime import check_gpu_idle, configure_gpu
check_gpu_idle()
configure_gpu()
from tabletop.engine import Engine
from tabletop.scene import REGIONS

out = Path("runs/vlm-evaluation")
out.mkdir(parents=True, exist_ok=True)
engine = Engine()
results = []

# Pixel locations determine the image-left/right ground truth; not world axes.
def red_left_of_blue(image):
    r,g,b = np.moveaxis(image.astype(float), -1, 0)
    red = (r > 90) & (r > 1.8*g) & (r > 1.8*b)
    blue = (b > 70) & (b > 1.5*r) & (b > 1.5*g)
    if red.sum() < 20 or blue.sum() < 20:
        raise RuntimeError("Insufficient visible pixels for QA reference")
    return np.nonzero(red)[1].mean() < np.nonzero(blue)[1].mean()

for seed in range(100, 110):
    images = engine.reset(seed)
    front = images["front"]
    relation = "左" if red_left_of_blue(front) else "右"
    cases = [
        ("桌上有哪些颜色的积木？只回答颜色名称。", "colors", ["红", "绿", "蓝"]),
        ("桌面共有几个积木？只回答阿拉伯数字。", "count", "3"),
        ("按当前正面图片，红色积木在蓝色积木的左边还是右边？只回答左或右。", "relation", relation),
    ]
    for question, category, expected in cases:
        try:
            response, raw = engine.vlm.infer(front, question)
            text = response.get("text", "")
            success = response["type"] == "answer" and (
                all(c in text for c in expected) if category == "colors"
                else (text.strip("。 .！!") == expected))
            result = {"success": bool(success), "raw": raw}
        except (ValueError, RuntimeError) as exc:
            result = {"success": False, "error": str(exc)}
        result.update(test="qa", category=category, seed=seed, question=question, expected=expected)
        results.append(result)
        if not result["success"]:
            Image.fromarray(front).save(out / f"qa-{seed}-{category}.png")
        print(json.dumps(result, ensure_ascii=False), flush=True)
        (out / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))

commands = [
    ("抓起红色积木", "pick", "red", None),
    ("请把绿色方块拿起来", "pick", "green", None),
    ("拿起蓝色的那块积木并保持", "pick", "blue", None),
    ("帮我抓住红色方块并抬高", "pick", "red", None),
    ("把绿色积木举起来", "pick", "green", None),
    ("请抓起蓝色积木", "pick", "blue", None),
    ("把红色积木放到左侧区域", "place", "red", "left"),
    ("请把绿色方块放到右侧区域", "place", "green", "right"),
    ("把蓝色积木移到左边的放置区", "place", "blue", "left"),
    ("将红色方块摆到右侧区域", "place", "red", "right"),
    ("请将绿色积木搬到左侧区域", "place", "green", "left"),
    ("把蓝色方块放到右边区域", "place", "blue", "right"),
    ("把红色积木叠在蓝色积木上", "stack", "red", "blue"),
    ("请把绿色方块放在红色方块上面", "stack", "green", "red"),
    ("将蓝色积木堆到绿色积木上", "stack", "blue", "green"),
    ("红色积木放上面，绿色积木在下面，叠起来", "stack", "red", "green"),
    ("把绿色积木叠在蓝色积木顶上", "stack", "green", "blue"),
    ("让蓝色方块站在红色方块上面", "stack", "blue", "red"),
    ("把红色方块放到蓝色方块顶部", "stack", "red", "blue"),
    ("请将绿色积木堆在红色积木上方", "stack", "green", "red"),
]
for index, (instruction, skill, obj, target) in enumerate(commands):
    seed = 200 + index
    engine.reset(seed)
    log = engine.run(instruction, lambda images, status: None)
    responses = log.get("responses", [])
    call = responses[0]["parsed"] if responses else {}
    parsed = call.get("skill") == skill and call.get("object") == obj and call.get("target") == target
    pos = engine.env.object_pos(obj)
    if skill == "pick":
        physical = engine.skills.is_grasping(obj) and pos[2] > 0.9
    else:
        xyz = np.r_[REGIONS[target], 0.82] if skill == "place" else engine.env.object_pos(target)+[0,0,0.04]
        physical = (np.linalg.norm(pos[:2]-xyz[:2]) < 0.018 and abs(pos[2]-xyz[2]) < 0.008
                    and engine.env.object_speed(obj) < 0.02 and not engine.skills.is_grasping(obj))
    result = {"test": "command", "seed": seed, "instruction": instruction,
              "expected": {"skill":skill,"object":obj,"target":target},
              "parse_success": bool(parsed), "physical_success": bool(physical),
              "success": bool(physical and "error" not in log), "interaction": log}
    results.append(result)
    if not result["success"]:
        Image.fromarray(engine.env.images()["front"]).save(out / f"command-{seed}.png")
    print(json.dumps({k:v for k,v in result.items() if k!="interaction"}, ensure_ascii=False), flush=True)
    (out / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))

summary = {
    "qa": {"passed":sum(r["success"] for r in results if r["test"]=="qa"), "total":30},
    "command_parse": {"passed":sum(r["parse_success"] for r in results if r["test"]=="command"), "total":20},
    "command_physics": {"passed":sum(r["physical_success"] for r in results if r["test"]=="command"), "total":20},
    "command_complete": {"passed":sum(r["success"] for r in results if r["test"]=="command"), "total":20},
}
Path("reports/vlm.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2), flush=True)
engine.env.close()
if summary["command_complete"]["passed"] < 18:
    raise SystemExit(1)
