"""One simulation owner, bounded tool loop, and per-run audit artifacts."""
import json
import time
import threading
from pathlib import Path
from datetime import datetime, timezone
import imageio.v2 as imageio
from PIL import Image
from .scene import TabletopEnv
from .skills import Skills, SkillError
from .vlm import VLM, MODEL_ID, MODEL_REVISION


class Engine:
    def __init__(self, seed=0, load_model=True):
        self.stop = threading.Event()
        self.env = TabletopEnv(seed)
        self.vlm = VLM() if load_model else None
        self.reset(seed)

    def reset(self, seed):
        self.stop.clear()
        self.seed = int(seed)
        self.env.reset_seed(self.seed)
        self.skills = Skills(self.env, self.stop)
        self.skills.wait(15)
        self.skills.steps = 0
        return self.env.images()

    def run(self, instruction, emit, use_wrist=False):
        self.stop.clear()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        out = Path("runs") / stamp
        out.mkdir(parents=True)
        log = {"seed": self.seed, "instruction": instruction, "model": MODEL_ID,
               "revision": MODEL_REVISION, "history": [], "started": stamp,
               "cameras": ["front", "wrist"] if use_wrist else ["front"]}
        started = time.monotonic()
        frames = []
        status = []
        final = ""
        def progress(step):
            if step % 10 == 0:
                images = self.env.images(size=384)
                frames.append(images["front"])
                emit(images, "\n".join(status + ["机械臂正在操作…"]))
        self.skills.on_step = progress
        try:
            for turn in range(6):
                if self.stop.is_set():
                    raise SkillError("用户已停止")
                images = self.env.images()
                Image.fromarray(images["front"]).save(out / f"observe-{turn}.png")
                if use_wrist:
                    Image.fromarray(images["wrist"]).save(out / f"observe-{turn}-wrist.png")
                emit(images, "\n".join(status + ["模型正在看图…"]))
                response, raw = self.vlm.infer(images["front"], instruction, log["history"], self.stop,
                                               images["wrist"] if use_wrist else None)
                log.setdefault("responses", []).append({"raw": raw, "parsed": response})
                if response["type"] == "answer":
                    final = response["text"]
                    break
                if turn == 5:
                    raise SkillError("已达到本轮五个技能的上限")
                names = {"red": "红色积木", "green": "绿色积木", "blue": "蓝色积木",
                         "left": "左侧区域", "center": "中央区域", "right": "右侧区域"}
                verb = {"pick": "抓起", "place": "放置", "stack": "堆叠"}[response["skill"]]
                target = f" → {names[response['target']]}" if "target" in response else ""
                status.append(f"{verb}{names[response['object']]}{target}")
                result = self.skills.execute(response["skill"], response["object"], response.get("target"))
                log["history"].append({"call": response, "result": result})
                status.append("操作已完成")
        except (SkillError, ValueError, RuntimeError) as exc:
            final = f"已停止：{exc}"
            log["error"] = str(exc)
        finally:
            self.skills.on_step = None
            log["answer"] = final
            log["seconds"] = time.monotonic() - started
            (out / "interaction.json").write_text(json.dumps(log, ensure_ascii=False, indent=2))
            images = self.env.images()
            Image.fromarray(images["front"]).save(out / "final.png")
            if frames:
                imageio.mimsave(out / "video.mp4", frames, fps=2, macro_block_size=16)
            emit(images, "\n".join(status + [final]))
        return log
