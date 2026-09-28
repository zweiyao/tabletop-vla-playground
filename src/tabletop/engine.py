"""One simulation owner, bounded tool loop, and per-run audit artifacts."""
import json
import os
import time
import threading
from pathlib import Path
from datetime import datetime, timezone
import imageio.v2 as imageio
import numpy as np
from PIL import Image
from .scene import TabletopEnv
from .skills import Skills, SkillError
from .vlm import VLM, MODEL_ID, MODEL_REVISION


class Engine:
    def __init__(self, seed=0, load_model=True):
        self.stop = threading.Event()
        self.env = TabletopEnv(seed)
        self.vlm = VLM() if load_model else None
        self.pi05 = None
        self.reset(seed)

    def reset(self, seed):
        self.stop.clear()
        self.seed = int(seed)
        self.env.reset_seed(self.seed)
        self.skills = Skills(self.env, self.stop)
        self.skills.wait(15)
        self.skills.steps = 0
        return self.env.images()

    def run(self, instruction, emit, use_wrist=False, mode="vlm", max_steps=300):
        if mode in ("pi05", "pi05_lora"):
            adapter = os.environ.get("TABLETOP_PI05_LORA", "adapters/tabletop-four-v1/best") if mode == "pi05_lora" else None
            return self.run_pi05(instruction, emit, max_steps, adapter)
        if mode != "vlm":
            raise ValueError("未知模型选项")
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

    def run_pi05(self, instruction, emit, max_steps=300, adapter=None):
        from .pi05 import Pi05, MODEL_ID as PI05_ID, MODEL_REVISION as PI05_REVISION
        max_steps = int(max_steps)
        if not 5 <= max_steps <= 1000:
            raise ValueError("π0.5 执行步数必须在 5–1000 之间")
        self.stop.clear()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        out = Path("runs") / stamp
        out.mkdir(parents=True)
        log = {"model": PI05_ID, "revision": PI05_REVISION, "mode": "pi05_lora" if adapter else "pi05", "seed": self.seed,
               "instruction": instruction, "cameras": ["agentview", "wrist"], "started": stamp,
               "max_steps": max_steps, "steps": 0, "actions": [], "timings": [],
               "action_format": "LIBERO normalized OSC delta xyz/rotvec/gripper, clipped [-1,1]"}
        frames = []
        started = time.monotonic()
        final = ""
        try:
            if adapter:
                adapter = str(Path(adapter).resolve())
                config_path = Path(adapter) / "config.json"
                if not config_path.is_file() or not (Path(adapter) / "adapter.safetensors").is_file():
                    raise RuntimeError("LoRA 权重尚未准备好")
                log["adapter"] = {"path": adapter, "config": json.loads(config_path.read_text())}
            english = instruction
            if any("\u4e00" <= c <= "\u9fff" for c in instruction):
                emit(self.env.images(), "正在将任务翻译为 π0.5 使用的英文指令…")
                english = self.vlm.translate_instruction(instruction, self.stop)
            log["policy_instruction"] = english
            if self.stop.is_set():
                raise RuntimeError("用户已停止")
            if self.pi05 is not None and getattr(self.pi05, "adapter", None) != adapter:
                self.pi05.close()
                self.pi05 = None
            if self.pi05 is None or not self.pi05.alive:
                emit(self.env.images(), "正在加载 π0.5，首次使用需要稍等…")
                self.pi05 = Pi05(self.stop, adapter)
            while log["steps"] < max_steps:
                if self.stop.is_set():
                    raise RuntimeError("用户已停止")
                observation = self.env.pi05_observation()
                if log["steps"] == 0:
                    Image.fromarray(observation["image"]).save(out / "input-agent.png")
                    Image.fromarray(observation["wrist_image"]).save(out / "input-wrist.png")
                    log["initial_state"] = observation["state"].tolist()
                chunk, timing = self.pi05.infer(observation, english, self.stop)
                log["timings"].append(timing)
                for action in chunk[:min(5, max_steps-log["steps"])]:
                    if self.stop.is_set():
                        raise RuntimeError("用户已停止")
                    self.env.step(action)
                    log["actions"].append(action.tolist())
                    log["steps"] += 1
                    if not np.isfinite(self.env.sim.data.qpos).all():
                        raise RuntimeError("仿真出现非有限状态，已停止")
                images = self.env.images(size=384)
                frames.append(images["front"])
                emit(images, f"π0.5 正在执行：{english}\n进度：{log['steps']} / {max_steps} 步，可随时停止。")
            final = f"π0.5 已执行 {log['steps']} 步。请根据画面检查任务结果；执行结束不代表任务成功。"
            log["termination"] = "step_limit"
        except (ValueError, RuntimeError) as exc:
            log["error"] = str(exc)
            log["termination"] = "stopped" if self.stop.is_set() else "error"
            final = f"已停止：{exc}"
        finally:
            log["answer"] = final
            log["seconds"] = time.monotonic() - started
            (out / "interaction.json").write_text(json.dumps(log, ensure_ascii=False, indent=2))
            images = self.env.images()
            Image.fromarray(images["front"]).save(out / "final.png")
            if frames:
                imageio.mimsave(out / "video.mp4", frames, fps=4, macro_block_size=16)
            emit(images, final)
        return log
