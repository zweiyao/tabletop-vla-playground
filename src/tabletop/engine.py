"""One simulation owner, bounded tool loop, and per-run audit artifacts."""
import json
import os
import time
import threading
import uuid
from pathlib import Path
from datetime import datetime, timezone
import imageio.v2 as imageio
import numpy as np
from PIL import Image
from .scene import TabletopEnv
from .skills import Skills, SkillError
from .vlm import VLM, MODEL_ID, MODEL_REVISION


class Engine:
    def __init__(self, seed=0, load_model=True, hybrid_settings=None):
        from .hybrid_config import load_config
        self.hybrid_settings = hybrid_settings or load_config()
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
        from .hybrid import robot_snapshot
        self.scene_id = uuid.uuid4().hex
        self.initial_robot = robot_snapshot(self.env)
        return self.env.images()

    def pi05_status(self):
        if self.pi05 is None or not self.pi05.alive:
            return "π0.5 未启动"
        return "π0.5 已启动：" + ("LoRA 权重" if self.pi05.adapter else "原始权重")

    def start_pi05(self, mode):
        from .pi05 import Pi05
        if mode not in ("pi05", "pi05_lora"):
            raise ValueError("请先选择 π0.5 原始权重或 LoRA")
        adapter = str(Path(os.environ.get("TABLETOP_PI05_LORA", "adapters/tabletop-four-v1/best")).resolve()) if mode == "pi05_lora" else None
        if self.pi05 is not None and self.pi05.alive:
            if self.pi05.adapter != adapter:
                raise RuntimeError("请先关闭当前 π0.5，再启动所选权重")
            return self.pi05_status()
        if adapter and not all((Path(adapter) / name).is_file() for name in ("config.json", "adapter.safetensors")):
            raise RuntimeError("LoRA 权重尚未准备好")
        self.close_pi05()
        self.stop.clear()
        self.pi05 = Pi05(self.stop, adapter)
        return self.pi05_status()

    def close_pi05(self):
        if self.pi05 is not None:
            self.pi05.close()
            self.pi05 = None
        return self.pi05_status()

    def run(self, instruction, emit, use_wrist=False, mode="vlm", max_steps=300,
            hybrid_weight="pi05_lora", reviewer_model=None):
        if mode == "hybrid":
            from .hybrid import HybridSession
            config, models = self.hybrid_settings
            if hybrid_weight not in ("pi05", "pi05_lora"):
                raise ValueError("混合模式权重选项无效")
            model = models.get(reviewer_model or config.default_model)
            if model is None:
                raise ValueError("所选审查模型不在配置中")
            adapter = os.environ.get("TABLETOP_PI05_LORA", "adapters/tabletop-four-v1/best") if hybrid_weight == "pi05_lora" else None
            return self.run_pi05(instruction, emit, max_steps, adapter, HybridSession(config, model))
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

    def run_pi05(self, instruction, emit, max_steps=300, adapter=None, review_session=None):
        from .pi05 import MODEL_ID as PI05_ID, MODEL_REVISION as PI05_REVISION
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
            if review_session:
                if max_steps % review_session.config.execute_steps:
                    raise ValueError("混合模式步数上限必须是 5 的倍数")
                log["mode"] = "hybrid"
                log["hybrid"] = review_session.snapshot()
                log["scene_id"] = self.scene_id
                (out / "review-prompt.md").write_text(review_session.prompt)
            adapter = str(Path(adapter).resolve()) if adapter else None
            if self.pi05 is None or not self.pi05.alive:
                raise RuntimeError("π0.5 未启动，请先点击“启动 π0.5”")
            if getattr(self.pi05, "adapter", None) != adapter:
                raise RuntimeError("当前启动的权重与所选模型不同，请先关闭 π0.5，再启动所选权重")
            if adapter:
                config_path = Path(adapter) / "config.json"
                if not config_path.is_file() or not (Path(adapter) / "adapter.safetensors").is_file():
                    raise RuntimeError("LoRA 权重尚未准备好")
                log["adapter"] = {"path": adapter, "config": json.loads(config_path.read_text())}
            if review_session:
                emit(self.env.images(), "正在检查 OpenRouter 模型、密钥和费用上限…")
                review_session.preflight(self.stop)
            english = instruction
            if any("\u4e00" <= c <= "\u9fff" for c in instruction):
                emit(self.env.images(), "正在将任务翻译为 π0.5 使用的英文指令…")
                english = self.vlm.translate_instruction(instruction, self.stop)
            log["policy_instruction"] = english
            if self.stop.is_set():
                raise RuntimeError("用户已停止")
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
                if review_session:
                    emit(self.env.images(), f"VLM 正在审查未来 10 步，仿真暂不前进…\n进度：{log['steps']} / {max_steps}")
                    chunk = review_session.choose(self.env, chunk, instruction, english, self.initial_robot,
                                                  self.scene_id, log["steps"], out, self.stop)
                executed = []
                try:
                    for action in chunk[:min(5, max_steps-log["steps"])]:
                        if self.stop.is_set():
                            raise RuntimeError("用户已停止")
                        if review_session and self.scene_id != log["scene_id"]:
                            raise RuntimeError("场景已变化，拒绝执行过期动作")
                        self.env.step(action)
                        executed.append(action.tolist())
                        log["actions"].append(action.tolist())
                        log["steps"] += 1
                        if not np.isfinite(self.env.sim.data.qpos).all():
                            raise RuntimeError("仿真出现非有限状态，已停止")
                finally:
                    if review_session:
                        review_session.record_execution(executed)
                images = self.env.images(size=384)
                frames.append(images["front"])
                review_status = f"\n{review_session.status()}" if review_session else ""
                emit(images, f"π0.5 正在执行：{english}{review_status}\n进度：{log['steps']} / {max_steps} 步，可随时停止。")
            final = f"π0.5 已执行 {log['steps']} 步。请根据画面检查任务结果；执行结束不代表任务成功。"
            log["termination"] = "step_limit"
        except (ValueError, RuntimeError) as exc:
            log["error"] = str(exc)
            log["termination"] = "stopped" if self.stop.is_set() else "error"
            final = f"已停止：{exc}"
        finally:
            if review_session:
                log["hybrid_metrics"] = review_session.metrics()
                log["reviews"] = review_session.records
            log["answer"] = final
            log["seconds"] = time.monotonic() - started
            (out / "interaction.json").write_text(json.dumps(log, ensure_ascii=False, indent=2))
            images = self.env.images()
            Image.fromarray(images["front"]).save(out / "final.png")
            if frames:
                imageio.mimsave(out / "video.mp4", frames, fps=4, macro_block_size=16)
            emit(images, final)
        return log
