"""LIBERO π0.5 adapter. Actions are normalized OSC commands, not metre deltas."""
import atexit
import base64
import io
import json
import os
from pathlib import Path
import select
import subprocess
import time
import numpy as np
from PIL import Image

MODEL_ID = "RLinf/RLinf-Pi05-LIBERO-SFT"
MODEL_REVISION = "45ccfcc4e28634f1576ebf78cab0fbe2fd82432d"


def validate_actions(actions):
    values = np.asarray(actions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 7 or not 5 <= len(values) <= 50:
        raise ValueError("π0.5 must return at least five 7D actions")
    if not np.isfinite(values).all():
        raise ValueError("π0.5 returned non-finite actions")
    # Upstream LIBERO sends these directly to OSC_POSE; no metre conversion or gripper inversion.
    return np.clip(values, -1, 1)


class Pi05:
    def __init__(self, stop, adapter=None):
        root = Path(__file__).resolve().parents[2]
        legacy = Path(os.environ.get("TABLETOP_PI05_ROOT", root.parent / "simulation"))
        python = Path(os.environ.get("TABLETOP_PI05_PYTHON", legacy / ".venv/bin/python"))
        checkpoint = Path(os.environ.get("TABLETOP_PI05_CHECKPOINT", legacy / "checkpoints/RLinf-Pi05-LIBERO-SFT"))
        if not python.is_file() or not (checkpoint / "model.safetensors").is_file():
            raise RuntimeError("π0.5 环境或权重未配置，请查看 README 中的 π0.5 部署说明")
        env = os.environ.copy()
        self.adapter = str(Path(adapter).resolve()) if adapter else None
        env.pop("TABLETOP_PI05_ADAPTER", None)
        if self.adapter:
            env["TABLETOP_PI05_ADAPTER"] = self.adapter
        env.update({"TABLETOP_PI05_CHECKPOINT": str(checkpoint),
                    "PYTHONPATH": str(legacy / "repos/RLinf"),
                    "OPENPI_DATA_HOME": str(legacy / ".cache/openpi"),
                    "JAX_PLATFORMS": "cpu", "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                    "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false",
                    "TORCH_COMPILE_DISABLE": "1"})
        env.pop("MUJOCO_EGL_DEVICE_ID", None)  # The worker never renders; avoid legacy import-time namespace checks.
        (root / "runs").mkdir(exist_ok=True)
        self.log = open(root / "runs/pi05-worker.log", "a")
        self.process = subprocess.Popen([str(python), "-u", str(Path(__file__).with_name("pi05_worker.py"))],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self.log, text=True, bufsize=1, cwd=root, env=env)
        atexit.register(self.close)
        try:
            if not self._receive(stop, timeout=300).get("ready"):
                raise RuntimeError("π0.5 启动未就绪")
        except Exception:
            self.close()
            raise

    @property
    def alive(self):
        return self.process.poll() is None

    def close(self):
        if self.alive:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.log.close()

    def _receive(self, stop, timeout=60):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if stop.is_set():
                self.close()
                raise RuntimeError("用户已停止")
            ready, _, _ = select.select([self.process.stdout], [], [], 0.1)
            if ready:
                line = self.process.stdout.readline()
                if not line:
                    raise RuntimeError("π0.5 推理进程退出，请查看 runs/pi05-worker.log")
                result = json.loads(line)
                if "error" in result:
                    raise RuntimeError(result["error"])
                return result
        self.close()
        raise RuntimeError("π0.5 推理超时")

    def infer(self, observation, instruction, stop):
        def encode(image):
            stream = io.BytesIO()
            Image.fromarray(image).save(stream, format="PNG")
            return base64.b64encode(stream.getvalue()).decode("ascii")
        request = {"image": encode(observation["image"]), "wrist_image": encode(observation["wrist_image"]),
                   "state": observation["state"].tolist(), "instruction": instruction}
        try:
            self.process.stdin.write(json.dumps(request) + "\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError("π0.5 推理进程连接已断开") from exc
        result = self._receive(stop)
        return validate_actions(result["actions"]), result.get("timing", {})
