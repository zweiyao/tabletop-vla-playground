"""Isolated openpi inference process. stdout is a JSON-line protocol only."""
import base64
import contextlib
import io
import json
import os
import sys
from pathlib import Path


def main():
    wire = sys.stdout
    with contextlib.redirect_stdout(sys.stderr):
        import numpy as np
        import torch
        from PIL import Image
        from toolkits.standalone_eval_scripts import openpi as api
        from rlinf.models.embodiment.openpi.dataconfig import _CONFIGS_DICT

        torch.set_num_threads(4)
        torch.manual_seed(7)
        def checked_load(config, path):
            model = api.pi0_pytorch.PI0Pytorch(config=config.model)
            missing, unexpected = api.safetensors.torch.load_model(model, path, strict=False)
            if missing or unexpected:
                raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
            return model
        api.load_pytorch = checked_load
        checkpoint = Path(os.environ["TABLETOP_PI05_CHECKPOINT"])
        policy = api.create_trained_policy(
            _CONFIGS_DICT["pi05_libero"], checkpoint,
            sample_kwargs={"num_steps": 10}, pytorch_device="cuda:0",
        )
        if adapter := os.environ.get("TABLETOP_PI05_ADAPTER"):
            from lora import load
            adapter_config = load(policy._model, adapter)
            print(f"Loaded LoRA: {adapter}; training step={adapter_config['step']}", file=sys.stderr)
    wire.write(json.dumps({"ready": True, "model": "RLinf/RLinf-Pi05-LIBERO-SFT"}) + "\n")
    wire.flush()
    for line in sys.stdin:
        try:
            request = json.loads(line)
            def decode(name):
                return np.array(Image.open(io.BytesIO(base64.b64decode(request[name]))).convert("RGB"))
            observation = {
                "observation/image": decode("image"),
                "observation/wrist_image": decode("wrist_image"),
                "observation/state": np.asarray(request["state"], dtype=np.float32),
                "prompt": request["instruction"],
            }
            with contextlib.redirect_stdout(sys.stderr), torch.inference_mode():
                result = policy.infer(observation)
            response = {"actions": np.asarray(result["actions"]).tolist(), "timing": result.get("policy_timing", {})}
        except Exception as exc:
            response = {"error": f"{type(exc).__name__}: {exc}"}
        wire.write(json.dumps(response, allow_nan=False) + "\n")
        wire.flush()


if __name__ == "__main__":
    main()
