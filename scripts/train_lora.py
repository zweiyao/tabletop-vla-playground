"""Single-GPU supervised flow-matching LoRA with frozen visual/language prefix."""
import argparse
import io
import json
import math
import os
import random
import time
import zipfile
from pathlib import Path
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import torch
from PIL import Image
from openpi.models.model import Observation
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
from toolkits.standalone_eval_scripts import openpi as api
from rlinf.models.embodiment.openpi.dataconfig import _CONFIGS_DICT
from tabletop import lora

PROMPTS = {color: [f"pick up the {color} cube", f"Pick up the {color} block", f"grasp and lift the {color} cube"]
           for color in ("red", "green", "blue")}
PROMPTS["home"] = ["return to the initial end effector position and orientation with the gripper open",
                   "return to the initial position", "move the arm back to its starting pose"]

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="data/tabletop-four-v1")
parser.add_argument("--output", default="adapters/tabletop-four-v1")
parser.add_argument("--checkpoint", default="../simulation/checkpoints/RLinf-Pi05-LIBERO-SFT")
parser.add_argument("--steps", type=int, default=1500)
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--rank", type=int, default=16)
parser.add_argument("--alpha", type=int, default=32)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--validate-every", type=int, default=250)
parser.add_argument("--smoke", action="store_true", help="Check gradients on six early episodes, not a production training run")
args = parser.parse_args()
root, out = Path(args.data), Path(args.output)
out.mkdir(parents=True, exist_ok=True)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.set_num_threads(4)
manifest = json.loads((root / "manifest.json").read_text())
train = [e for e in manifest["episodes"] if e["split"] == "train"]
val = [e for e in manifest["episodes"] if e["split"] == "val"]
if args.smoke:
    val, train = train[4:6], train[:4]
elif len(train) != 320 or len(val) != 48:
    raise ValueError(f"Expected all 320 train / 48 validation episodes, got {len(train)}/{len(val)}")
archives, arrays = {}, {}
for e in train + val:
    archives[e["file"]] = zipfile.ZipFile(root / e["file"])
    with np.load(io.BytesIO(archives[e["file"]].read("arrays.npz"))) as data:
        arrays[e["file"]] = {k: data[k] for k in data.files}


def checked_load(config, path):
    model = api.pi0_pytorch.PI0Pytorch(config=config.model)
    missing, unexpected = api.safetensors.torch.load_model(model, path, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: {missing}, {unexpected}")
    return model


api.load_pytorch = checked_load
policy = api.create_trained_policy(_CONFIGS_DICT["pi05_libero"], args.checkpoint,
                                    sample_kwargs={"num_steps": 10}, pytorch_device="cuda:0")
model = lora.attach(policy._model, args.rank, args.alpha)
model.train()
model.paligemma_with_expert.paligemma.eval()
# The prefix is frozen: cache K/V under no_grad, then differentiate only the action expert.
model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "sdpa"
model.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"
parameters = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(parameters, lr=args.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01)
metadata = vars(args) | {"base_model": "RLinf/RLinf-Pi05-LIBERO-SFT", "base_revision": "45ccfcc4e28634f1576ebf78cab0fbe2fd82432d",
                       "trainable_parameters": sum(p.numel() for p in parameters),
                       "trainable_dtypes": sorted({str(p.dtype) for p in parameters}),
                       "total_parameters": sum(p.numel() for p in model.parameters()),
                       "target_modules": lora.TARGETS, "train_episodes": len(train), "val_episodes": len(val),
                       "train_frames": sum(e["length"] for e in train), "action_horizon": 10,
                       "optimizer": "AdamW", "betas": [0.9, 0.95], "weight_decay": 0.01,
                       "warmup_steps": 100, "grad_clip": 1.0, "gripper_loss_weight": 2.0,
                       "augmentation": "no image augmentation; task paraphrases", "language_prompts": PROMPTS,
                       "effective_batch_size": args.batch_size,
                       "prefix_frozen": True, "discrete_state_input": False, "normalization": "unchanged base checkpoint quantiles",
                       "loss": "flow MSE: first 7 dimensions only, gripper weight 2; padding ignored", "dropout": 0.0}
(out / "training_config.json").write_text(json.dumps(metadata, indent=2))
print("CONFIG", json.dumps(metadata), flush=True)


def batch(examples):
    transformed = []
    for episode, t in examples:
        archive = archives[episode["file"]]
        data = arrays[episode["file"]]
        chunk = data["actions"][np.minimum(np.arange(t, t + 10), episode["length"] - 1)]
        prompt = random.choice(PROMPTS[episode["task"]]) if episode["split"] == "train" and not args.smoke else episode["instruction"]
        obs = {"observation/state": data["states"][t], "prompt": prompt, "actions": chunk}
        for target, key in (("observation/image", "image"), ("observation/wrist_image", "wrist_image")):
            obs[target] = np.asarray(Image.open(io.BytesIO(archive.read(f"{key}/{t:04d}.jpg"))).convert("RGB"))
        transformed.append(policy._input_transform(obs))
    def stack(values):
        if isinstance(values[0], dict):
            return {k: stack([v[k] for v in values]) for k in values[0]}
        tensor = torch.from_numpy(np.stack(values))
        if tensor.is_floating_point():
            tensor = tensor.float()
        return tensor.to("cuda:0")
    values = stack(transformed)
    actions = values.pop("actions")
    return Observation.from_dict(values), actions


def loss(observation, actions):
    images, masks, tokens, token_masks, state = model._preprocess_observation(observation, train=False)
    with torch.no_grad():
        prefix, pad, att = model.embed_prefix(images, masks, tokens, token_masks)
        prefix = prefix.to(torch.bfloat16)
        att4 = model._prepare_attention_masks_4d(make_att_2d_masks(pad, att)).to(prefix.dtype)
        _, cache = model.paligemma_with_expert.forward(attention_mask=att4, position_ids=torch.cumsum(pad, dim=1) - 1,
                                                       inputs_embeds=[prefix, None], use_cache=True)
    noise = model.sample_noise(actions.shape, actions.device)
    times = model.sample_time(len(actions), actions.device)
    xt = times[:, None, None] * noise + (1 - times[:, None, None]) * actions
    prediction = model.denoise_step(state, pad, cache, xt, times)
    squared = (prediction[..., :7] - (noise - actions)[..., :7]).square()
    weights = squared.new_tensor([1, 1, 1, 1, 1, 1, 2])
    return (squared * weights).sum(-1).mean() / weights.sum()


validation = [(e, int(t)) for e in val for t in np.linspace(0, e["length"] - 1, 4)]


def validate():
    model.eval()
    values = []
    with torch.random.fork_rng(devices=[0]), torch.no_grad():
        torch.manual_seed(2026)
        for start in range(0, len(validation), args.batch_size):
            values.append(float(loss(*batch(validation[start:start + args.batch_size]))))
    model.train()
    model.paligemma_with_expert.paligemma.eval()
    return float(np.mean(values))


best = validate()
print("BASE_VAL", best, flush=True)
history = [{"step": 0, "val_loss": best}]
started = time.monotonic()
for step in range(1, args.steps + 1):
    examples = []
    for _ in range(args.batch_size):
        e = random.choice(train)
        data = arrays[e["file"]]
        # Balanced open/closed examples within pick episodes, uniform frame sampling for home.
        choices = np.flatnonzero((data["actions"][:, 6] > 0) == (random.random() < 0.5)) if e["task"] != "home" else np.arange(e["length"])
        examples.append((e, int(random.choice(choices))))
    warm = min(1.0, step / 100)
    cosine = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * max(0, step - 100) / max(1, args.steps - 100)))
    for group in optimizer.param_groups:
        group["lr"] = args.lr * warm * cosine
    optimizer.zero_grad(set_to_none=True)
    objective = loss(*batch(examples))
    if not torch.isfinite(objective):
        raise RuntimeError("Nonfinite training loss")
    objective.backward()
    grad = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
    optimizer.step()
    if step == 1 or step % 25 == 0:
        row = {"step": step, "train_loss": float(objective.detach()), "grad_norm": float(grad),
               "lr": optimizer.param_groups[0]["lr"], "seconds": time.monotonic() - started,
               "max_gpu_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3}
        history.append(row)
        print(json.dumps(row), flush=True)
    if step % args.validate_every == 0 or step == args.steps:
        metric = validate()
        record = metadata | {"step": step, "val_loss": metric}
        lora.save(model, out / "last", record)
        if metric < best:
            best = metric
            lora.save(model, out / "best", record)
        torch.save({"step": step, "optimizer": optimizer.state_dict(), "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state(), "python_rng": random.getstate(), "numpy_rng": np.random.get_state()}, out / "optimizer-last.pt")
        history.append({"step": step, "val_loss": metric, "best_val_loss": best})
        (out / "history.json").write_text(json.dumps(history, indent=2))
        print("VALIDATION", history[-1], flush=True)
(out / "complete.json").write_text(json.dumps({"steps": args.steps, "seconds": time.monotonic() - started, "best_val_loss": best}))
