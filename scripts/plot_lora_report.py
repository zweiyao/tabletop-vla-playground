"""Render the measured training trace and held-out task success counts."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path("reports/lora")
root.mkdir(parents=True, exist_ok=True)
history = json.loads(Path("adapters/tabletop-four-v1/history.json").read_text())
train = [r for r in history if "train_loss" in r]
validation = [r for r in history if "val_loss" in r]
fig, ax = plt.subplots(figsize=(8, 4.3), constrained_layout=True)
ax.plot([r["step"] for r in train], [r["train_loss"] for r in train], alpha=.6, label="Sampled training batch")
ax.plot([r["step"] for r in validation], [r["val_loss"] for r in validation], "o-", label="Fixed validation set")
ax.set(xlabel="Optimizer updates", ylabel="Weighted flow-matching MSE", yscale="log")
ax.grid(alpha=.2)
ax.legend()
fig.savefig(root / "training-curves.png", dpi=180)
plt.close(fig)
paths = [Path("runs/lora-baseline/result.json"), Path("runs/lora-finetuned/result.json")]
if all(p.exists() for p in paths):
    reports = [json.loads(p.read_text()) for p in paths]
    tasks = ["red", "green", "blue", "home"]
    if any(r["summary"][t]["total"] != 20 for r in reports for t in tasks):
        raise SystemExit("Training curve saved; task-success plot requires both complete 80-case evaluations")
    fig, ax = plt.subplots(figsize=(8, 4.3), constrained_layout=True)
    for offset, label, report in zip((-.2, .2), ("Base", "LoRA"), reports):
        summary = report["summary"]
        heights = [100 * summary[t]["success"] / max(1, summary[t]["total"]) for t in tasks]
        bars = ax.bar([i + offset for i in range(4)], heights, width=.4, label=label)
        ax.bar_label(bars, labels=[f"{summary[t]['success']}/{summary[t]['total']}" for t in tasks], padding=3)
    ax.set(xticks=range(4), xticklabels=tasks, ylabel="Task success (%)", ylim=(0, 115))
    ax.legend()
    fig.savefig(root / "task-success.png", dpi=180)
