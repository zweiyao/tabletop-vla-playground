"""Download a pinned snapshot only when the remaining files fit the disk budget."""
import json
import os
import shutil
from pathlib import Path
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_HUB_DISABLE_XET"] = "1"
from huggingface_hub import HfApi, snapshot_download
from tabletop.vlm import MODEL_ID, MODEL_REVISION

root = Path(__file__).resolve().parents[1]
target = root / "models/qwen3-vl-8b"
info = HfApi().model_info(MODEL_ID, revision=MODEL_REVISION, files_metadata=True)
allowed = [f.rfilename for f in info.siblings if f.rfilename.endswith((".json", ".safetensors", ".txt", ".jinja"))]
remaining = sum(f.size or 0 for f in info.siblings if f.rfilename in allowed
                and (not (target / f.rfilename).exists() or (target / f.rfilename).stat().st_size != f.size))
free = shutil.disk_usage(root).free
reserve = 8 * 1024**3
print(json.dumps({"model": MODEL_ID, "revision": MODEL_REVISION, "remaining_bytes": remaining,
                  "free_bytes": free, "reserve_bytes": reserve}), flush=True)
if free < remaining + reserve + 512 * 1024**2:
    raise SystemExit("Insufficient space: preserving the 8 GiB reserve; no files deleted")
snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_dir=target,
                  allow_patterns=allowed, max_workers=3)
(root / "reports/model.json").write_text(json.dumps({"model": MODEL_ID, "revision": MODEL_REVISION}, indent=2))
print("MODEL_READY", flush=True)
