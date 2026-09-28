"""Check splits, paired data, finite control labels, and content hashes before training."""
import argparse
import collections
import hashlib
import io
import json
import zipfile
from pathlib import Path
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="data/tabletop-four-v1")
args = parser.parse_args()
root = Path(args.data)
manifest = json.loads((root / "manifest.json").read_text())
counts = collections.Counter()
frames = collections.Counter()
seeds = set()
hashes = {}
for episode in manifest["episodes"]:
    seed = episode["seed"]
    assert seed not in seeds, f"Repeated seed {seed}"
    seeds.add(seed)
    key = f"{episode['split']}/{episode['task']}"
    counts[key] += 1
    frames[key] += episode["length"]
    path = root / episode["file"]
    hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
        with np.load(io.BytesIO(archive.read("arrays.npz"))) as data:
            assert data["states"].shape == (episode["length"], 8)
            assert data["actions"].shape == (episode["length"], 7)
            assert np.isfinite(data["states"]).all() and np.isfinite(data["actions"]).all()
            assert np.abs(data["actions"]).max() <= 1
            assert set(np.unique(data["actions"][:, -1])).issubset({-1, 1})
        for frame in range(episode["length"]):
            for camera in ("image", "wrist_image"):
                with Image.open(io.BytesIO(archive.read(f"{camera}/{frame:04d}.jpg"))) as image:
                    assert image.size == (256, 256) and image.mode == "RGB"
                    image.verify()
for split, expected in (("train", 80), ("val", 12), ("test", 20)):
    for task in ("red", "green", "blue", "home"):
        assert counts[f"{split}/{task}"] == expected
report = {"episodes": dict(counts), "frames": dict(frames), "total_frames": sum(frames.values()),
          "failed_attempts": len(manifest["failures"]), "collection_seconds": manifest["seconds"],
          "unique_seeds": len(seeds), "sha256": hashes}
(root / "audit.json").write_text(json.dumps(report, indent=2))
print(json.dumps({k: v for k, v in report.items() if k != "sha256"}, indent=2))
