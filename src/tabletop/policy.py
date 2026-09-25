"""Public VLA interface. No pretrained VLA is included in v1."""
from dataclasses import dataclass
from typing import Protocol
import numpy as np


@dataclass
class Observation:
    images: dict[str, np.ndarray]  # uint8 RGB, HWC, top-left origin
    proprio: np.ndarray
    instruction: str


class Policy(Protocol):
    def predict(self, observation: Observation) -> np.ndarray:
        """Return (T,7): world-frame xyz delta (m), rotation vector (rad), gripper.

        Executed at 20 Hz. Translation limited to ±0.025 m/axis per step;
        rotation limited to ±0.25 rad/axis. Gripper -1=open, +1=close.
        Reset/re-observe between chunks; maximum chunk length 20.
        """
        ...


class HoldPolicy:
    def predict(self, observation):
        return np.array([[0, 0, 0, 0, 0, 0, -1]], dtype=float)


def to_controller_actions(chunk):
    chunk = np.asarray(chunk, dtype=float)
    if chunk.ndim != 2 or chunk.shape[1] != 7 or not 1 <= len(chunk) <= 20:
        raise ValueError("Expected a (T,7) action chunk, 1 <= T <= 20")
    limits = np.array([0.025] * 3 + [0.25] * 3 + [1.0])
    if not np.isfinite(chunk).all() or np.any(np.abs(chunk) > limits + 1e-8):
        raise ValueError("Non-finite or out-of-range action")
    return chunk / np.array([0.05] * 3 + [0.5] * 3 + [1.0])
