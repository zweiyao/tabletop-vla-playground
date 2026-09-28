"""Normalized action contract, receding-horizon execution, and cancellation."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import threading
from types import SimpleNamespace
import numpy as np
import pytest
from tabletop.pi05 import validate_actions
from tabletop.engine import Engine


def test_normalized_actions_preserve_scale_and_gripper():
    actions = np.tile([0.2, -0.4, 0.1, 0, 0, 0, -1], (10, 1))
    np.testing.assert_array_equal(validate_actions(actions), actions)
    actions[0, 0] = 1.2
    assert validate_actions(actions)[0, 0] == 1


@pytest.mark.parametrize("actions", [np.zeros((4, 7)), np.zeros((10, 8)),
                                     np.full((10, 7), np.nan)])
def test_invalid_actions_rejected(actions):
    with pytest.raises(ValueError):
        validate_actions(actions)


def fake_engine(tmp_path, monkeypatch, stop_at=None):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("tabletop.engine.imageio.mimsave", lambda *a, **kw: None)
    engine = Engine.__new__(Engine)
    engine.seed = 0
    engine.stop = threading.Event()
    engine.executed = []
    engine.inferences = []
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    def step(action):
        engine.executed.append(action)
        if len(engine.executed) == stop_at:
            engine.stop.set()
    engine.env = SimpleNamespace(
        images=lambda **kw: {"front": image}, step=step,
        pi05_observation=lambda: {"image": image, "wrist_image": image, "state": np.zeros(8)},
        sim=SimpleNamespace(data=SimpleNamespace(qpos=np.zeros(7))))
    def infer(observation, instruction, stop):
        engine.inferences.append(instruction)
        return np.zeros((10, 7)), {}
    engine.pi05 = SimpleNamespace(alive=True, infer=infer)
    engine.vlm = SimpleNamespace(translate_instruction=lambda text, stop: "pick up the red cube")
    # Deliberately no skills object: direct control must not call scripted skills.
    return engine


def test_pi05_reobserves_and_obeys_step_limit(tmp_path, monkeypatch):
    engine = fake_engine(tmp_path, monkeypatch)
    log = engine.run("抓起红色积木", lambda *a: None, mode="pi05", max_steps=12)
    assert len(engine.executed) == 12
    assert engine.inferences == ["pick up the red cube"] * 3
    assert log["termination"] == "step_limit"
    assert "success" not in log


def test_pi05_stop_discards_remaining_chunk(tmp_path, monkeypatch):
    engine = fake_engine(tmp_path, monkeypatch, stop_at=2)
    log = engine.run("pick up the red cube", lambda *a: None, mode="pi05", max_steps=12)
    assert len(engine.executed) == 2
    assert len(engine.inferences) == 1
    assert log["termination"] == "stopped"


def test_lora_mode_records_selected_adapter(tmp_path, monkeypatch):
    engine = fake_engine(tmp_path, monkeypatch)
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "config.json").write_text('{"step": 250}')
    (adapter / "adapter.safetensors").write_bytes(b"mock weights; model is replaced in this test")
    engine.pi05.adapter = str(adapter)
    monkeypatch.setenv("TABLETOP_PI05_LORA", str(adapter))
    log = engine.run("pick up the red cube", lambda *a: None, mode="pi05_lora", max_steps=5)
    assert log["mode"] == "pi05_lora"
    assert log["adapter"]["config"]["step"] == 250
    assert len(engine.executed) == 5


def test_missing_lora_does_not_fall_back_to_base(tmp_path, monkeypatch):
    engine = fake_engine(tmp_path, monkeypatch)
    monkeypatch.setenv("TABLETOP_PI05_LORA", str(tmp_path / "missing"))
    log = engine.run("pick up the red cube", lambda *a: None, mode="pi05_lora", max_steps=5)
    assert log["termination"] == "error"
    assert not engine.executed and not engine.inferences
