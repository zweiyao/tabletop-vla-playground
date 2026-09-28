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
    engine.pi05 = SimpleNamespace(alive=True, adapter=None, infer=infer)
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


@pytest.mark.parametrize("alive", [None, False])
def test_send_never_starts_pi05(tmp_path, monkeypatch, alive):
    engine = fake_engine(tmp_path, monkeypatch)
    engine.pi05 = None if alive is None else SimpleNamespace(alive=False)
    def unexpected(*args):
        pytest.fail("Sending must not load or translate before a model is started")
    monkeypatch.setattr("tabletop.pi05.Pi05", unexpected)
    engine.vlm.translate_instruction = unexpected
    log = engine.run("抓红块", lambda *a: None, mode="pi05", max_steps=5)
    assert "未启动" in log["error"]
    assert not engine.executed


def test_send_different_weights_does_not_reload(tmp_path, monkeypatch):
    engine = fake_engine(tmp_path, monkeypatch)
    original = engine.pi05
    log = engine.run("pick red", lambda *a: None, mode="pi05_lora", max_steps=5)
    assert "权重与所选模型不同" in log["error"]
    assert engine.pi05 is original
    assert not engine.executed


def test_explicit_start_is_idempotent_and_close_unloads(tmp_path, monkeypatch):
    engine = fake_engine(tmp_path, monkeypatch)
    engine.pi05 = None
    created = []
    def create(stop, adapter):
        policy = SimpleNamespace(alive=True, adapter=adapter)
        policy.close = lambda: setattr(policy, "alive", False)
        created.append(policy)
        return policy
    monkeypatch.setattr("tabletop.pi05.Pi05", create)
    assert "原始权重" in engine.start_pi05("pi05")
    engine.start_pi05("pi05")
    assert len(created) == 1
    assert "未启动" in engine.close_pi05()
    assert engine.pi05 is None and not created[0].alive


def test_stop_drains_response_without_unloading(monkeypatch):
    import io
    from tabletop.pi05 import Pi05
    policy = Pi05.__new__(Pi05)
    output = io.StringIO('{"actions": []}\n{"actions": [[1]]}\n')
    policy.process = SimpleNamespace(stdout=output)
    policy.close = lambda: pytest.fail("Stopping an instruction must retain the model")
    monkeypatch.setattr("tabletop.pi05.select.select", lambda *a: ([output], [], []))
    stop = threading.Event()
    stop.set()
    with pytest.raises(RuntimeError, match="用户已停止"):
        policy._receive(stop, close_on_stop=False)
    stop.clear()
    assert policy._receive(stop, close_on_stop=False)["actions"] == [[1]]
