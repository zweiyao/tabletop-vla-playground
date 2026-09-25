"""Tool-loop limits independent of model weights and simulator execution."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import threading
from types import SimpleNamespace
import numpy as np
from tabletop.engine import Engine
from tabletop.skills import SkillError


def make_engine(tmp_path, monkeypatch, response, fail=False):
    monkeypatch.chdir(tmp_path)
    engine = Engine.__new__(Engine)
    engine.seed = 0
    engine.stop = threading.Event()
    engine.env = SimpleNamespace(images=lambda: {"front":np.zeros((16,16,3),dtype=np.uint8)})
    engine.calls = 0
    def execute(*args):
        engine.calls += 1
        if fail:
            raise SkillError("test physical failure")
        return {"success":True}
    engine.skills = SimpleNamespace(execute=execute, on_step=None)
    engine.vlm = SimpleNamespace(infer=lambda *args: (response, str(response)))
    return engine


def test_five_skill_limit(tmp_path, monkeypatch):
    engine = make_engine(tmp_path, monkeypatch, {"type":"skill","skill":"pick","object":"red"})
    log = engine.run("test", lambda *args: None)
    assert engine.calls == 5
    assert len(log["history"]) == 5
    assert "五个技能" in log["error"]


def test_failure_ends_loop(tmp_path, monkeypatch):
    engine = make_engine(tmp_path, monkeypatch, {"type":"skill","skill":"pick","object":"red"}, fail=True)
    log = engine.run("test", lambda *args: None)
    assert engine.calls == 1
    assert "test physical failure" in log["error"]


def test_answer_does_not_move(tmp_path, monkeypatch):
    engine = make_engine(tmp_path, monkeypatch, {"type":"answer","text":"三个积木"})
    log = engine.run("test", lambda *args: None)
    assert engine.calls == 0
    assert log["answer"] == "三个积木"
