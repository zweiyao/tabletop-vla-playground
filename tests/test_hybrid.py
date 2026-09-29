"""Hybrid gates: no network, model downloads, or physical simulation in unit tests."""
import asyncio
import json
import threading
from types import SimpleNamespace
import httpx
import numpy as np
import pytest
from tabletop.hybrid_config import HybridConfig, load_config
from tabletop.review import ReviewDecision, apply_review
from tabletop.openrouter import OpenRouterReviewer, SpendLedger
from tabletop.hybrid import HybridSession
from test_pi05 import fake_engine


def decision(identifier="p", correct=False, **updates):
    value = dict(proposal_id=identifier, previous_assessment="尚无执行", next_intent="接近积木",
                 decision="correct" if correct else "accept", reason="测试审查结果", corrections=None)
    if correct:
        value["corrections"] = [dict(translation_m=[0.005, 0., 0.], rotation_rad=[0., 0., 0.02],
                                      gripper="open") for _ in range(5)]
    return ReviewDecision.model_validate(value | updates)


def test_accept_preserves_actions_and_correction_converts_units():
    proposal = np.full((10, 7), 0.2)
    original = proposal.copy()
    np.testing.assert_array_equal(apply_review(proposal, decision(), "p", HybridConfig()), proposal[:5])
    revised = apply_review(proposal, decision(correct=True), "p", HybridConfig())
    np.testing.assert_allclose(revised[:, 0], 0.3)
    np.testing.assert_allclose(revised[:, 5], 0.24)
    np.testing.assert_array_equal(revised[:, 6], -1)
    np.testing.assert_array_equal(proposal, original)


@pytest.mark.parametrize("kind", ["stale", "count", "norm", "overflow", "nan", "accept_edits", "proposal_shape"])
def test_invalid_review_never_returns_actions(kind):
    proposal = np.zeros((10, 7))
    review = decision(correct=True)
    if kind == "stale": review.proposal_id = "old"
    if kind == "count": review.corrections.pop()
    if kind == "norm": review.corrections[0].translation_m = [0.009, 0.009, 0.]
    if kind == "overflow": proposal[0, 0] = 1
    if kind == "nan": review.corrections[0].rotation_rad = [float("nan"), 0., 0.]
    if kind == "accept_edits": review.decision = "accept"
    if kind == "proposal_shape": proposal = proposal[:5]
    with pytest.raises(ValueError):
        apply_review(proposal, review, "p", HybridConfig())


@pytest.mark.parametrize("extra", [{"done": True}, {"decision": "stop"}, {"reason": ""}])
def test_schema_rejects_extra_commands(extra):
    with pytest.raises(ValueError):
        ReviewDecision.model_validate(decision().model_dump() | extra)


def test_budget_persists_and_uncertain_charge_blocks(tmp_path):
    config, models = load_config()
    model = models[config.default_model]
    path = tmp_path / "budget.json"
    ledger = SpendLedger(path, config)
    reservation = ledger.reserve(model, 0)
    with pytest.raises(RuntimeError, match="尚未确认"):
        SpendLedger(path, config).reserve(model, 0)
    with pytest.raises(RuntimeError): ledger.settle(reservation, None, "unknown")
    ledger.settle(reservation, 0.001, "g1")
    assert json.loads(path.read_text())["spent_usd"] == 0.001
    with pytest.raises(RuntimeError, match="预算"):
        SpendLedger(path, config).reserve(model, 0.89)


def reviewer_fixture(tmp_path):
    config, models = load_config()
    return OpenRouterReviewer(config, models[config.default_model], "test prompt",
                              SpendLedger(tmp_path / "ledger.json", config), api_key="test-only")


def mocked_requests(reviewer, response=None, key_update=None):
    calls = []
    def request(method, path, stop, body=None):
        calls.append((method, path, body))
        if path == "/key":
            return {"data": dict(limit=1., limit_reset=None, usage=0., byok_usage=0.,
                                 include_byok_in_limit=True) | (key_update or {})}
        if path == "/models":
            return {"data": [{"id": reviewer.model.id, "context_length": reviewer.model.context_tokens,
                              "architecture": {"input_modalities": ["text", "image"]},
                              "supported_parameters": ["response_format", "structured_outputs"]}]}
        return response or {"id": "mock-generation", "usage": {"cost": 0.001},
                            "choices": [{"finish_reason": "stop", "message": {"content": decision().model_dump_json()}}]}
    reviewer.request = request
    return calls


def test_router_payload_and_actual_cost_accounting(tmp_path):
    reviewer = reviewer_fixture(tmp_path)
    calls = mocked_requests(reviewer)
    stop = threading.Event()
    reviewer.preflight(stop)
    result = reviewer.review({"proposal_id": "p"}, [("front", np.zeros((8, 8, 3), np.uint8))], stop)
    assert result.decision == "accept"
    payload = calls[-1][2]
    assert payload["response_format"]["type"] == reviewer.model.response_format
    assert payload["provider"]["allow_fallbacks"] is False
    assert payload["messages"][1]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert json.loads(reviewer.ledger.path.read_text())["spent_usd"] == .001


@pytest.mark.parametrize("change", [{"limit": 0}, {"limit": -1}, {"limit": "invalid"},
                                    {"byok_usage": -1}, {"usage": True}])
def test_key_gate_prevents_paid_calls(tmp_path, change):
    reviewer = reviewer_fixture(tmp_path)
    calls = mocked_requests(reviewer, key_update=change)
    with pytest.raises(RuntimeError): reviewer.preflight(threading.Event())
    assert not any(c[0] == "POST" for c in calls)


@pytest.mark.parametrize("cost", [None, "0.01", -1])
def test_missing_or_invalid_cost_blocks_future_calls(tmp_path, cost):
    reviewer = reviewer_fixture(tmp_path)
    mocked_requests(reviewer, {"usage": {"cost": cost}, "choices": []})
    stop = threading.Event()
    reviewer.preflight(stop)
    with pytest.raises(RuntimeError): reviewer.review({}, [], stop)
    assert json.loads(reviewer.ledger.path.read_text())["pending"]


@pytest.mark.parametrize("content", ["not JSON", "[]", '[{"$defs": {}}]'])
def test_bad_json_is_billed_but_never_accepted(tmp_path, content):
    reviewer = reviewer_fixture(tmp_path)
    mocked_requests(reviewer, {"usage": {"cost": .001}, "choices": [
        {"finish_reason": "stop", "message": {"content": content}}]})
    stop = threading.Event()
    reviewer.preflight(stop)
    with pytest.raises(ValueError): reviewer.review({}, [], stop)
    assert not json.loads(reviewer.ledger.path.read_text())["pending"]


@pytest.mark.parametrize("cancel", [False, True])
def test_http_timeout_and_stop_cancel_inflight_request(tmp_path, monkeypatch, cancel):
    reviewer = reviewer_fixture(tmp_path)
    reviewer.config = reviewer.config.model_copy(update={"timeout_seconds": .15})
    stop = threading.Event()
    cancelled = []
    async def handler(request):
        try:
            if cancel: stop.set()
            await asyncio.sleep(10)
        finally:
            cancelled.append(True)
    original = httpx.AsyncClient
    monkeypatch.setattr("tabletop.openrouter.httpx.AsyncClient",
                        lambda **kw: original(transport=httpx.MockTransport(handler), **kw))
    with pytest.raises(RuntimeError): reviewer.request("POST", "/chat/completions", stop, {})
    assert cancelled


class StubReviewer:
    last_response = None
    def __init__(self, fail=False, correct=False, stop_on_review=False):
        self.fail, self.correct, self.stop_on_review = fail, correct, stop_on_review
        self.contexts, self.images = [], []
    def preflight(self, stop):
        if self.fail: raise RuntimeError("preflight rejected")
    def review(self, context, images, stop):
        self.contexts.append(context)
        self.images.append(images)
        if self.stop_on_review: stop.set()
        return decision(context["proposal_id"], self.correct)


def session_engine(tmp_path, monkeypatch, reviewer, stop_at=None):
    engine = fake_engine(tmp_path, monkeypatch, stop_at)
    engine.scene_id, engine.initial_robot = "scene", {}
    engine.env.control_freq = 20
    image = np.zeros((16, 16, 3), np.uint8)
    engine.env.images = lambda *a, **kw: {"front": image, "wrist": image}
    monkeypatch.setattr("tabletop.hybrid.robot_snapshot", lambda env: {"frame": "controller_base"})
    config, models = load_config()
    session = HybridSession(config, models[config.default_model], reviewer)
    return engine, session


@pytest.mark.parametrize("correct", [False, True])
def test_engine_reviews_ten_executes_five_reobserves(tmp_path, monkeypatch, correct):
    reviewer = StubReviewer(correct=correct)
    engine, session = session_engine(tmp_path, monkeypatch, reviewer)
    log = engine.run_pi05("pick red", lambda *a: None, 10, review_session=session)
    assert log["steps"] == 10 and len(engine.inferences) == 2
    assert [len(c["proposal_normalized"]) for c in reviewer.contexts] == [10, 10]
    assert [len(i) for i in reviewer.images] == [2, 4]
    assert [len(r["executed_actions"]) for r in session.records] == [5, 5]
    assert reviewer.contexts[1]["previous_execution"]["executed_actions"] == session.records[0]["executed_actions"]
    assert all(a[6] == (-1 if correct else 0) for a in engine.executed)


@pytest.mark.parametrize("failure", ["preflight", "stop", "stale"])
def test_engine_failure_never_executes_proposal(tmp_path, monkeypatch, failure):
    reviewer = StubReviewer(fail=failure == "preflight", stop_on_review=failure == "stop")
    engine, session = session_engine(tmp_path, monkeypatch, reviewer)
    if failure == "stale":
        original = reviewer.review
        def change_scene(*args):
            engine.scene_id = "new scene"
            return original(*args)
        reviewer.review = change_scene
    log = engine.run_pi05("pick red", lambda *a: None, 10, review_session=session)
    assert not engine.executed
    assert log["termination"] in ("error", "stopped")
    if failure == "preflight": assert not engine.inferences


def test_partial_chunk_stop_audits_only_executed_actions(tmp_path, monkeypatch):
    engine, session = session_engine(tmp_path, monkeypatch, StubReviewer(), stop_at=2)
    log = engine.run_pi05("pick red", lambda *a: None, 10, review_session=session)
    assert log["steps"] == len(session.records[0]["executed_actions"]) == 2
    assert engine.pi05.alive


def test_robot_pose_uses_actual_rotated_controller_origin():
    from tabletop.hybrid import robot_snapshot
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    origin = np.array([2., 3., 4.])
    controller = SimpleNamespace(input_ref_frame="base", input_type="delta", origin_pos=origin, origin_ori=rotation)
    robot = SimpleNamespace(part_controllers={"right": controller}, eef_site_id={"right": 0})
    data = SimpleNamespace(site_xmat=np.array([rotation.flatten()]),
                           cam_xpos=np.array([origin + rotation @ [1., 0., 0.]]),
                           cam_xmat=np.array([rotation.flatten()]))
    model = SimpleNamespace(camera_name2id=lambda name: 0, cam_fovy=[45.])
    env = SimpleNamespace(robots=[robot], sim=SimpleNamespace(data=data, model=model),
        _get_observations=lambda **kw: {"robot0_eef_pos": origin + rotation @ [.1, .2, .3],
                                      "robot0_gripper_qpos": np.array([.04, -.04])})
    state = robot_snapshot(env)
    np.testing.assert_allclose(state["eef_position_m"], [.1, .2, .3])
    np.testing.assert_allclose(state["eef_rotation_matrix"], np.eye(3))
    np.testing.assert_allclose(state["cameras"]["front"]["position_base_m"], [1., 0., 0.])
    assert not any("object" in key for key in state)


def test_json_object_model_still_has_local_strict_validation(tmp_path):
    reviewer = reviewer_fixture(tmp_path)
    reviewer.model = load_config()[1]["qwen/qwen3.7-flash"]
    calls = mocked_requests(reviewer)
    stop = threading.Event()
    reviewer.preflight(stop)
    reviewer.review({}, [], stop)
    assert calls[-1][2]["response_format"] == {"type": "json_object"}


def test_missing_key_does_not_create_spend_ledger(tmp_path):
    reviewer = reviewer_fixture(tmp_path)
    reviewer._key = ""
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        reviewer.preflight(threading.Event())
    assert not reviewer.ledger.path.exists()


def test_unconfirmed_cost_metrics_do_not_mask_original_failure(tmp_path):
    config, models = load_config()
    session = HybridSession(config, models[config.default_model], StubReviewer())
    session.records = [{"response": {"usage": {"cost": "invalid"}}}]
    assert session.metrics()["unconfirmed_costs"] == 1


def test_budget_includes_long_context_tiers_and_image_fees(tmp_path):
    config, models = load_config()
    ledger = SpendLedger(tmp_path / "ledger.json", config)
    model = models["qwen/qwen3.7-flash"]
    reservation = ledger.reserve(model, 0)
    state = json.loads(ledger.path.read_text())
    assert state["pending"][reservation]["upper_bound_usd"] > .25
    ledger.settle(reservation, 0, "mock")
    gemini = models["google/gemini-3.1-flash-lite"].model_copy(update={"disabled_reason": None})
    reviewer = OpenRouterReviewer(config, gemini, "prompt", ledger, api_key="test-only")
    calls = mocked_requests(reviewer)
    reviewer.preflight(threading.Event())
    reviewer.review({}, [], threading.Event())
    assert calls[-1][2]["provider"]["max_price"]["image"] == gemini.image_price > 0


def test_account_key_limit_does_not_expand_project_budget(tmp_path):
    reviewer = reviewer_fixture(tmp_path)
    mocked_requests(reviewer, key_update={"limit": 50., "include_byok_in_limit": False})
    stop = threading.Event()
    reviewer.preflight(stop)
    reviewer.review({}, [], stop)
    assert reviewer.config.budget_usd == 1
    with pytest.raises(RuntimeError, match="预算"):
        reviewer.ledger.reserve(reviewer.model, .85)


def test_byok_upstream_charge_is_included(tmp_path):
    reviewer = reviewer_fixture(tmp_path)
    mocked_requests(reviewer, {"usage": {"cost": .00005, "is_byok": True, "cost_details": {"upstream_inference_cost": .001}},
                              "choices": [{"finish_reason": "stop", "message": {"content": decision().model_dump_json()}}]})
    stop = threading.Event()
    reviewer.preflight(stop)
    reviewer.review({}, [], stop)
    assert json.loads(reviewer.ledger.path.read_text())["spent_usd"] == pytest.approx(.00105)


def test_non_byok_upstream_breakdown_is_not_double_counted(tmp_path):
    reviewer = reviewer_fixture(tmp_path)
    mocked_requests(reviewer, {"usage": {"cost": .001, "is_byok": False, "cost_details": {"upstream_inference_cost": .001}},
                              "choices": [{"finish_reason": "stop", "message": {"content": decision().model_dump_json()}}]})
    stop = threading.Event()
    reviewer.preflight(stop)
    reviewer.review({}, [], stop)
    assert json.loads(reviewer.ledger.path.read_text())["spent_usd"] == pytest.approx(.001)


def test_disabled_model_rejected_before_network(tmp_path):
    reviewer = reviewer_fixture(tmp_path)
    reviewer.model = load_config()[1]["openai/gpt-4.1-mini"]
    reviewer.request = lambda *a, **kw: pytest.fail("disabled model must not call API")
    with pytest.raises(RuntimeError, match="HTTP 403"):
        reviewer.preflight(threading.Event())
