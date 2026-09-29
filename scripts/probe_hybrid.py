"""Real Panda + real pi0.5, deterministic mock review. No paid API calls."""
import json
from pathlib import Path
import numpy as np
from tabletop.runtime import check_gpu_idle, configure_gpu


def main():
    check_gpu_idle()
    configure_gpu()
    from tabletop.engine import Engine
    from tabletop.hybrid import HybridSession, robot_snapshot
    from tabletop.review import ReviewDecision

    class MockReviewer:
        last_response = None
        def __init__(self):
            self.calls = 0
        def preflight(self, stop):
            pass
        def review(self, context, images, stop):
            self.calls += 1
            # In the second chunk move each saturated XYZ command slightly toward zero.
            # This verifies correction plumbing; it does NOT simulate visual judgment.
            corrections = None
            if self.calls == 2:
                corrections = [{"translation_m": (-np.asarray(a[:3]) * .001).tolist(),
                                "rotation_rad": [0., 0., 0.], "gripper": "open"}
                               for a in context["proposal_normalized"][:5]]
            return ReviewDecision(proposal_id=context["proposal_id"], previous_assessment="mock only",
                                  next_intent="mock transport verification", reason="非 VLM 判断",
                                  decision="correct" if corrections else "accept", corrections=corrections)

    out = Path("runs/hybrid-smoke")
    out.mkdir(parents=True, exist_ok=True)
    engine = Engine(load_model=False)
    config, models = engine.hybrid_settings
    session = HybridSession(config, models[config.default_model], MockReviewer())
    try:
        snapshot = robot_snapshot(engine.env)
        assert snapshot["frame"] == "controller_base"
        assert set(snapshot["cameras"]) == {"front", "wrist"}
        engine.start_pi05("pi05_lora")
        adapter = engine.pi05.adapter
        log = engine.run_pi05("pick up the red cube", lambda *a: None, 10, adapter, session)
        assert log["termination"] == "step_limit", log.get("error")
        assert log["steps"] == 10 and len(session.records) == 2
        first, second = session.records
        np.testing.assert_allclose(first["executed_actions"], first["context"]["proposal_normalized"][:5])
        assert all(a[6] == -1 for a in second["executed_actions"])
        assert len(first["executed_actions"]) == len(second["executed_actions"]) == 5
        np.testing.assert_allclose(np.asarray(second["executed_actions"])[:, :3],
                                   np.asarray(second["context"]["proposal_normalized"])[:5, :3] * .98)
        result = {"passed": True, "reviewer": "deterministic mock, NOT real VLM",
                  "pi05": "real four-task LoRA", "physics": "real unchanged Panda scene",
                  "steps": log["steps"], "review_chunks": 2, "api_cost_usd": 0,
                  "initial_robot": snapshot, "run_started": log["started"],
                  "accept_and_correction_verified": True, "task_success_evaluated": False}
        (out / "result.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
    finally:
        engine.close_pi05()
        engine.env.close()


if __name__ == "__main__":
    main()
