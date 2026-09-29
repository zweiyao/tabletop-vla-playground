"""Paid reviewer compatibility probe using a saved real simulation observation."""
import argparse
import json
from pathlib import Path
import threading
from datetime import datetime, timezone
import numpy as np
from PIL import Image
from tabletop.hybrid_config import ROOT, load_config
from tabletop.openrouter import OpenRouterReviewer, SpendLedger
from tabletop.review import apply_review


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("review_directory", type=Path, help="Saved review-XXXX directory")
    parser.add_argument("--model", action="append", help="Defaults to all configured reviewers")
    parser.add_argument("--wrong-gripper", action="store_true", help="Inject a labeled gripper-only contradiction; no simulator action")
    args = parser.parse_args()
    config, models = load_config()
    saved = json.loads((args.review_directory / "review.json").read_text())
    context = saved["context"]
    if args.wrong_gripper:
        context["task"] = "保持当前末端位置和姿态，把空夹爪张开并保持张开。"
        context["policy_task"] = "keep the current end effector pose; open the empty gripper and keep it open"
        context["proposal_normalized"] = [[0., 0., 0., 0., 0., 0., 1.] for _ in range(10)]
        context["proposal_metric"] = context["proposal_normalized"]
        context["previous_execution"] = None
    context["proposal_gripper_labels"] = ["CLOSE" if a[6] > 0 else "OPEN" if a[6] < 0 else "HOLD" for a in context["proposal_normalized"]]
    images = [(f"当前观察 / {name}", np.asarray(Image.open(args.review_directory / f"{name}.png").convert("RGB")))
              for name in ("front", "wrist")]
    key = (ROOT / "openrouter_key").read_text().strip()
    output = ROOT / "runs" / ("openrouter-probe-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    output.mkdir(exist_ok=True)
    for identifier in args.model or [m.id for m in models.values() if not m.disabled_reason]:
        reviewer = OpenRouterReviewer(config, models[identifier], (ROOT / config.prompt_path).read_text(),
                                      SpendLedger(ROOT / config.ledger_path, config), api_key=key)
        result = {"model": identifier, "source": str(args.review_directory), "paid": True,
                  "case": "synthetic closing command contradicts open instruction" if args.wrong_gripper else "saved real pi0.5 proposal"}
        try:
            reviewer.preflight(threading.Event())
            decision = reviewer.review(context, images, threading.Event())
            selected = apply_review(context["proposal_normalized"], decision, context["proposal_id"], config)
            result.update(passed=True, decision=decision.model_dump(), selected_actions=selected.tolist())
            if args.wrong_gripper:
                result["passed"] = bool(decision.decision == "correct" and np.all(selected[:, 6] == -1))
        except (ValueError, RuntimeError) as exc:
            result.update(passed=False, error=str(exc))
        result["response"] = reviewer.last_response
        (output / (identifier.replace("/", "--") + ".json")).write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(json.dumps({k:v for k,v in result.items() if k not in ("selected_actions", "response")}, ensure_ascii=False), flush=True)
        # An unresolved charge intentionally blocks following models via the same ledger.
    print("Results:", output, flush=True)


if __name__ == "__main__":
    main()
