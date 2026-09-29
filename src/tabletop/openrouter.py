"""OpenRouter transport and persistent spend guard. Never imports the simulator."""
import asyncio
import base64
from contextlib import contextmanager, suppress
import fcntl
import io
import json
import math
import os
from pathlib import Path
import time
import uuid
import httpx
from PIL import Image
from .review import ReviewDecision

API_BASE = "https://openrouter.ai/api/v1"


class SpendLedger:
    """Reserve the full model-context upper bound; uncertain charges block more calls."""
    def __init__(self, path, config):
        self.path, self.config = Path(path), config

    @contextmanager
    def locked(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = json.loads(self.path.read_text()) if self.path.exists() else {"spent_usd": 0.0, "pending": {}, "requests": []}
            yield state
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(json.dumps(state, indent=2))
            temporary.replace(self.path)

    def reserve(self, model, key_usage):
        # A deliberately loose upper bound, not an image-token estimate.
        bound = ((model.context_tokens * model.budget_input_price + self.config.max_output_tokens * model.budget_output_price) / 1e6 + 4 * model.image_price) * 1.05
        with self.locked() as state:
            if state["pending"] or state.get("blocked"):
                raise RuntimeError("上次接口费用尚未确认，停止付费调用；请核对费用记录")
            spent = max(float(state["spent_usd"]), key_usage)
            if not math.isfinite(spent) or spent + bound > self.config.stop_spend_usd:
                raise RuntimeError("OpenRouter 预算余量不足，已停止本轮")
            reservation = uuid.uuid4().hex
            state["pending"][reservation] = {"upper_bound_usd": bound, "model": model.id, "time": time.time()}
        return reservation

    def settle(self, reservation, cost, generation_id):
        if isinstance(cost, bool) or not isinstance(cost, (int, float)) or not math.isfinite(cost) or cost < 0:
            raise RuntimeError("接口未返回有效费用，暂停后续调用以防超支")
        with self.locked() as state:
            pending = state["pending"].pop(reservation)
            state["spent_usd"] += cost
            state["requests"].append(pending | {"cost_usd": cost, "generation_id": generation_id})
            if cost > pending["upper_bound_usd"] + 1e-9:
                state["blocked"] = True
        if cost > pending["upper_bound_usd"] + 1e-9:
            raise RuntimeError("实际费用超出预留上限，已禁用后续付费调用")


class OpenRouterReviewer:
    def __init__(self, config, model, prompt, ledger, api_key=None):
        self.config, self.model, self.prompt, self.ledger = config, model, prompt, ledger
        self._key = api_key if api_key is not None else os.getenv("OPENROUTER_API_KEY", "")
        self.key_usage = 0.0
        self.ready = False
        self.last_response = None

    async def _request_async(self, method, path, stop, body=None):
        if stop.is_set():
            raise RuntimeError("用户已停止")
        if not self._key:
            raise RuntimeError("未配置 OPENROUTER_API_KEY，请在远程 openrouter_key 文件或 .env 中配置后重启服务")
        headers = {"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"}
        async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
            task = asyncio.create_task(client.request(method, API_BASE + path, headers=headers, json=body))
            try:
                async with asyncio.timeout(self.config.timeout_seconds):
                    while not task.done():
                        if stop.is_set():
                            raise RuntimeError("用户已停止")
                        await asyncio.wait({task}, timeout=0.1)
                    response = await task
                    if response.status_code != 200:
                        try:
                            error = response.json().get("error", {})
                            message = str(error.get("message", ""))
                            raw = error.get("metadata", {}).get("raw")
                            if raw:
                                message += " " + str(raw)
                        except (ValueError, AttributeError):
                            message = ""
                        message = message.replace(self._key, "[redacted]")[:400]
                        raise RuntimeError(f"OpenRouter HTTP {response.status_code}：{message}；本轮已停止，不重试或降级")
                    value = response.json()
                    if not isinstance(value, dict):
                        raise RuntimeError("OpenRouter 返回非法响应，本轮已停止")
                    if "error" in value:
                        raise RuntimeError("OpenRouter 返回接口错误，本轮已停止")
                    return value
            finally:
                if not task.done():
                    task.cancel()
                    with suppress(asyncio.CancelledError):
                        await task

    def request(self, method, path, stop, body=None):
        try:
            return asyncio.run(self._request_async(method, path, stop, body))
        except (httpx.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"OpenRouter 请求失败（{type(exc).__name__}），本轮已停止") from exc

    def check_key(self, stop):
        data = self.request("GET", "/key", stop).get("data", {})
        if not isinstance(data, dict):
            raise RuntimeError("无法确认 OpenRouter 密钥额度")
        limit, usage = data.get("limit"), data.get("usage")
        if limit is not None and (type(limit) not in (int, float) or not math.isfinite(limit) or limit <= 0):
            raise RuntimeError("OpenRouter 密钥额度无效或已禁用")
        if type(usage) not in (int, float) or not math.isfinite(usage) or usage < 0:
            raise RuntimeError("无法确认 OpenRouter 已用额度")
        byok = data.get("byok_usage", 0)
        if type(byok) not in (int, float) or not math.isfinite(byok) or byok < 0:
            raise RuntimeError("无法确认 OpenRouter BYOK 已用额度")
        self.key_usage = usage + byok

    def preflight(self, stop):
        if self.model.disabled_reason:
            raise RuntimeError(self.model.disabled_reason)
        self.check_key(stop)
        catalog = self.request("GET", "/models", stop).get("data", [])
        if not isinstance(catalog, list):
            raise RuntimeError("无法读取 OpenRouter 模型列表")
        info = next((m for m in catalog if isinstance(m, dict) and m.get("id") == self.model.id), None)
        if info is None or "image" not in info.get("architecture", {}).get("input_modalities", []):
            raise RuntimeError("所选视觉模型不可用，不自动更换模型")
        if not isinstance(info.get("context_length"), int) or info["context_length"] > self.model.context_tokens:
            raise RuntimeError("模型上下文超过配置的费用预留范围，请更新模型配置")
        if "response_format" not in info.get("supported_parameters", []):
            raise RuntimeError("所选模型不支持配置要求的 JSON 输出")
        if self.model.response_format == "json_schema" and "structured_outputs" not in info.get("supported_parameters", []):
            raise RuntimeError("所选模型不支持配置要求的严格结构化输出")
        self.ready = True

    def review(self, context, images, stop):
        if not self.ready:
            raise RuntimeError("审查接口尚未完成预检查")
        if len(images) > 4:
            raise ValueError("审查最多四张图片，超出费用预留范围")
        self.last_response = None
        self.check_key(stop)
        content = []
        for label, pixels in images:
            stream = io.BytesIO()
            Image.fromarray(pixels).save(stream, format="PNG")
            content.extend([{"type": "text", "text": label}, {"type": "image_url", "image_url": {
                "url": "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii")}}])
        content.append({"type": "text", "text": json.dumps(context, ensure_ascii=False, allow_nan=False)})
        output_format = {"type": self.model.response_format}
        if self.model.response_format == "json_schema":
            output_format["json_schema"] = {"name": "action_review", "strict": True, "schema": ReviewDecision.model_json_schema()}
        payload = {"model": self.model.id, "messages": [{"role": "system", "content": self.prompt},
                    {"role": "user", "content": content}], "response_format": output_format,
                   "temperature": self.config.temperature, "max_tokens": self.config.max_output_tokens,
                   "provider": {"require_parameters": True, "allow_fallbacks": False,
                                "max_price": {"prompt": self.model.input_price, "completion": self.model.output_price,
                                              "image": self.model.image_price, "request": 0}}}
        if self.model.reasoning != "none":
            payload["reasoning"] = {"enabled": False} if self.model.reasoning == "off" else {"effort": "minimal"}
        reservation = self.ledger.reserve(self.model, self.key_usage)
        started = time.monotonic()
        # On timeout/cancellation the pending reservation intentionally remains:
        # the provider may still bill the request. No further calls until reconciled.
        result = self.request("POST", "/chat/completions", stop, payload)
        usage = result.get("usage")
        if not isinstance(usage, dict):
            usage = {}
        self.last_response = {"id": result.get("id"), "model": result.get("model"),
                              "usage": usage, "choices": result.get("choices"),
                              "seconds": time.monotonic() - started}
        cost = usage.get("cost")
        upstream = ((usage.get("cost_details") or {}).get("upstream_inference_cost") or 0) if usage.get("is_byok") is True else 0
        if (type(cost) in (int, float) and type(upstream) in (int, float)
                and math.isfinite(cost) and math.isfinite(upstream) and cost >= 0 and upstream >= 0):
            cost += upstream
        else:
            cost = None
        self.last_response["accounted_cost_usd"] = cost
        self.ledger.settle(reservation, cost, result.get("id"))
        if stop.is_set():
            raise RuntimeError("用户已停止")
        try:
            choice = result["choices"][0]
            if choice.get("finish_reason") != "stop":
                raise ValueError("审查输出未正常结束")
            return ReviewDecision.model_validate_json(choice["message"]["content"])
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ValueError("VLM 返回非法或不完整的审查结果，未执行动作") from exc
