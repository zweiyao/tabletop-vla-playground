"""Image-only VLM reasoning; structured output is validated before any motion."""
import json
import os
import threading
from pathlib import Path
from PIL import Image

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
MODEL_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
SYSTEM = '''你是桌面机械臂助手。只根据当前正面相机图片和用户要求回答，不虚构看不到的信息。
场景有 red(红)、green(绿)、blue(蓝)积木。左右以图片视角为准。
只输出一个 JSON 对象，不要代码围栏：
回答问题或完成任务时：{"type":"answer","text":"中文回答"}
需要动作时：{"type":"skill","skill":"pick","object":"red"}
或 {"type":"skill","skill":"place","object":"red","target":"left"}
或 {"type":"skill","skill":"stack","object":"red","target":"blue"}
pick抓起并保持；place自动抓起并放到桌面left/center/right区域；stack自动抓起并堆叠到另一积木上。
每次只提出一个动作，执行后将收到结果和新图片。已成功完成的动作不要重复。
不要为纯问答执行动作。不存在的物体或不支持的任务用中文说明。
'''


def parse_response(text):
    value = text.strip()
    if value.startswith("```") and value.endswith("```"):
        value = value.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    data = json.loads(value)
    if not isinstance(data, dict):
        raise ValueError("模型输出必须是 JSON 对象")
    if data.get("type") == "answer":
        if set(data) != {"type", "text"} or not isinstance(data["text"], str) or not data["text"].strip():
            raise ValueError("无效回答")
        return data
    if data.get("type") != "skill" or data.get("skill") not in ("pick", "place", "stack"):
        raise ValueError("未知动作类型")
    if data.get("object") not in ("red", "green", "blue"):
        raise ValueError("不存在的积木")
    fields = {"type", "skill", "object"}
    if data["skill"] != "pick":
        fields.add("target")
        choices = ("left", "center", "right") if data["skill"] == "place" else ("red", "green", "blue")
        if data.get("target") not in choices or data["target"] == data["object"]:
            raise ValueError("无效动作目标")
    if set(data) != fields:
        raise ValueError("模型输出字段不匹配")
    return data


class VLM:
    def __init__(self, model_path=None):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        self.torch = torch
        self.path = str(model_path or os.environ.get("TABLETOP_MODEL", "models/qwen3-vl-8b"))
        if not Path(self.path).is_dir():
            raise FileNotFoundError(f"模型未下载：{self.path}，请运行 scripts/download_model.py")
        self.processor = AutoProcessor.from_pretrained(self.path, local_files_only=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.path, dtype=torch.bfloat16, device_map={"": "cuda:0"},
            attn_implementation="sdpa", local_files_only=True,
        ).eval()

    def infer(self, image, instruction, history=None, stop=None):
        from transformers import StoppingCriteria, StoppingCriteriaList
        stop = stop if stop is not None else threading.Event()

        class Cancel(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return stop.is_set()

        if stop.is_set():
            raise RuntimeError("用户已停止")
        # History contains prior tool calls/results, never simulator coordinates.
        text = instruction + "\n本轮已执行记录：" + json.dumps(history or [], ensure_ascii=False)
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": [
            {"type": "image", "image": Image.fromarray(image)}, {"type": "text", "text": text},
        ]}]
        inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                     return_dict=True, return_tensors="pt").to("cuda:0")
        with self.torch.inference_mode():
            outputs = self.model.generate(**inputs, max_new_tokens=256, do_sample=False,
                                          stopping_criteria=StoppingCriteriaList([Cancel()]))
        if stop.is_set():
            raise RuntimeError("用户已停止")
        raw = self.processor.decode(outputs[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return parse_response(raw), raw
