"""Optional, action-expert-only PEFT LoRA; shared by training and inference."""
import json
from pathlib import Path
import torch
from peft import LoraConfig, inject_adapter_in_model, get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file, save_file

TARGETS = r"(?:paligemma_with_expert\.gemma_expert\.model\.layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))|action_in_proj|action_out_proj|time_mlp_in|time_mlp_out)"


def attach(model, rank=16, alpha=32):
    model.requires_grad_(False)
    inject_adapter_in_model(LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.0,
                                       target_modules=TARGETS, bias="none"), model)
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and "lora_" not in name:
            raise RuntimeError(f"Unexpected trainable base parameter: {name}")
    return model


def save(model, directory, metadata):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu().contiguous() for k, v in get_peft_model_state_dict(model).items()}
    save_file(state, str(directory / "adapter.safetensors"))
    (directory / "config.json").write_text(json.dumps(metadata, indent=2))


def load(model, directory):
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text())
    attach(model, config["rank"], config["alpha"])
    state = load_file(str(directory / "adapter.safetensors"))
    expected = get_peft_model_state_dict(model)
    if set(state) != set(expected) or any(state[k].shape != expected[k].shape for k in state):
        raise ValueError("LoRA adapter keys or tensor shapes do not match the model")
    result = set_peft_model_state_dict(model, state)
    if result.unexpected_keys or any("lora_" in key for key in result.missing_keys):
        raise ValueError(f"LoRA loading failed: {result}")
    model.eval()
    return config
