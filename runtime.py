import json
import os
from contextlib import nullcontext
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch
import yaml

from models import ACDA


def read_config(path):
    with Path(path).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def configure_runtime(settings, override=None):
    requested = override or settings["device"]
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type not in {"cuda", "cpu"}:
        raise ValueError(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(requested)
        torch.cuda.set_device(device if device.index is not None else torch.cuda.current_device())
    if device.type == "cpu" and settings["cpu_threads"]:
        torch.set_num_threads(settings["cpu_threads"])
    return device


def precision(settings, device):
    if settings["amp"] and device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.autocast("cuda", dtype=dtype)
    return nullcontext()


def seed_runtime(seed, device):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)


def build_model(config, device):
    return ACDA(config["model"], config["data"]["image_size"]).to(device)


def read_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def load_model_state(model, checkpoint):
    expected = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    weights = checkpoint["model"]
    if set(weights) != expected:
        raise ValueError({"missing": sorted(expected - weights.keys()), "unexpected": sorted(weights.keys() - expected)})
    model.load_state_dict(weights, strict=False)


def open_model(checkpoint_path, device_override):
    checkpoint = read_checkpoint(checkpoint_path)
    config = checkpoint["config"]
    config["data"]["categories"] = checkpoint["categories"]
    device = configure_runtime(config["runtime"], device_override)
    model = build_model(config, device)
    load_model_state(model, checkpoint)
    return model.eval(), config, device


def save_checkpoint(path, checkpoint):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=".checkpoint-", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(checkpoint, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(payload, path=None):
    encoded = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
    if path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


def sampling_settings(config, device, seed=None):
    settings = config["evaluation"]
    generator = torch.Generator(device=device).manual_seed(config["seed"] if seed is None else seed)
    return {"steps": settings["ddim_steps"], "start_step": settings["start_step"], "generator": generator}
