import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data.mvtec_ad import IMAGE_EXTENSIONS, read_image
from runtime import open_model, precision, sampling_settings, write_json


def input_images(source, output):
    if source.is_file():
        return source.parent, [source]
    if not source.is_dir():
        raise FileNotFoundError(source)
    if source == output:
        raise ValueError(output)
    images = sorted(
        path for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and not path.is_relative_to(output)
    )
    if not images:
        raise ValueError(source)
    return source, images


def save_result(destination, reconstruction, anomaly_map):
    destination.parent.mkdir(parents=True, exist_ok=True)
    reconstruction = reconstruction.permute(1, 2, 0).cpu().mul(255).round().byte().numpy()
    Image.fromarray(reconstruction).save(destination.with_name(destination.name + ".reconstruction.png"))
    anomaly = anomaly_map.squeeze(0).cpu().numpy()
    np.save(destination.with_name(destination.name + ".anomaly.npy"), anomaly)
    normalized = (anomaly - anomaly.min()) / max(float(np.ptp(anomaly)), 1e-8)
    heatmap = np.stack((
        np.clip(2 * normalized, 0, 1),
        np.clip(2 * normalized - 0.5, 0, 1),
        np.clip(3 * normalized - 2, 0, 1),
    ), axis=-1)
    Image.fromarray(np.rint(255 * heatmap).astype(np.uint8)).save(
        destination.with_name(destination.name + ".heatmap.png")
    )


@torch.inference_mode()
def run(model, config, device, source, output, seed):
    parent, inputs = input_images(source, output)
    sampling = sampling_settings(config, device, seed)
    images = []
    for path in inputs:
        image = read_image(path, config["data"]["image_size"])[None].to(device)
        with precision(config["runtime"], device):
            result = model.predict(image, **sampling)
        relative = path.relative_to(parent)
        save_result(output / relative, result["reconstruction"][0], result["anomaly_map"][0])
        images.append({"input": relative.as_posix(), "image_score": result["image_score"].item()})
    return {"images": images}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    model, config, device = open_model(args.checkpoint, args.device)
    output = args.output.resolve()
    results = run(model, config, device, args.input.resolve(strict=True), output, args.seed)
    write_json(results, output / "predictions.json")


if __name__ == "__main__":
    main()
