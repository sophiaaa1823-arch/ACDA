import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from PIL import Image

from data.mvtec_ad import MVTecAD
from runtime import build_model, read_checkpoint, read_config, seed_runtime


PROJECT = Path(__file__).resolve().parents[1]


def save_image(path, pixels):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(path)


def create_dataset(root):
    generator = np.random.default_rng(7)
    for category in ("bottle", "cable"):
        base = root / category
        for index in range(2):
            photo = generator.integers(35, 215, (37, 43, 3), dtype=np.uint8)
            save_image(base / "train" / "good" / f"{index:03d}.png", photo)
        save_image(base / "test" / "good" / "000.png", photo[:, :, 0])
        defect = photo.copy()
        defect[9:22, 11:26] = 255
        mask = np.zeros(photo.shape[:2], dtype=np.uint8)
        mask[9:22, 11:26] = 255
        save_image(base / "test" / "defect" / "000.png", defect)
        save_image(base / "ground_truth" / "defect" / "000_mask.png", mask)


def run_module(module, *args):
    completed = subprocess.run(
        [sys.executable, "-B", "-m", module, *map(str, args)], cwd=PROJECT,
        capture_output=True, text=True, encoding="utf-8",
    )
    if completed.returncode:
        raise RuntimeError(completed.stderr)
    return completed.stdout


def main():
    with TemporaryDirectory(prefix="acda-smoke-") as directory:
        root = Path(directory)
        dataset_root = root / "mvtec_ad"
        create_dataset(dataset_root)
        preprocess = json.loads(run_module("data.mvtec_ad", "--root", dataset_root, "--image-size", 32))
        assert preprocess["train"]["bottle"] == {"normal": 2, "anomalous": 0}
        assert preprocess["test"]["cable"] == {"normal": 1, "anomalous": 1}
        dataset = MVTecAD(dataset_root, "test", 32)
        assert dataset[0]["image"].shape == (3, 32, 32)
        assert dataset[0]["mask"].dtype == torch.bool and dataset[0]["mask"].any()
        assert not dataset[1]["mask"].any()

        settings = PROJECT / "config" / "smoke.yaml"
        training = json.loads(run_module(
            "engine.train", "--config", settings, "--data-root", dataset_root, "--output-dir", root / "run"
        ))
        for key in ("total", "asc", "mfc", "local", "diffusion"):
            assert np.isfinite(training[key]) and training[key] > 0
        checkpoint_path = root / "run" / "last.pt"
        checkpoint = read_checkpoint(checkpoint_path)
        config = read_config(settings)
        device = torch.device("cpu")
        torch.set_num_threads(2)
        seed_runtime(config["seed"], device)
        model = build_model(config, device)
        initial = dict(model.named_parameters())
        for prefix in ("acst.adapter.", "msadn.", "unet.fusion.", "unet.input."):
            assert any(
                key.startswith(prefix) and not torch.equal(value, initial[key].detach())
                for key, value in checkpoint["model"].items()
            ), prefix
        assert all(key in initial and initial[key].requires_grad for key in checkpoint["model"])
        model.train()
        assert not model.pyramid.training and not model.vae.training and not model.acst.vision.training
        clean = torch.stack((dataset[0]["image"], dataset[2]["image"], dataset[2]["image"]))
        original, perturbed, adversarial = model.acst.training_pair(clean, torch.tensor([0, 1, 1]))
        with torch.no_grad():
            tokens, _ = model.acst.vision(clean)
            mask = model.acst.saliency_mask(tokens, clean.shape[-2:])
        assert torch.equal(clean.masked_select(~mask), adversarial.masked_select(~mask))
        assert ((adversarial - clean).abs().flatten(1).amax(dim=1) > 0).all()
        assert torch.isfinite(original).all() and not torch.equal(original, perturbed)

        metrics_path = root / "metrics.json"
        run_module("engine.test", "--checkpoint", checkpoint_path, "--data-root", dataset_root, "--output", metrics_path)
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        assert set(metrics["categories"]) == {"bottle", "cable"}
        assert all(0 <= value <= 1 for value in metrics["macro"].values())

        target = dataset_root / "bottle" / "test"
        prediction_dir = root / "predictions"
        results = json.loads(run_module(
            "engine.infer", "--checkpoint", checkpoint_path, "--input", target, "--output", prediction_dir
        ))
        assert {item["input"] for item in results["images"]} == {"defect/000.png", "good/000.png"}
        for key in ("defect", "good"):
            expected = prediction_dir / key
            array = np.load(expected / "000.png.anomaly.npy")
            assert array.shape == (32, 32) and np.isfinite(array).all()
            for filename in ("000.png.reconstruction.png", "000.png.heatmap.png"):
                with Image.open(expected / filename) as image:
                    assert image.size == (32, 32)

        print(json.dumps({"status": "passed", "train": training, "test": metrics["macro"], "inference_images": 2}))


if __name__ == "__main__":
    main()
