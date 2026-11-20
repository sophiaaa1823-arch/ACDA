import argparse
from itertools import groupby
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset

from data.mvtec_ad import MVTecAD
from runtime import open_model, precision, sampling_settings, write_json


def auroc(labels, scores):
    return float(roc_auc_score(labels, scores)) if labels.min() != labels.max() else None


def macro_average(results, metric):
    values = [value[metric] for value in results.values() if value[metric] is not None]
    return float(np.mean(values)) if values else None


@torch.inference_mode()
def evaluate(model, dataset, config, device):
    sampling = sampling_settings(config, device)
    results = {}
    for category_index, group in groupby(enumerate(dataset.samples), key=lambda item: item[1].category):
        subset = Subset(dataset, [index for index, _ in group])
        loader = DataLoader(
            subset, batch_size=config["evaluation"]["batch_size"], num_workers=config["runtime"]["workers"],
            pin_memory=device.type == "cuda", shuffle=False,
        )
        image_labels, image_scores, masks, maps = [], [], [], []
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            with precision(config["runtime"], device):
                prediction = model.predict(images, **sampling)
            image_labels.append(batch["label"].numpy())
            image_scores.append(prediction["image_score"].cpu().numpy())
            masks.append(batch["mask"].numpy().reshape(-1))
            maps.append(prediction["anomaly_map"].cpu().numpy().reshape(-1))
        results[dataset.categories[category_index]] = {
            "n_images": len(subset),
            "image_auroc": auroc(np.concatenate(image_labels), np.concatenate(image_scores)),
            "pixel_auroc": auroc(np.concatenate(masks), np.concatenate(maps)),
        }
    return {
        "categories": results,
        "macro": {metric: macro_average(results, metric) for metric in ("image_auroc", "pixel_auroc")},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    model, config, device = open_model(args.checkpoint, args.device)
    settings = config["data"]
    root = args.data_root or settings["root"]
    dataset = MVTecAD(root, "test", settings["image_size"], settings["categories"])
    write_json(evaluate(model, dataset, config, device), args.output)


if __name__ == "__main__":
    main()
