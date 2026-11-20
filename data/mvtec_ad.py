import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


CATEGORIES = (
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut",
    "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class Sample:
    image: Path
    mask: Path | None
    category: int

    @property
    def label(self):
        return int(self.mask is not None)


def image_paths(directory):
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def read_image(path, image_size):
    with Image.open(path) as source:
        rgb = source.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
        pixels = np.array(rgb, dtype=np.uint8)
    return torch.from_numpy(pixels).permute(2, 0, 1).float().div_(255)


def read_mask(path, image_size):
    if path is None:
        return torch.zeros((1, image_size, image_size), dtype=torch.bool)
    with Image.open(path) as source:
        mask = source.convert("L").resize((image_size, image_size), Image.Resampling.NEAREST)
        pixels = np.asarray(mask) > 0
    return torch.from_numpy(pixels).unsqueeze(0)


class MVTecAD(Dataset):
    def __init__(self, root, split, image_size, categories=None):
        if split not in {"train", "test"}:
            raise ValueError(split)
        self.root = Path(root).expanduser().resolve(strict=True)
        self.image_size = image_size
        self.split = split
        self.categories = tuple(sorted(categories if categories is not None else [
            category for category in CATEGORIES if (self.root / category).is_dir()
        ]))
        if not self.categories or set(self.categories) - set(CATEGORIES) or len(set(self.categories)) != len(self.categories):
            raise ValueError(self.categories)
        self.samples = []
        for index, category in enumerate(self.categories):
            base = self.root / category
            split_root = base / split
            if split == "train":
                labels = (split_root / "good",)
            else:
                labels = sorted(path for path in split_root.iterdir() if path.is_dir())
            start = len(self.samples)
            for label in labels:
                for image in image_paths(label):
                    mask = None
                    if label.name != "good":
                        mask = base / "ground_truth" / label.name / (image.stem + "_mask.png")
                        if not mask.is_file():
                            raise FileNotFoundError(mask)
                    self.samples.append(Sample(image, mask, index))
            if len(self.samples) == start:
                raise ValueError(split_root)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        result = {"image": read_image(sample.image, self.image_size), "category": sample.category}
        if self.split == "test":
            result.update(label=sample.label, mask=read_mask(sample.mask, self.image_size), path=str(sample.image))
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--categories", choices=CATEGORIES, nargs="+")
    args = parser.parse_args()
    counts = {}
    for split in ("train", "test"):
        dataset = MVTecAD(args.root, split, args.image_size, args.categories)
        counts[split] = {
            name: {
                "normal": sum(sample.category == index and sample.label == 0 for sample in dataset.samples),
                "anomalous": sum(sample.category == index and sample.label == 1 for sample in dataset.samples),
            }
            for index, name in enumerate(dataset.categories)
        }
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
