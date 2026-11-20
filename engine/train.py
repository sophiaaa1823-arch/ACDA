import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Sampler

from data.mvtec_ad import MVTecAD
from losses import ACDALoss
from runtime import build_model, configure_runtime, load_model_state, precision, read_checkpoint, read_config, save_checkpoint, seed_runtime


class CategoryPairSampler(Sampler):
    def __init__(self, dataset, batch_size, seed):
        if batch_size < 2 or batch_size % 2:
            raise ValueError(batch_size)
        groups = defaultdict(list)
        for index, sample in enumerate(dataset.samples):
            groups[sample.category].append(index)
        self.groups = tuple(groups.values())
        self.pairs_per_batch = batch_size // 2
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        pairs = []
        for indices in self.groups:
            permuted = [indices[index] for index in torch.randperm(len(indices), generator=generator).tolist()]
            if len(permuted) % 2:
                permuted.append(permuted[0])
            pairs.extend(zip(permuted[::2], permuted[1::2]))
        order = torch.randperm(len(pairs), generator=generator).tolist()
        for offset in range(0, len(order), self.pairs_per_batch):
            yield [index for key in order[offset:offset + self.pairs_per_batch] for index in pairs[key]]

    def __len__(self):
        pairs = sum((len(indices) + 1) // 2 for indices in self.groups)
        return (pairs + self.pairs_per_batch - 1) // self.pairs_per_batch


def train_epoch(model, criterion, loader, optimizer, scaler, config, device):
    model.train()
    totals = defaultdict(float)
    count = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        categories = batch["category"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with precision(config["runtime"], device):
            outputs = model(images, categories)
            losses = criterion(outputs, model.acst.adapter)
        if not torch.isfinite(losses["total"]).item():
            raise FloatingPointError({name: value.detach().item() for name, value in losses.items()})
        scaler.scale(losses["total"]).backward()
        scaler.step(optimizer)
        scaler.update()
        count += len(images)
        for name, value in losses.items():
            totals[name] += value.detach().item() * len(images)
    return {name: value / count for name, value in totals.items()}


def run(config, resume=None, device_override=None):
    device = configure_runtime(config["runtime"], device_override)
    seed_runtime(config["seed"], device)
    settings = config["training"]
    if settings["epochs"] < 1 or settings["checkpoint_every"] < 1:
        raise ValueError((settings["epochs"], settings["checkpoint_every"]))
    dataset = MVTecAD(split="train", **config["data"])
    sampler = CategoryPairSampler(dataset, settings["batch_size"], config["seed"])
    workers = config["runtime"]["workers"]
    loader = DataLoader(
        dataset, batch_sampler=sampler, num_workers=workers, persistent_workers=workers > 0,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(config["seed"]),
    )
    model = build_model(config, device)
    parameters = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    optimizer = torch.optim.AdamW(parameters.values(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings["epochs"])
    use_scaler = config["runtime"]["amp"] and device.type == "cuda" and not torch.cuda.is_bf16_supported()
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    criterion = ACDALoss(config["loss"])
    start = 0
    if resume:
        checkpoint = read_checkpoint(resume)
        saved = checkpoint["config"]
        if any(saved[key] != config[key] for key in ("model", "loss", "seed")):
            raise ValueError("model/loss/seed")
        if saved["data"]["image_size"] != config["data"]["image_size"] or checkpoint["categories"] != list(dataset.categories):
            raise ValueError("data")
        if any(saved["training"][key] != settings[key] for key in ("epochs", "batch_size", "learning_rate", "weight_decay")):
            raise ValueError("training")
        load_model_state(model, checkpoint)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if use_scaler and checkpoint["scaler"]:
            scaler.load_state_dict(checkpoint["scaler"])
        torch.set_rng_state(checkpoint["torch_rng"])
        if device.type == "cuda" and checkpoint["cuda_rng"] is not None:
            torch.cuda.set_rng_state(checkpoint["cuda_rng"], device)
        start = checkpoint["epoch"]
        del checkpoint

    destination = Path(settings["output_dir"]) / "last.pt"
    for epoch in range(start, settings["epochs"]):
        sampler.epoch = epoch
        metrics = train_epoch(model, criterion, loader, optimizer, scaler, config, device)
        print(json.dumps({"epoch": epoch + 1, "learning_rate": optimizer.param_groups[0]["lr"], **metrics}), flush=True)
        scheduler.step()
        if (epoch + 1) % settings["checkpoint_every"] == 0 or epoch + 1 == settings["epochs"]:
            save_checkpoint(destination, {
                "config": config,
                "categories": list(dataset.categories),
                "epoch": epoch + 1,
                "model": {name: parameter.detach() for name, parameter in parameters.items()},
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
            })
    return destination


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/mvtec_ad.yaml"))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    config = read_config(args.config)
    if args.data_root:
        config["data"]["root"] = str(args.data_root.resolve())
    if args.output_dir:
        config["training"]["output_dir"] = str(args.output_dir.resolve())
    run(config, args.resume, args.device)


if __name__ == "__main__":
    main()
