# ACDA

Adversarial-Consistent Diffusion Architecture for multi-class industrial anomaly localization. 

## Environment

Use Python 3.10 or 3.11 and run the commands from the `ACDA` directory. For GPU training, install the matching CUDA builds of PyTorch and torchvision before installing the remaining requirements.

```
python -m pip install -r requirements.txt
```

## MVTec AD

Extract the dataset so that the specified root directly contains folders such as `bottle` and `cable`. Training uses `train/good`; testing uses `test/good` and each defect folder under `test`. The mask for `bottle/test/broken_large/000.png` is `bottle/ground_truth/broken_large/000_mask.png`.

The main configuration resizes images to 256 × 256 using bilinear interpolation, converts grayscale and color inputs to RGB, and scales pixels to `[0, 1]`. Masks use nearest-neighbor resizing and binary values. Normal test images receive zero masks. The model applies the CLIP, ResNet-50, and VAE normalizations separately.

[MVTec AD dataset](https://www.mvtec.com/research-teaching/datasets/mvtec-ad) · [Anomalib MVTec AD data reference](https://anomalib.readthedocs.io/en/v1/markdown/guides/reference/data/image/mvtec.html)
[MVTec AD dataset].(https://github.com/amazon-science/spot-diff）
```
python -m data.mvtec_ad --root datasets/mvtec_ad
```

## Training, testing, and inference

```
python -m engine.train --config config/mvtec_ad.yaml --data-root datasets/mvtec_ad
python -m engine.test --checkpoint runs/mvtec_ad/last.pt --data-root datasets/mvtec_ad --output runs/mvtec_ad/metrics.json
python -m engine.infer --checkpoint runs/mvtec_ad/last.pt --input path/to/image.png --output runs/mvtec_ad/predictions
```

