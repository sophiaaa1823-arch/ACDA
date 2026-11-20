import torch
from torch import nn

from .adversarial_semantic_consistency import semantic_consistency
from .diffusion_denoising import diffusion_denoising
from .local_alignment import local_alignment
from .multi_level_feature_congruence import multi_level_feature_congruence


class ACDALoss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = weights

    def forward(self, outputs, adapter):
        with torch.autocast(device_type=outputs["noise"].device.type, enabled=False):
            asc = semantic_consistency(
                outputs["clean_semantic"], outputs["perturbed_semantic"], adapter, self.weights["asc"]
            )
            mfc = multi_level_feature_congruence(outputs["encoded"], outputs["local_features"])
            local = local_alignment(outputs["teacher"], outputs["decoded"], outputs["filtered"], self.weights["local"])
            diffusion = diffusion_denoising(outputs["predicted_noise"], outputs["noise"])
        weights = self.weights["total"]
        total = (
            weights["asc"] * asc
            + weights["mfc"] * (mfc + self.weights["local_scale"] * local)
            + weights["diffusion"] * diffusion
        )
        return {"total": total, "asc": asc, "mfc": mfc, "local": local, "diffusion": diffusion}
