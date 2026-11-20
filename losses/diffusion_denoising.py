import torch.nn.functional as F


def diffusion_denoising(predicted_noise, noise):
    return F.mse_loss(predicted_noise.float(), noise)
