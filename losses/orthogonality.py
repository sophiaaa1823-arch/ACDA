import torch


def projection_orthogonality(down_weight, up_weight):
    down, up = down_weight.float(), up_weight.float()
    identity = torch.eye(down.shape[0], device=down.device, dtype=down.dtype)
    return (down @ down.T - identity).square().sum() + (up.T @ up - identity).square().sum()
