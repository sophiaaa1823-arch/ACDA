import math


def l2_per_sample(tensor):
    values = tensor.float().flatten(1)
    return values.norm(dim=1) / math.sqrt(values.shape[1])
