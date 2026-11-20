import torch.nn.functional as F


def kl_divergence(clean, perturbed):
    clean_distribution = clean.float().log_softmax(dim=1)
    perturbed_distribution = perturbed.float().log_softmax(dim=1)
    divergence = F.kl_div(perturbed_distribution, clean_distribution, reduction="none", log_target=True)
    return divergence.sum(dim=1).mean()
