from .feature_norm import l2_per_sample


def filtered_feature_consistency(clean, perturbed):
    return l2_per_sample(clean - perturbed).mean()
