from .feature_norm import l2_per_sample


def semantic_feature_congruence(clean, perturbed):
    return l2_per_sample(clean - perturbed).mean()
