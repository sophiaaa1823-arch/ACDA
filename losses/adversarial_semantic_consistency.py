from .kl_divergence import kl_divergence
from .orthogonality import projection_orthogonality
from .semantic_feature_congruence import semantic_feature_congruence


def semantic_consistency(clean, perturbed, adapter, weights):
    return (
        weights["kl"] * kl_divergence(clean, perturbed)
        + weights["feature"] * semantic_feature_congruence(clean, perturbed)
        + weights["orthogonality"] * projection_orthogonality(adapter.down.weight, adapter.up.weight)
    )
