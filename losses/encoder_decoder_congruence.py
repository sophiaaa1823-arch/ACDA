from .feature_norm import l2_per_sample


def encoder_decoder_congruence(encoded, decoded):
    residual = l2_per_sample(encoded - decoded).square()
    return (residual / l2_per_sample(encoded).clamp_min(1e-6)).mean()
