from .encoder_decoder_congruence import encoder_decoder_congruence
from .total_variation import total_variation


def multi_level_feature_congruence(encoded, decoded):
    reconstruction = sum(
        encoder_decoder_congruence(source, target)
        for source, target in zip(encoded[1:-1], decoded[1:-1])
    )
    return reconstruction + sum(total_variation(feature) for feature in decoded)
