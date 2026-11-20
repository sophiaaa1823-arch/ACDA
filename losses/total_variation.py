def total_variation(features):
    features = features.float()
    vertical = features[:, :, 1:] - features[:, :, :-1]
    horizontal = features[:, :, :, 1:] - features[:, :, :, :-1]
    return vertical.abs().sum() / max(1, vertical.numel()) + horizontal.abs().sum() / max(1, horizontal.numel())
