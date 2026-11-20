import torch
import torch.nn.functional as F
from torch import nn

from .blocks import ResidualBlock


class AnomalySuppressionGate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.project = nn.Conv2d(channels, channels, 1)

    def forward(self, features):
        return features * self.project(self.pool(features)).sigmoid()


class OCBE(nn.Module):
    def __init__(self, channels, width):
        super().__init__()
        self.projections = nn.ModuleList(nn.Conv2d(c, width, 1) for c in channels)
        self.reduce = ResidualBlock(width * len(channels), width)

    def forward(self, features):
        size = features[-1].shape[-2:]
        pooled = [
            project(F.adaptive_avg_pool2d(feature, size))
            for feature, project in zip(features, self.projections)
        ]
        return self.reduce(torch.cat(pooled, dim=1))


class MSADN(nn.Module):
    def __init__(self, channels, widths, bottleneck_width):
        super().__init__()
        if len(channels) != len(widths):
            raise ValueError((channels, widths))
        self.stem = nn.Conv2d(3, widths[0], 7, stride=4, padding=3)
        self.encoder = nn.ModuleList(
            ResidualBlock(widths[max(level - 1, 0)], width, stride=1 if level == 0 else 2)
            for level, width in enumerate(widths)
        )
        self.ocbe = OCBE(widths, bottleneck_width)
        self.gates = nn.ModuleList(AnomalySuppressionGate(width) for width in widths)
        decoder_inputs = (*widths[1:], bottleneck_width)
        self.decoder = nn.ModuleList(
            ResidualBlock(incoming + width, width)
            for incoming, width in zip(decoder_inputs, widths)
        )
        self.decoded_projections = nn.ModuleList(
            nn.Conv2d(width, channel, 1) for width, channel in zip(widths, channels)
        )

    def forward(self, images, encode_features=False):
        encoded = []
        hidden = self.stem(images * 2 - 1)
        for encode in self.encoder:
            hidden = encode(hidden)
            encoded.append(hidden)
        hidden = self.ocbe(encoded)
        decoded = []
        filtered = []
        for skip, gate, decode in zip(reversed(encoded), reversed(self.gates), reversed(self.decoder)):
            hidden = F.interpolate(hidden, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            filtered_skip = gate(skip)
            hidden = decode(torch.cat((hidden, filtered_skip), dim=1))
            filtered.append(filtered_skip)
            decoded.append(hidden)
        local = tuple(reversed(decoded))
        if not encode_features:
            return local
        decoder_features = tuple(project(feature) for project, feature in zip(self.decoded_projections, local))
        return local, tuple(encoded), decoder_features, tuple(reversed(filtered))
