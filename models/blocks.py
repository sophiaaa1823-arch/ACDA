from torch import nn


def group_norm(channels):
    groups = min(32, max(1, channels // 2))
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1, time_dim=None):
        super().__init__()
        self.norm1 = group_norm(input_channels)
        self.conv1 = nn.Conv2d(input_channels, output_channels, 3, stride, 1)
        self.norm2 = group_norm(output_channels)
        self.conv2 = nn.Conv2d(output_channels, output_channels, 3, padding=1)
        self.act = nn.SiLU()
        self.time = nn.Linear(time_dim, output_channels * 2) if time_dim else None
        self.skip = (
            nn.Conv2d(input_channels, output_channels, 1, stride)
            if stride != 1 or input_channels != output_channels
            else nn.Identity()
        )

    def forward(self, x, time=None):
        hidden = self.conv1(self.act(self.norm1(x)))
        hidden = self.norm2(hidden)
        if self.time is not None:
            scale, shift = self.time(self.act(time)).chunk(2, dim=1)
            hidden = hidden * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        return self.skip(x) + self.conv2(self.act(hidden))
