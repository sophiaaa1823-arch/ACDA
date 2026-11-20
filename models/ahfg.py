import torch
import torch.nn.functional as F
from torch import nn


class CrossScaleAttention(nn.Module):
    def __init__(self, global_dim, local_dim, output_dim, heads):
        super().__init__()
        if output_dim % heads:
            raise ValueError((output_dim, heads))
        self.heads = heads
        self.query = nn.Linear(global_dim, output_dim)
        self.key = nn.Linear(local_dim, output_dim)
        self.value = nn.Linear(local_dim, output_dim)
        self.output = nn.Linear(output_dim, output_dim)

    def forward(self, global_tokens, local_features):
        batch, tokens, _ = global_tokens.shape
        local_tokens = local_features.flatten(2).transpose(1, 2)
        query = self.query(global_tokens).reshape(batch, tokens, self.heads, -1).transpose(1, 2)
        key = self.key(local_tokens).reshape(batch, -1, self.heads, query.shape[-1]).transpose(1, 2)
        value = self.value(local_tokens).reshape(batch, -1, self.heads, query.shape[-1]).transpose(1, 2)
        fused = F.scaled_dot_product_attention(query, key, value)
        return self.output(fused.transpose(1, 2).flatten(2))


class DMGC(nn.Module):
    def __init__(self, global_dim, local_channels, control_dim):
        super().__init__()
        self.global_projection = nn.Linear(global_dim, control_dim)
        self.local_projections = nn.ModuleList(nn.Linear(c, control_dim) for c in local_channels)
        self.controller = nn.Sequential(
            nn.Linear(2 * control_dim, control_dim), nn.SiLU(), nn.Linear(control_dim, 1)
        )

    def forward(self, global_tokens, local_features):
        global_summary = self.global_projection(global_tokens.mean(dim=1))
        scores = []
        for feature, project in zip(local_features, self.local_projections):
            local_summary = project(feature.mean(dim=(2, 3)))
            scores.append(self.controller(torch.cat((local_summary, global_summary), dim=1)))
        return torch.cat(scores, dim=1).softmax(dim=1)


class AHFG(nn.Module):
    def __init__(self, global_dim, grid, local_channels, decoder_channels, heads, control_dim):
        super().__init__()
        self.grid = grid
        self.controller = DMGC(global_dim, local_channels, control_dim)
        self.attention = nn.ModuleList(
            CrossScaleAttention(global_dim, c, c, heads) for c in local_channels
        )
        total_channels = sum(local_channels)
        self.projections = nn.ModuleList(nn.Conv2d(total_channels, c, 1) for c in decoder_channels)
        self.gates = nn.ModuleList(nn.Conv2d(2 * c, c, 1) for c in decoder_channels)

    def prepare(self, global_tokens, local_features, latent_size):
        weights = self.controller(global_tokens, local_features)
        global_patches = global_tokens[:, 1:]
        scales = []
        for index, (attention, feature) in enumerate(zip(self.attention, local_features)):
            fused = attention(global_patches, feature)
            fused = fused.transpose(1, 2).reshape(fused.shape[0], -1, self.grid, self.grid)
            scales.append(fused * weights[:, index, None, None, None])
        combined = torch.cat(scales, dim=1)
        guidance = []
        size = latent_size
        for project in self.projections:
            resized = F.interpolate(combined, size=size, mode="bilinear", align_corners=False)
            guidance.append(project(resized))
            size = tuple((dimension + 1) // 2 for dimension in size)
        return tuple(guidance)

    def fuse(self, level, diffusion_features, fused_features):
        gate = self.gates[level](torch.cat((diffusion_features, fused_features), dim=1)).sigmoid()
        return gate * diffusion_features + (1 - gate) * fused_features
