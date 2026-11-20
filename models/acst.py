import math

import torch
import torch.nn.functional as F
from torch import nn


def simplex_corner(i, j, x, y):
    angle = ((i * 1619 + j * 31337) ^ ((i * 6971) + (j * 1013))) & 1023
    angle = angle.to(x.dtype) * (2 * math.pi / 1024)
    attenuation = (0.5 - x.square() - y.square()).clamp_min(0)
    return attenuation.pow(4) * (angle.cos() * x + angle.sin() * y)


def simplex_noise(images, frequency):
    batch, _, height, width = images.shape
    yy, xx = torch.meshgrid(
        torch.arange(height, device=images.device, dtype=torch.float32) / height * frequency,
        torch.arange(width, device=images.device, dtype=torch.float32) / width * frequency,
        indexing="ij",
    )
    offsets = torch.rand(batch, 2, 1, 1, device=images.device) * 128
    x, y = xx[None] + offsets[:, 0], yy[None] + offsets[:, 1]
    skew = (math.sqrt(3) - 1) / 2
    unskew = (3 - math.sqrt(3)) / 6
    i = (x + (x + y) * skew).floor().long()
    j = (y + (x + y) * skew).floor().long()
    x0 = x - i + (i + j) * unskew
    y0 = y - j + (i + j) * unskew
    i1 = (x0 > y0).long()
    j1 = 1 - i1
    corners = (
        simplex_corner(i, j, x0, y0)
        + simplex_corner(i + i1, j + j1, x0 - i1 + unskew, y0 - j1 + unskew)
        + simplex_corner(i + 1, j + 1, x0 - 1 + 2 * unskew, y0 - 1 + 2 * unskew)
    )
    return (70 * corners[:, None]).clamp(-1, 1)


class SemanticAdapter(nn.Module):
    def __init__(self, input_dim, rank, heads, token_count):
        super().__init__()
        if rank % heads:
            raise ValueError((rank, heads))
        self.heads = heads
        self.down = nn.Linear(input_dim, rank, bias=False)
        self.norm = nn.LayerNorm(rank)
        self.qkv = nn.Linear(rank, rank * 3, bias=False)
        self.positional_bias = nn.Parameter(torch.zeros(1, heads, token_count, token_count))
        self.up = nn.Linear(rank, input_dim, bias=False)
        nn.init.orthogonal_(self.down.weight)
        nn.init.orthogonal_(self.up.weight)

    def forward(self, tokens):
        batch, count, _ = tokens.shape
        query, key, value = self.qkv(self.norm(self.down(tokens))).chunk(3, dim=-1)
        query = query.reshape(batch, count, self.heads, -1).transpose(1, 2)
        key = key.reshape(batch, count, self.heads, -1).transpose(1, 2)
        value = value.reshape(batch, count, self.heads, -1).transpose(1, 2)
        output = F.scaled_dot_product_attention(
            query, key, value, attn_mask=self.positional_bias.to(query.dtype)
        )
        return tokens + self.up(output.transpose(1, 2).flatten(2))


class ACST(nn.Module):
    def __init__(self, vision, rank, heads, saliency_threshold, perturbation_strength, simplex_frequency):
        super().__init__()
        self.vision = vision.requires_grad_(False).eval()
        self.grid = vision.grid
        self.adapter = SemanticAdapter(vision.dim, rank, heads, 1 + self.grid ** 2)
        self.saliency_threshold = saliency_threshold
        self.perturbation_strength = perturbation_strength
        self.simplex_frequency = simplex_frequency

    def forward(self, images):
        with torch.no_grad():
            tokens, _ = self.vision(images)
        return self.adapter(tokens)

    def saliency_mask(self, tokens, size):
        patches = F.cosine_similarity(tokens[:, 1:].float(), tokens[:, :1].float(), dim=-1)
        saliency = patches.reshape(-1, 1, self.grid, self.grid)
        saliency = F.interpolate(saliency, size=size, mode="bilinear", align_corners=False)
        minimum = saliency.amin(dim=(2, 3), keepdim=True)
        span = saliency.amax(dim=(2, 3), keepdim=True) - minimum
        return (saliency - minimum) > (self.saliency_threshold * span)

    def training_pair(self, images, categories):
        with torch.no_grad(), torch.autocast(device_type=images.device.type, enabled=False):
            tokens, embedding = self.vision(images.float())
            neighbors = (categories[:, None] == categories[None, :]).float()
            prototypes = (neighbors @ embedding) / neighbors.sum(dim=1, keepdim=True)
            mask = self.saliency_mask(tokens, images.shape[-2:])
            noise = simplex_noise(images, self.simplex_frequency)
            requires_probe = (embedding - prototypes).square().sum(dim=1) < 1e-12
            probe = images.float() + requires_probe[:, None, None, None] * (0.01 * noise * mask)

        with torch.enable_grad(), torch.autocast(device_type=images.device.type, enabled=False):
            probe = probe.clamp(0, 1).requires_grad_(True)
            _, probe_embedding = self.vision(probe)
            discrepancy = (probe_embedding - prototypes).square().sum()
            gradient, = torch.autograd.grad(discrepancy, probe)

        with torch.no_grad(), torch.autocast(device_type=images.device.type, enabled=False):
            gradient_scale = gradient.abs().amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-8)
            perturbation = noise * (gradient / gradient_scale * mask).tanh()
            adversarial = (images + self.perturbation_strength * perturbation).clamp(0, 1)
            adversarial_tokens, _ = self.vision(adversarial.float())
        return self.adapter(tokens), self.adapter(adversarial_tokens), adversarial
