import math

import torch
import torch.nn.functional as F
from torch import nn

from .blocks import ResidualBlock, group_norm


class TimestepEmbedding(nn.Module):
    def __init__(self, width):
        super().__init__()
        frequencies = torch.exp(-math.log(10000) * torch.arange(width // 2) / (width // 2))
        self.register_buffer("frequencies", frequencies)
        self.project = nn.Sequential(nn.Linear(width, width * 4), nn.SiLU(), nn.Linear(width * 4, width * 4))

    def forward(self, timesteps):
        angles = timesteps[:, None].float() * self.frequencies[None]
        return self.project(torch.cat((angles.cos(), angles.sin()), dim=1))


class DiffusionUNet(nn.Module):
    def __init__(self, channels, time_width, blocks_per_level, fusion):
        super().__init__()
        if time_width < 2 or time_width % 2 or blocks_per_level < 1:
            raise ValueError((time_width, blocks_per_level))
        self.fusion = fusion
        self.time_embedding = TimestepEmbedding(time_width)
        time_dim = time_width * 4
        self.input = nn.Conv2d(4, channels[0], 3, padding=1)
        self.down = nn.ModuleList()
        self.downsample = nn.ModuleList()
        self.up = nn.ModuleList()
        self.upsample = nn.ModuleList()
        for level, width in enumerate(channels):
            self.down.append(nn.ModuleList(
                ResidualBlock(width, width, time_dim=time_dim) for _ in range(blocks_per_level)
            ))
            if level + 1 < len(channels):
                deeper_width = channels[level + 1]
                self.downsample.append(nn.Conv2d(width, deeper_width, 3, stride=2, padding=1))
                self.upsample.append(nn.Conv2d(deeper_width, width, 3, padding=1))
            self.up.append(nn.ModuleList(
                ResidualBlock(2 * width if index == 0 else width, width, time_dim=time_dim)
                for index in range(blocks_per_level)
            ))
        self.middle = nn.ModuleList(
            ResidualBlock(channels[-1], channels[-1], time_dim=time_dim) for _ in range(2)
        )
        self.output = nn.Sequential(group_norm(channels[0]), nn.SiLU(), nn.Conv2d(channels[0], 4, 3, padding=1))

    def forward(self, latents, timesteps, guidance):
        time = self.time_embedding(timesteps)
        hidden = self.input(latents)
        skips = []
        for level, blocks in enumerate(self.down):
            for block in blocks:
                hidden = block(hidden, time)
            skips.append(hidden)
            if level < len(self.downsample):
                hidden = self.downsample[level](hidden)
        for block in self.middle:
            hidden = block(hidden, time)
        for level in reversed(range(len(self.up))):
            skip = skips[level]
            if level < len(self.upsample):
                hidden = F.interpolate(hidden, size=skip.shape[-2:], mode="nearest")
                hidden = self.upsample[level](hidden)
            hidden = torch.cat((hidden, skip), dim=1)
            for block in self.up[level]:
                hidden = block(hidden, time)
            hidden = self.fusion.fuse(level, hidden, guidance[level])
        return self.output(hidden)


class LatentDiffusionSchedule(nn.Module):
    def __init__(self, train_steps, beta_start, beta_end):
        super().__init__()
        if not 0 < beta_start < beta_end < 1 or train_steps < 2:
            raise ValueError((train_steps, beta_start, beta_end))
        betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, train_steps).square()
        self.register_buffer("alpha_bars", torch.cumprod(1 - betas, dim=0))
        self.train_steps = train_steps

    def add_noise(self, clean, noise, times):
        alpha = self.alpha_bars[times][:, None, None, None]
        return alpha.sqrt() * clean + (1 - alpha).sqrt() * noise

    def reconstruct(self, denoiser, clean, guidance, steps, start_step, generator):
        if not 1 <= steps <= start_step + 1 <= self.train_steps:
            raise ValueError((steps, start_step, self.train_steps))
        timesteps = torch.linspace(start_step, 0, steps).round().long().tolist()
        batch = clean.shape[0]
        times = torch.full((batch,), start_step, device=clean.device, dtype=torch.long)
        noise = torch.randn(clean.shape, device=clean.device, dtype=clean.dtype, generator=generator)
        latents = self.add_noise(clean, noise, times)
        for index, time in enumerate(timesteps):
            times = torch.full((batch,), time, device=clean.device, dtype=torch.long)
            noise = denoiser(latents, times, guidance).float()
            alpha = self.alpha_bars[time]
            estimate = (latents - (1 - alpha).sqrt() * noise) / alpha.sqrt()
            if index + 1 == len(timesteps):
                return estimate
            next_alpha = self.alpha_bars[timesteps[index + 1]]
            latents = next_alpha.sqrt() * estimate + (1 - next_alpha).sqrt() * noise
