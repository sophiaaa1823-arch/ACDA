import torch
import torch.nn.functional as F
from torch import nn

from .blocks import ResidualBlock


class CLIPVision(nn.Module):
    def __init__(self, source, image_size, local_files_only):
        super().__init__()
        from transformers import CLIPVisionModelWithProjection

        self.model = CLIPVisionModelWithProjection.from_pretrained(
            source, local_files_only=local_files_only
        )
        self.dim = self.model.config.hidden_size
        self.grid = image_size // self.model.config.patch_size
        if image_size % self.model.config.patch_size:
            raise ValueError(image_size)
        self.register_buffer("mean", torch.tensor([0.48145466, 0.4578275, 0.40821073])[None, :, None, None])
        self.register_buffer("std", torch.tensor([0.26862954, 0.26130258, 0.27577711])[None, :, None, None])

    def forward(self, images):
        result = self.model(pixel_values=(images - self.mean) / self.std, interpolate_pos_encoding=True)
        return result.last_hidden_state, F.normalize(result.image_embeds.float(), dim=-1)


class LatentVAE(nn.Module):
    def __init__(self, source, local_files_only):
        super().__init__()
        from diffusers import AutoencoderKL

        self.model = AutoencoderKL.from_pretrained(source, local_files_only=local_files_only)
        self.scaling = self.model.config.scaling_factor
        if self.model.config.latent_channels != 4 or len(self.model.config.block_out_channels) != 4:
            raise ValueError(source)

    def encode(self, images):
        return self.model.encode(images * 2 - 1).latent_dist.mode() * self.scaling

    def decode(self, latents):
        return (self.model.decode(latents / self.scaling).sample * 0.5 + 0.5).clamp(0, 1)


class ResNet50Pyramid(nn.Module):
    channels = (256, 512, 1024)

    def __init__(self, weights):
        super().__init__()
        from torchvision.models import ResNet50_Weights, resnet50

        source = resnet50(weights=ResNet50_Weights[weights])
        self.stem = nn.Sequential(source.conv1, source.bn1, source.relu, source.maxpool)
        self.layers = nn.ModuleList((source.layer1, source.layer2, source.layer3))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])

    def forward(self, images):
        features = []
        hidden = self.stem((images - self.mean) / self.std)
        for layer in self.layers:
            hidden = layer(hidden)
            features.append(hidden)
        return tuple(features)


class CompactVision(nn.Module):
    dim = 48

    def __init__(self, image_size):
        super().__init__()
        self.grid = image_size // 8
        self.patch = nn.Conv2d(3, self.dim, 8, stride=8)
        self.cls = nn.Parameter(torch.zeros(1, 1, self.dim))
        self.positions = nn.Parameter(torch.randn(1, 1 + self.grid ** 2, self.dim) * 0.02)
        layer = nn.TransformerEncoderLayer(self.dim, 4, self.dim * 2, 0, batch_first=True)
        self.blocks = nn.TransformerEncoder(layer, 1)
        self.projection = nn.Linear(self.dim, 32, bias=False)

    def forward(self, images):
        patches = self.patch(images).flatten(2).transpose(1, 2)
        tokens = torch.cat((self.cls.expand(images.shape[0], -1, -1), patches), dim=1)
        tokens = self.blocks(tokens + self.positions)
        return tokens, F.normalize(self.projection(tokens[:, 0]).float(), dim=-1)


class CompactVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(16, 24, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(24, 4, 4, 2, 1),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4, 24, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(24, 16, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(16, 3, 4, 2, 1),
            nn.Sigmoid(),
        )

    def encode(self, images):
        return self.encoder(images * 2 - 1)

    def decode(self, latents):
        return self.decoder(latents)


class CompactPyramid(nn.Module):
    channels = (16, 24, 32)

    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 16, 4, 4)
        self.layers = nn.ModuleList(
            (ResidualBlock(16, 16), ResidualBlock(16, 24, 2), ResidualBlock(24, 32, 2))
        )

    def forward(self, images):
        features = []
        hidden = self.stem(images)
        for layer in self.layers:
            hidden = layer(hidden)
            features.append(hidden)
        return tuple(features)
