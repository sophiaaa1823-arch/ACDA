import torch
import torch.nn.functional as F
from torch import nn

from .acst import ACST
from .ahfg import AHFG
from .backbones import CLIPVision, CompactPyramid, CompactVAE, CompactVision, LatentVAE, ResNet50Pyramid
from .diffusion import DiffusionUNet, LatentDiffusionSchedule
from .msadn import MSADN


class ACDA(nn.Module):
    def __init__(self, settings, image_size):
        super().__init__()
        if image_size < 32 or image_size % 16:
            raise ValueError(image_size)
        family = settings["backbones"]
        if family["kind"] == "compact":
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(family["seed"])
                vision = CompactVision(image_size)
                vae = CompactVAE()
                pyramid = CompactPyramid()
        elif family["kind"] == "pretrained":
            vision = CLIPVision(family["clip"], image_size, family["local_files_only"])
            vae = LatentVAE(family["vae"], family["local_files_only"])
            pyramid = ResNet50Pyramid(family["resnet_weights"])
        else:
            raise ValueError(family["kind"])

        self.vae = vae.requires_grad_(False).eval()
        self.pyramid = pyramid.requires_grad_(False).eval()
        self.acst = ACST(vision, **settings["acst"])
        local = settings["msadn"]
        self.msadn = MSADN(pyramid.channels, **local)
        diffusion = settings["unet"]
        fusion = AHFG(vision.dim, vision.grid, local["widths"], diffusion["channels"], **settings["ahfg"])
        self.unet = DiffusionUNet(fusion=fusion, **diffusion)
        self.schedule = LatentDiffusionSchedule(**settings["diffusion"])

    def train(self, mode=True):
        super().train(mode)
        self.vae.eval()
        self.pyramid.eval()
        self.acst.vision.eval()
        return self

    def forward(self, images, categories):
        clean_semantic, perturbed_semantic, perturbed = self.acst.training_pair(images, categories)
        with torch.no_grad():
            clean_latents = self.vae.encode(images).float()
            teacher = self.pyramid(images)
        local, encoded, decoded, filtered = self.msadn(torch.cat((images, perturbed)), encode_features=True)
        global_tokens = torch.cat((clean_semantic, perturbed_semantic))
        guidance = self.unet.fusion.prepare(global_tokens, local, clean_latents.shape[-2:])
        timesteps = torch.randint(self.schedule.train_steps, (2 * len(images),), device=images.device)
        noise = torch.randn((2 * len(images), *clean_latents.shape[1:]), device=images.device)
        latents = self.schedule.add_noise(torch.cat((clean_latents, clean_latents)), noise, timesteps)
        predicted = self.unet(latents, timesteps, guidance)
        return {
            "clean_semantic": clean_semantic,
            "perturbed_semantic": perturbed_semantic,
            "encoded": encoded,
            "local_features": local,
            "decoded": decoded,
            "filtered": filtered,
            "teacher": teacher,
            "noise": noise,
            "predicted_noise": predicted,
        }

    @torch.no_grad()
    def reconstruct(self, images, steps, start_step, generator):
        latents = self.vae.encode(images).float()
        guidance = self.unet.fusion.prepare(self.acst(images), self.msadn(images), latents.shape[-2:])
        decoded_latents = self.schedule.reconstruct(self.unet, latents, guidance, steps, start_step, generator)
        return self.vae.decode(decoded_latents)

    @torch.no_grad()
    def predict(self, images, steps, start_step, generator):
        reconstructed = self.reconstruct(images, steps, start_step, generator)
        source_features = self.pyramid(images)
        reconstructed_features = self.pyramid(reconstructed)
        levels = []
        for source, target in zip(source_features, reconstructed_features):
            difference = (1 - F.cosine_similarity(source.float(), target.float(), dim=1)).clamp(0, 2)
            levels.append(F.interpolate(difference[:, None], size=images.shape[-2:], mode="bilinear", align_corners=False))
        anomaly = torch.stack(levels).sum(dim=0)
        return {"reconstruction": reconstructed, "anomaly_map": anomaly, "image_score": anomaly.flatten(1).amax(dim=1)}
