import math
from abc import ABC
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

import const
from core import Petri, PetriMapping, Run, StepMetrics, TissueConfig, TrainingStep
from gui_tensor import TensorModality

# Based on tutorial:
# https://bytepawn.com/building-a-pytorch-autoencoder-for-mnist-digits.html

@dataclass
class AutoencoderTissue(TissueConfig):
    def create(self, *args, **kwargs) -> nn.Module:
        return AutoencoderModern()

# class Autoencoder(nn.Module):
#     def __init__(self):
#         super(Autoencoder, self).__init__()
#         self.encoder = nn.Sequential(
#             # 28 x 28
#             nn.Conv2d(1, 4, kernel_size=5),
#             # 4 x 24 x 24
#             nn.ReLU(True),
#             nn.Conv2d(4, 8, kernel_size=5),
#             nn.ReLU(True),
#             # 8 x 20 x 20 = 3200
#             nn.Flatten(),
#             nn.Linear(3200, 10),
#             # 10
#             nn.Softmax(),
#         )
#         self.decoder = nn.Sequential(
#             # 10
#             nn.Linear(10, 400),
#             # 400
#             nn.ReLU(True),
#             nn.Linear(400, 4000),
#             # 4000
#             nn.ReLU(True),
#             nn.Unflatten(1, (10, 20, 20)),
#             # 10 x 20 x 20
#             nn.ConvTranspose2d(10, 10, kernel_size=5),
#             # 24 x 24
#             nn.ConvTranspose2d(10, 1, kernel_size=5),
#             # 28 x 28
#             nn.Sigmoid(),
#         )
#
#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity  # Skip connection
        out = F.relu(out)
        return out


class AutoencoderModern(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()

        # Keep encoder mostly the same but remove input normalization
        self.enc_conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            ResBlock(32),

            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            ResBlock(64),

            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            ResBlock(128),
        )

        # Rest of encoder stays the same
        self.enc_flat = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 10)
        )

        # Decoder stays mostly the same
        self.dec_flat = nn.Sequential(
            nn.Linear(latent_dim, 128 * 7 * 7),
            nn.BatchNorm1d(128 * 7 * 7),
            nn.ReLU(True)
        )

        self.dec_conv = nn.Sequential(
            nn.Unflatten(1, (128, 7, 7)),
            ResBlock(128),

            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            ResBlock(64),

            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            ResBlock(32),

            # Final reconstruction - no sigmoid, let it match input range
            nn.Conv2d(32, 1, 3, padding=1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.enc_conv(x)
        x = self.enc_flat(x)
        return x

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.dec_flat(z)
        x = self.dec_conv(x)
        return x

    def forward(self, x: torch.Tensor, return_latent: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        z = self.encode(x)
        recon = self.decode(z)
        logits = self.classifier(z)

        if return_latent:
            return recon, logits, z
        return recon, logits


@dataclass
class AutoencoderLoss:
    """Combines reconstruction and classification losses"""

    def __call__(self,
                 recon: torch.Tensor,
                 x: torch.Tensor,
                 logits: torch.Tensor,
                 labels: torch.Tensor,
                 beta: float = 0.1
                 ) -> Tuple[torch.Tensor, dict]:
        # MSE for reconstruction (now both are in same range)
        recon_loss = F.mse_loss(recon, x)

        # Cross entropy for classification
        class_loss = F.cross_entropy(logits, labels)

        # Combined loss
        total_loss = recon_loss + beta * class_loss

        metrics = {
            'loss/total': total_loss.item(),
            'loss/recon': recon_loss.item(),
            'loss/class': class_loss.item(),
            'acc':        (logits.argmax(1) == labels).float().mean().item()
        }

        return total_loss, metrics

# ----------------------------------------


class Attn(nn.Module):
    def __init__(self, d: int, h: int, p: float = 0.1):
        super().__init__()
        assert d % h == 0
        self.h, self.k = h, d // h  # heads, key_dim
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.drop = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        qkv = self.qkv(x).reshape(B, -1, 3, self.h, self.k)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        # attn = softmax(QK'/sqrt(d_k))V
        a = F.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.k), -1)
        x = torch.matmul(self.drop(a), v)

        return self.proj(x.transpose(1, 2).reshape(B, -1, self.h * self.k))


class Block(nn.Module):
    def __init__(self, d: int, h: int):
        super().__init__()
        self.attn = Attn(d, h)
        self.n1 = nn.LayerNorm(d)
        self.n2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, 4 * d),
            nn.GELU(),
            nn.Linear(4 * d, d)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(self.n2(x + self.attn(self.n1(x))))


class TrAE(nn.Module):  # Transformer AutoEncoder
    def __init__(self, *, size: int = 28, p: int = 7):  # p = patch_size
        super().__init__()
        d, h = 256, 8  # dim, heads
        n = size // p  # patches per side

        # img -> tokens
        self.tok = nn.Sequential(
            nn.Conv2d(1, d, p, stride=p),
            nn.Flatten(2),
        )

        # pos emb + cls
        self.pos = nn.Parameter(torch.randn(1, n * n, d) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)

        # stacks
        self.enc = nn.ModuleList([Block(d, h) for _ in range(6)])
        self.dec = nn.ModuleList([Block(d, h) for _ in range(6)])

        # heads
        self.recon = nn.ConvTranspose2d(d, 1, p, stride=p)
        self.clf = nn.Linear(d, 10)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tok(x).transpose(1, 2) + self.pos
        x = torch.cat([self.cls.expand(x.shape[0], -1, -1), x], dim=1)
        for block in self.enc: x = block(x)
        return x

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, 1:]  # rm cls
        for block in self.dec: x = block(x)
        # (B,P,C) -> (B,C,H,W)
        B, P, C = x.shape
        H = W = int(math.sqrt(P))
        return self.recon(x.transpose(1, 2).reshape(B, C, H, W))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decode(z), self.clf(z[:, 0])

class AutoencoderPetri(Petri, ABC):
    def __init__(self, run: Run):
        super().__init__(run)

        # Setup data
        # self.transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.1307,), (0.3081,))
        # ])
        # Simpler transform - just convert to tensor and scale to [0,1]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.train_dataset = MNIST('./data', train=True, download=True, transform=self.transform)
        self.val_dataset = MNIST('./data', train=False, transform=self.transform)
        self.train_loader = self.create_training_loader(1)
        self.val_loader = self.create_validation_loader(1)

        # Combined loss function
        self.criterion = AutoencoderLoss()



    def gui(self):
        pass
    def train_step(self, step_num: int, step: TrainingStep) -> StepMetrics:
        """Execute a single training step and return metrics"""
        # Get next batch
        self.tissue.train()
        imgs, labels = next(iter(self.train_loader))
        imgs = imgs.to(const.device)
        labels = labels.to(const.device)

        # Forward pass
        self.physics.zero_grad()
        recon, logits = self.tissue(imgs)  # Now handling tuple return

        # Calculate loss
        loss, metrics = self.criterion(recon, imgs, logits, labels)
        # loss_val = 0
        loss_val = self.validate()

        # Backward pass
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.tissue.parameters(), 1.0)
        self.physics.step()

        # Save visualization state
        self.view('input', imgs[0].detach().cpu().numpy(), TensorModality.CHW_IMAGE_GRAYSCALE)
        self.view('output', recon[0].detach().cpu().numpy(), TensorModality.CHW_IMAGE_GRAYSCALE)

        # Compute metrics
        step_metrics = StepMetrics(
            loss=metrics['loss/total'],
            loss_val=loss_val,
            grad_norm=grad_norm.item(),
            param_norm=sum(p.norm().item() ** 2 for p in self.tissue.parameters()) ** 0.5,
            learning_rate=self.physics.param_groups[0]['lr'],
            batch_time=0.0,  # TODO: Add timing
            memory_used=torch.cuda.max_memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else 0,
        )

        return step_metrics

    def train_epoch(self) -> Dict[str, float]:
        """Run full training epoch"""
        self.tissue.train()
        epoch_metrics: Dict[str, float] = {}

        for batch in self.train_loader:
            # Create a dummy training step since we need the optimizer
            step = TrainingStep(
                learning_rate=self.physics.param_groups[0]['lr'],
                physics_type=None,
                physics_config=None,
                tissue_config=None,
                freeze_layers=None,
                unfreeze_layers=None,
                steps=1,
                _optimizer=self.physics
            )

            metrics = self.train_step(0, step)

            # Accumulate metrics
            for k, v in metrics.__dict__.items():
                epoch_metrics[k] = epoch_metrics.get(k, 0) + v

        # Average metrics
        for k in epoch_metrics:
            epoch_metrics[k] /= len(self.train_loader)

        return epoch_metrics

    def validate(self) -> float:
        self.tissue.eval()
        total_loss = 0
        total_samples = 0

        with torch.no_grad():
            batch = next(iter(self.val_loader))
            imgs, labels = batch
            imgs = imgs.to(const.device)
            labels = labels.to(const.device)

            recon, logits = self.tissue(imgs)
            loss, metrics = self.criterion(recon, imgs, logits, labels)

            total_loss += loss.item() * imgs.size(0)
            total_samples += imgs.size(0)

        return total_loss / total_samples

    def inference(self):
        """Run inference on current tissue state"""
        self.tissue.eval()
        with torch.no_grad():
            # Get a batch of validation images
            imgs, labels = next(iter(self.val_loader))
            imgs = imgs.to(const.device)

            # Generate reconstructions and classifications
            recon, logits = self.tissue(imgs)

            # Store first image and reconstruction for visualization
            self.current_input = imgs[0].detach().cpu().numpy()
            self.output = recon[0].detach().cpu().numpy()
# ------------------------------------------------------------

__petri__ = PetriMapping(
    tissue_class=AutoencoderModern,
    config_class=AutoencoderTissue,
    petri_class=AutoencoderPetri,
)

def main():
    # Example usage:
    model = AutoencoderModern()  # Your MNIST autoencoder model
    petri = AutoencoderPetri(model)


if __name__ == '__main__':
    main()
