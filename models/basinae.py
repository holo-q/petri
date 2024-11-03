from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn
from torchvision import transforms
from torchvision.datasets import MNIST

import const
from core import Petri, PetriMapping, Run, StepMetrics, TissueConfig, TrainingStep
from gui_tensor import TensorModality
from torch.functional import F

class ConvBasinBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        # Proper 2D normalization for convolutions
        self.norm = nn.BatchNorm2d(channels)
        # Keeping channels consistent
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.gate = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.norm(x)
        x = F.gelu(self.conv1(x))
        x = self.conv2(x)
        gate = torch.sigmoid(self.gate(identity))
        return identity + gate * x


class LinearBasinBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.gate = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.norm(x)
        x = self.ff(x)
        gate = torch.sigmoid(self.gate(identity))
        return identity + gate * x


class BasinAE(nn.Module):
    def __init__(self):
        super().__init__()
        # Track dimensions explicitly
        self.hidden_dim = 256  # Latent dimension

        self.enc = nn.Sequential(
            # 28x28 -> 28x28
            nn.Conv2d(1, 32, 3, padding=1),
            nn.GELU(),
            ConvBasinBlock(32),
            # 28x28 -> 14x14
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.GELU(),
            ConvBasinBlock(64),
            # 14x14 -> 7x7
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.GELU(),
            ConvBasinBlock(128)
        )

        # Calculate flattened dimension
        self.flat_dim = 128 * 7 * 7

        # Clear bottleneck for basin formation
        self.to_latent = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flat_dim, self.hidden_dim),
            LinearBasinBlock(self.hidden_dim)
        )

        self.from_latent = nn.Sequential(
            nn.Linear(self.hidden_dim, self.flat_dim),
            nn.GELU(),
            nn.Unflatten(1, (128, 7, 7))
        )

        self.dec = nn.Sequential(
            # 7x7 -> 14x14
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.GELU(),
            ConvBasinBlock(64),
            # 14x14 -> 28x28
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.GELU(),
            ConvBasinBlock(32),
            # Final reconstruction
            nn.Conv2d(32, 1, 3, padding=1)
        )

        # Classifier from latent space
        self.clf = nn.Sequential(
            LinearBasinBlock(self.hidden_dim),
            nn.Linear(self.hidden_dim, 10)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Progressive encoding
        feat = self.enc(x)
        # To latent space
        z = self.to_latent(feat)
        # Classification from latent
        logits = self.clf(z)
        # Back to spatial
        z_spatial = self.from_latent(z)
        # Decode
        recon = self.dec(z_spatial)
        return recon, logits
class BasinPetri(Petri):
    def __init__(self, run: Run):
        super().__init__(run)

        # Just ToTensor - let the noise injection handle normalization
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        # Data setup
        self.train_dataset = MNIST('./data', train=True, download=True,
            transform=self.transform)
        self.val_dataset = MNIST('./data', train=False,
            transform=self.transform)
        self.train_loader = self.create_training_loader(1)
        self.val_loader = self.create_validation_loader(1)

        # Training state
        self.epoch = 0
        self.noise_level = 0.1  # Initial noise
        self.noise_decay = 0.95  # Noise decay rate

    def train_step(self, step_num: int, step: TrainingStep) -> StepMetrics:
        # Get next batch
        imgs, labels = next(iter(self.train_loader))
        imgs = imgs.to(const.device)
        labels = labels.to(const.device)

        # Forward pass
        self.physics.zero_grad()
        recon, logits = self.tissue(imgs)

        # Loss with noise-aware scaling
        recon_loss = F.mse_loss(recon, imgs)  # Target is clean image
        cls_loss = F.cross_entropy(logits, labels)

        # Scale losses based on noise level
        # As noise decreases, reconstruction becomes more important
        beta = 1.0 - self.noise_level
        loss = beta * recon_loss + (1 - beta) * cls_loss

        # Backward pass
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.tissue.parameters(), 1.0)
        self.physics.step()

        # Update noise level
        self.noise_level *= self.noise_decay
        self.noise_level = max(0.001, self.noise_level)  # Min noise floor

        # Save visualization state
        self.view('input', imgs[0].detach().cpu().numpy(), TensorModality.CHW_IMAGE_GRAYSCALE)
        self.view('output', recon[0].detach().cpu().numpy(), TensorModality.CHW_IMAGE_GRAYSCALE)

        # Metrics
        step_metrics = StepMetrics(
            loss=loss.item(),
            grad_norm=grad_norm.item(),
            param_norm=sum(p.norm().item() ** 2
                           for p in self.tissue.parameters()) ** 0.5,
            learning_rate=self.physics.param_groups[0]['lr'],
            batch_time=0.0,
            memory_used=torch.cuda.max_memory_allocated() / 1024 ** 2
            if torch.cuda.is_available() else 0,
        )

        return step_metrics

    def validate(self) -> float:
        self.tissue.eval()
        total_loss = 0
        total_samples = 0

        with torch.no_grad():
            for imgs, labels in self.val_loader:
                imgs = imgs.to(const.device)
                labels = labels.to(const.device)

                recon, logits = self.tissue(imgs)
                recon_loss = F.mse_loss(recon, imgs)
                cls_loss = F.cross_entropy(logits, labels)

                # Use same loss scaling as training
                beta = 1.0 - self.noise_level
                loss = beta * recon_loss + (1 - beta) * cls_loss

                total_loss += loss.item() * imgs.size(0)
                total_samples += imgs.size(0)

        return total_loss / total_samples


@dataclass
class BasinTissue(TissueConfig):
    size: int = 28
    def create(self, *args, **kwargs) -> nn.Module:
        return BasinAE()

__petri__ = PetriMapping(
    tissue_class=BasinAE,
    config_class=BasinTissue,
    petri_class=BasinPetri,
)