from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn

import const
from core import PetriMapping, TissueConfig

class CronkleBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        # Reference points from CBD doc
        self.refs = {
            'phi':   0.618034,  # φ growth
            'e':     0.367879,  # e decay
            'pi':    0.318309,  # π cycle
            'sqrt2': 0.707107,  # √2 diagonal
            'ln2':   0.693147  # ln(2) doubling
        }

        self.norm = nn.LayerNorm(dim)
        # Project to multi-reference space
        self.to_refs = nn.Linear(dim, dim * len(self.refs))
        # Combine references
        self.combine = nn.Linear(dim * len(self.refs), dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.norm(x)

        # Project to reference points
        refs = self.to_refs(x)
        B, N, D = x.shape
        refs = refs.view(B, N, len(self.refs), D)

        # Apply reference-based transforms
        results = []
        for i, (name, ref) in enumerate(self.refs.items()):
            # Bisection towards reference points
            delta = ref - refs[:, :, i].sigmoid()
            results.append(refs[:, :, i] + 0.1 * delta)

        # Combine reference results
        x = torch.stack(results, dim=2)
        x = self.combine(x.reshape(B, N, -1))

        return x + identity


class ResonanceLayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        # 3x3 lattice of mathematical constants
        self.patterns = [
            [0.618034, 0.367879, 0.318309],  # φ e π
            [0.367879, 0.318309, 0.618034],  # e π φ
            [0.318309, 0.618034, 0.367879]  # π φ e
        ]
        self.pattern_conv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply resonance pattern
        B, C, H, W = x.shape
        pattern = torch.tensor(self.patterns).to(x.device)
        pattern = pattern.view(1, 1, 3, 3).repeat(C, 1, 1, 1)
        self.pattern_conv.weight.data = pattern

        return self.pattern_conv(x)


class CronkleAE(nn.Module):
    def __init__(self, *, size: int = 28):
        super().__init__()
        # Use Fibonacci numbers for dimensions
        dims = [1, 8, 13, 21, 34, 55, 89, 144]

        # Encoder with resonance patterns
        self.enc = nn.Sequential(*(
            [nn.Sequential(
                nn.Conv2d(dims[i], dims[i + 1], 3, padding=1),
                ResonanceLayer(dims[i + 1]),
                CronkleBlock(dims[i + 1]),
                nn.GELU()
            ) for i in range(len(dims) - 1)]
        ))

        # Latent projection to φ-space
        self.phi_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(144 * size * size, 144),
            CronkleBlock(144)
        )

        # Decoder with inverse resonance
        self.dec = nn.Sequential(*(
            [nn.Sequential(
                nn.ConvTranspose2d(dims[-i - 1], dims[-i - 2], 3, padding=1),
                ResonanceLayer(dims[-i - 2]),
                CronkleBlock(dims[-i - 2]),
                nn.GELU()
            ) for i in range(len(dims) - 1)]
        ))

        # Classification from φ-aligned features
        self.clf = nn.Sequential(
            CronkleBlock(144),
            nn.Linear(144, 10)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Encode with resonance
        feat = self.enc(x)
        # Project to φ-space
        z = self.phi_proj(feat)
        # Decode with inverse resonance
        recon = self.dec(feat)
        # Classify from φ-aligned features
        logits = self.clf(z)

        return recon, logits


@dataclass
class CronkleConfig(TissueConfig):
    size: int = 28

    def create(self, *args, **kwargs) -> nn.Module:
        return CronkleAE(size=self.size)

from abc import ABC
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from core import Petri, Run, StepMetrics, TrainingStep
from gui_tensor import TensorModality


@dataclass
class CronkleLoss:
    """Loss function for CronkleAE"""

    def __call__(self,
                 recon: torch.Tensor,
                 x: torch.Tensor,
                 logits: torch.Tensor,
                 labels: torch.Tensor
                 ) -> Tuple[torch.Tensor, dict]:
        # Main losses
        recon_loss = F.mse_loss(recon, x)
        class_loss = F.cross_entropy(logits, labels)

        # Phi-space resonance target (golden ratio)
        phi = 0.618034
        latent_res = (logits.softmax(-1).mean() - phi).abs()

        # Combined with resonance term
        total_loss = recon_loss + 0.1 * class_loss + 0.01 * latent_res

        return total_loss, {
            'loss/total':     total_loss.item(),
            'loss/recon':     recon_loss.item(),
            'loss/class':     class_loss.item(),
            'loss/resonance': latent_res.item(),
            'acc':            (logits.argmax(1) == labels).float().mean().item()
        }


class CronklePetri(Petri, ABC):
    def __init__(self, run: Run):
        super().__init__(run)

        # Simple [0,1] scaling for clean parameter space
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.train_dataset = MNIST('./data', train=True, download=True,
            transform=self.transform)
        self.val_dataset = MNIST('./data', train=False,
            transform=self.transform)

        self.train_loader = self.create_training_loader(1)
        self.val_loader = self.create_validation_loader(1)
        self.criterion = CronkleLoss()

    def create_training_loader(self, batch_size: int) -> DataLoader:
        return DataLoader(self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )

    def create_validation_loader(self, batch_size: int) -> DataLoader:
        return DataLoader(self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )

    def train_step(self, step_num: int, step: TrainingStep) -> StepMetrics:
        """Execute a single training step and return metrics"""
        self.tissue.train()
        imgs, labels = next(iter(self.train_loader))
        imgs = imgs.to(const.device)
        labels = labels.to(const.device)

        # Forward
        self.physics.zero_grad()
        recon, logits = self.tissue(imgs)

        # Loss with resonance
        loss, metrics = self.criterion(recon, imgs, logits, labels)
        loss_val = self.validate()

        # Backward
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.tissue.parameters(), 1.0)
        self.physics.step()

        # Visualize
        self.view('input', imgs[0].detach().cpu().numpy(),
            TensorModality.CHW_IMAGE_GRAYSCALE)
        self.view('output', recon[0].detach().cpu().numpy(),
            TensorModality.CHW_IMAGE_GRAYSCALE)

        # Return metrics including resonance
        return StepMetrics(
            loss=metrics['loss/total'],
            loss_val=loss_val,
            resonance=metrics['loss/resonance'],  # Track resonance specifically
            grad_norm=grad_norm.item(),
            param_norm=sum(p.norm().item() ** 2 for p in self.tissue.parameters()) ** 0.5,
            learning_rate=self.physics.param_groups[0]['lr'],
            batch_time=0.0,
            memory_used=torch.cuda.max_memory_allocated() / 1024 ** 2
            if torch.cuda.is_available() else 0,
        )

    def train_epoch(self) -> Dict[str, float]:
        """Run full training epoch"""
        self.tissue.train()
        epoch_metrics: Dict[str, float] = {}

        for batch in self.train_loader:
            # Dummy step for optimizer
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

            # Accumulate
            for k, v in metrics.__dict__.items():
                epoch_metrics[k] = epoch_metrics.get(k, 0) + v

        # Average
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
        """Run inference mode"""
        self.tissue.eval()
        with torch.no_grad():
            imgs, labels = next(iter(self.val_loader))
            imgs = imgs.to(const.device)
            recon, logits = self.tissue(imgs)
            self.current_input = imgs[0].detach().cpu().numpy()
            self.output = recon[0].detach().cpu().numpy()

    def gui(self):
        pass


# Register the mapping
__petri__ = PetriMapping(
    tissue_class=CronkleAE,
    config_class=CronkleConfig,
    petri_class=CronklePetri,
)