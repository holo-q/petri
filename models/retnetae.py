import math
from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn
from torchvision import transforms
from torchvision.datasets import MNIST

import const
from core import Petri, PetriMapping, Run, StepMetrics, TissueConfig, TrainingStep
from gui_tensor import TensorModality
from models.retention import MultiScaleRetention

class RetNetAE(nn.Module):
    def __init__(self):
        super().__init__()
        # Architecture params
        hidden_dim = 256  # Base hidden dimension
        heads = 8  # Number of retention heads
        ffn_size = 512  # FFN intermediate size
        enc_layers = 4  # Number of encoder layers
        dec_layers = 4  # Number of decoder layers

        # Encoder path - convert image to sequence
        self.to_seq = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),  # 28x28 -> 14x14
            nn.GELU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 14x14 -> 7x7
            nn.GELU(),
            nn.Conv2d(64, hidden_dim, 3, padding=1),  # Final encoding
            nn.GELU(),
        )

        # Encoder retention layers
        self.enc_retentions = nn.ModuleList([
            MultiScaleRetention(hidden_dim, heads, double_v_dim=True)
            for _ in range(enc_layers)
        ])
        self.enc_ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, ffn_size),
                nn.GELU(),
                nn.Linear(ffn_size, hidden_dim)
            )
            for _ in range(enc_layers)
        ])
        self.enc_norms1 = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(enc_layers)
        ])
        self.enc_norms2 = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(enc_layers)
        ])

        # Decoder retention layers
        self.dec_retentions = nn.ModuleList([
            MultiScaleRetention(hidden_dim, heads, double_v_dim=True)
            for _ in range(dec_layers)
        ])
        self.dec_ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, ffn_size),
                nn.GELU(),
                nn.Linear(ffn_size, hidden_dim)
            )
            for _ in range(dec_layers)
        ])
        self.dec_norms1 = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(dec_layers)
        ])
        self.dec_norms2 = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(dec_layers)
        ])

        # Decoder path - convert sequence back to image
        self.to_img = nn.Sequential(
            nn.Conv2d(hidden_dim, 64, 3, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 7x7 -> 14x14
            nn.GELU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),  # 14x14 -> 28x28
        )

        # Optional classifier branch
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 10)
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # Convert image to sequence
        B = x.shape[0]
        x = self.to_seq(x)  # B,C,H,W
        x = x.flatten(2).transpose(1, 2)  # B,L,C

        # Apply retention layers
        for i in range(len(self.enc_retentions)):
            # Retention + residual
            y = self.enc_retentions[i](
                self.enc_norms1[i](x)
            ) + x
            # FFN + residual
            x = self.enc_ffns[i](
                self.enc_norms2[i](y)
            ) + y

        return x

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        # Apply retention layers
        for i in range(len(self.dec_retentions)):
            # Retention + residual
            y = self.dec_retentions[i](
                self.dec_norms1[i](x)
            ) + x
            # FFN + residual
            x = self.dec_ffns[i](
                self.dec_norms2[i](y)
            ) + y

        # Convert sequence back to image
        B, L, C = x.shape
        H = W = int(math.sqrt(L))
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.to_img(x)

        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Full forward pass
        z = self.encode(x)

        # Get reconstruction and classification
        recon = self.decode(z)
        logits = self.classifier(z.mean(dim=1))  # Pool sequence for classification

        return recon, logits


@dataclass
class RetNetTissue(TissueConfig):
    def create(self, *args, **kwargs) -> nn.Module:
        return RetNetAE()


class RetNetPetri(Petri):
    def __init__(self, run: Run):
        super().__init__(run)

        # Simple transform as RetNet handles position info
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

        # Loss functions
        self.recon_loss = nn.MSELoss()
        self.class_loss = nn.CrossEntropyLoss()

        # Training mode (can switch between parallel/recurrent)
        self.mode = 'parallel'  # or 'recurrent'

        # State for recurrent mode
        self.hidden_states = None

    def init_states(self, batch_size: int):
        """Initialize recurrent states for RetNet"""
        tissue = self.tissue
        device = next(tissue.parameters()).device

        # Initialize states for encoder
        enc_states = []
        for retention in tissue.enc_retentions:
            layer_states = []
            for head in retention.heads:
                head_state = torch.zeros(
                    batch_size,
                    retention.hidden_size // retention.heads,
                    retention.v_dim // retention.heads,
                    device=device
                )
                layer_states.append(head_state)
            enc_states.append(layer_states)

        # Initialize states for decoder
        dec_states = []
        for retention in tissue.dec_retentions:
            layer_states = []
            for head in retention.heads:
                head_state = torch.zeros(
                    batch_size,
                    retention.hidden_size // retention.heads,
                    retention.v_dim // retention.heads,
                    device=device
                )
                layer_states.append(head_state)
            dec_states.append(layer_states)

        return enc_states, dec_states

    def train_step(self, step_num: int, step: TrainingStep) -> StepMetrics:
        # Get next batch
        imgs, labels = next(iter(self.train_loader))
        imgs = imgs.to(const.device)
        labels = labels.to(const.device)

        # Forward pass
        self.physics.zero_grad()

        if self.mode == 'parallel':
            # Standard parallel processing
            recon, logits = self.tissue(imgs)

        else:  # recurrent mode
            # Initialize states if needed
            if self.hidden_states is None:
                self.hidden_states = self.init_states(imgs.shape[0])

            # Process sequentially
            enc_states, dec_states = self.hidden_states
            z = self.tissue.encode(imgs, enc_states)
            recon, logits = self.tissue.decode(z, dec_states)

            # Update states
            self.hidden_states = (enc_states, dec_states)

        # Combined loss
        recon_loss = self.recon_loss(recon, imgs)
        class_loss = self.class_loss(logits, labels)
        loss = recon_loss + 0.1 * class_loss  # Weight classification less

        # Backward pass
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.tissue.parameters(), 1.0)
        self.physics.step()

        # Save visualization state
        self.view('input', imgs[0].detach().cpu().numpy(), TensorModality.CHW_IMAGE_GRAYSCALE)
        self.view('output', recon[0].detach().cpu().numpy(), TensorModality.CHW_IMAGE_GRAYSCALE)

        # Save training mode info
        mode_info = f"Mode: {self.mode}"
        if self.mode == 'recurrent':
            mode_info += f" (step {step_num})"
        self.view('mode_info', mode_info, TensorModality.TEXT)

        # Compute metrics
        with torch.no_grad():
            acc = (logits.argmax(dim=1) == labels).float().mean()

        metrics = StepMetrics(
            loss=loss.item(),
            grad_norm=grad_norm.item(),
            param_norm=sum(p.norm().item() ** 2
                           for p in self.tissue.parameters()) ** 0.5,
            learning_rate=self.physics.param_groups[0]['lr'],
            batch_time=0.0,
            memory_used=torch.cuda.max_memory_allocated() / 1024 ** 2
            if torch.cuda.is_available() else 0,
            accuracy=acc.item()
        )

        return metrics

    def validate(self) -> float:
        self.tissue.eval()
        total_loss = 0
        total_samples = 0

        with torch.no_grad():
            for imgs, labels in self.val_loader:
                imgs = imgs.to(const.device)
                labels = labels.to(const.device)

                # Always use parallel mode for validation
                recon, logits = self.tissue(imgs)

                recon_loss = self.recon_loss(recon, imgs)
                class_loss = self.class_loss(logits, labels)
                loss = recon_loss + 0.1 * class_loss

                total_loss += loss.item() * imgs.size(0)
                total_samples += imgs.size(0)

        return total_loss / total_samples

    def inference(self):
        """Run inference and visualize reconstructions/attention"""
        self.tissue.eval()
        with torch.no_grad():
            imgs, labels = next(iter(self.val_loader))
            imgs = imgs.to(const.device)

            # Get reconstructions
            recon, logits = self.tissue(imgs)

            # Store visualizations
            self.current_input = imgs[0].detach().cpu().numpy()
            self.output = recon[0].detach().cpu().numpy()

__petri__ = PetriMapping(
    tissue_class=RetNetAE,
    config_class=RetNetTissue,
    petri_class=RetNetPetri,
)