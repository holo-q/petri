from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST

import const
from core import Petri, PetriMapping, Run, StepMetrics, TissueConfig, TrainingStep
from gui_tensor import TensorModality
from imgui_bundle import hello_imgui, imgui as gui, imgui_ctx, implot, ImVec2


class StableHolographicModule(nn.Module):
    """Stabilized spatial module with controlled complexity"""

    def __init__(self, dim: int, position: Tuple[float, float, float]):
        super().__init__()
        self.dim = dim
        self.position = torch.tensor(position)

        # Field components
        self.field_basis = nn.Parameter(torch.randn(8, dim) * 0.02)
        self.spatial_mix = nn.Parameter(torch.randn(3, dim) * 0.02)

        # Main transform
        self.transform = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim)
        )

        # Analysis metrics
        self.register_buffer('activation_history', torch.zeros(1000))
        self.register_buffer('neighbor_influence', torch.zeros(1000))
        self.history_idx = 0

    def forward(self, x: torch.Tensor,
                nearby_states: Optional[List[torch.Tensor]] = None) -> Tuple[torch.Tensor, Dict]:
        # Spatial influence
        pos_weights = torch.sigmoid(self.position @ self.spatial_mix)
        field_value = (self.field_basis * pos_weights).mean(0)

        # Neighbor integration
        neighbor_contribution = 0
        if nearby_states and len(nearby_states) > 0:
            neighbor_info = torch.stack(nearby_states).mean(0)
            neighbor_contribution = 0.1 * neighbor_info

        # Transform
        x = x + field_value + neighbor_contribution
        output = self.transform(x)

        # Track metrics
        with torch.no_grad():
            if self.training:
                self.activation_history[self.history_idx] = output.abs().mean().item()
                self.neighbor_influence[self.history_idx] = neighbor_contribution.abs().mean().item()
                self.history_idx = (self.history_idx + 1) % 1000

        return output, {
            'field_influence':    field_value.abs().mean().item(),
            'neighbor_influence': neighbor_contribution.abs().mean().item(),
            'activation':         output.abs().mean().item()
        }


class StableHolographicLayer(nn.Module):
    def __init__(self, dim: int, grid_size: int = 2):
        super().__init__()
        self.dim = dim
        self.grid_size = grid_size

        # Create module grid
        self.modules = nn.ModuleList([
            StableHolographicModule(
                dim,
                position=(x / (grid_size - 1), y / (grid_size - 1), z / (grid_size - 1))
            )
            for x in range(grid_size)
            for y in range(grid_size)
            for z in range(grid_size)
        ])

        self.combine = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim)
        )

    def get_nearby_indices(self, idx: int) -> List[int]:
        x = idx // (self.grid_size * self.grid_size)
        y = (idx % (self.grid_size * self.grid_size)) // self.grid_size
        z = idx % self.grid_size

        nearby = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    nx, ny, nz = x + dx, y + dy, z + dz
                    if (0 <= nx < self.grid_size and
                        0 <= ny < self.grid_size and
                        0 <= nz < self.grid_size):
                        nidx = nx * self.grid_size * self.grid_size + ny * self.grid_size + nz
                        if nidx != idx:
                            nearby.append(nidx)
        return nearby

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[Dict]]:
        module_states = []
        module_metrics = []

        # Process through spatial modules
        for i, module in enumerate(self.modules):
            nearby_idx = self.get_nearby_indices(i)
            nearby_states = [module_states[j][0] for j in nearby_idx if j < len(module_states)]
            state, metrics = module(x, nearby_states)
            module_states.append((state, metrics))
            module_metrics.append(metrics)

        # Combine outputs
        states = torch.stack([s[0] for s in module_states], dim=1)
        return self.combine(states.mean(dim=1)), module_metrics


class StableHolographicAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.dims = [32, 64, 128]

        # Image processing
        self.to_features = nn.Sequential(
            nn.Conv2d(1, self.dims[0], 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(self.dims[0], self.dims[0], 4, stride=2, padding=1),
            nn.GELU(),
        )

        # Holographic processing
        self.encoder = nn.ModuleList([
            nn.Sequential(
                StableHolographicLayer(self.dims[i], grid_size=2),
                nn.LayerNorm(self.dims[i]),
                nn.Linear(self.dims[i], self.dims[i + 1])
            )
            for i in range(len(self.dims) - 1)
        ])

        self.decoder = nn.ModuleList([
            nn.Sequential(
                StableHolographicLayer(self.dims[i + 1], grid_size=2),
                nn.LayerNorm(self.dims[i + 1]),
                nn.Linear(self.dims[i + 1], self.dims[i])
            )
            for i in range(len(self.dims) - 1)
        ])

        self.to_image = nn.Sequential(
            nn.ConvTranspose2d(self.dims[0], 32, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[List[Dict]]]:
        # Track metrics from all layers
        layer_metrics = []

        # Initial processing
        x = self.to_features(x)
        B, C, H, W = x.shape
        x = x.reshape(B, C, -1).transpose(1, 2)

        # Encode
        features = [x]
        for layer in self.encoder:
            x, metrics = layer[0](x)  # Get holographic metrics
            layer_metrics.append(metrics)
            x = layer[1:](x)  # Rest of processing
            features.append(x)

        # Decode
        for i, layer in enumerate(self.decoder):
            x, metrics = layer[0](x)
            layer_metrics.append(metrics)
            x = layer[1:](x)
            x = x + features[-(i + 2)]

        # Reconstruct
        x = x.transpose(1, 2).reshape(B, self.dims[0], H, W)
        return self.to_image(x), layer_metrics


class HolographicPetri(Petri):
    def __init__(self, run: Run):
        super().__init__(run)

        # Data setup
        self.transform = transforms.ToTensor()
        self.train_dataset = MNIST('./data', train=True, download=True,
            transform=self.transform)
        self.val_dataset = MNIST('./data', train=False,
            transform=self.transform)
        self.train_loader = self.create_training_loader(1)
        self.val_loader = self.create_validation_loader(1)

        # Visualization state
        self.current_metrics = None

    def train_step(self, step_num: int, step: TrainingStep) -> StepMetrics:
        imgs, _ = next(iter(self.train_loader))
        imgs = imgs.to(const.device)

        self.physics.zero_grad()
        recon, metrics = self.tissue(imgs)

        # Reconstruction loss
        loss = F.mse_loss(recon, imgs)

        # Track metrics
        self.current_metrics = metrics

        # Backward pass
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.tissue.parameters(), 1.0)
        self.physics.step()

        # Visualization
        self.view('input', imgs[0].detach().cpu().numpy(),
            TensorModality.CHW_IMAGE_GRAYSCALE)
        self.view('output', recon[0].detach().cpu().numpy(),
            TensorModality.CHW_IMAGE_GRAYSCALE)

        return StepMetrics(
            loss=loss.item(),
            grad_norm=grad_norm.item(),
            param_norm=sum(p.norm().item() ** 2
                           for p in self.tissue.parameters()) ** 0.5,
            learning_rate=self.physics.param_groups[0]['lr']
        )

    def gui(self):
        if imgui.begin("Holographic Analysis"):
            if self.current_metrics:
                if imgui.tree_node("Layer Analysis"):
                    for i, layer_metrics in enumerate(self.current_metrics):
                        if imgui.tree_node(f"Layer {i}"):
                            # Show average metrics for this layer
                            field_inf = np.mean([m['field_influence']
                                                 for m in layer_metrics])
                            neigh_inf = np.mean([m['neighbor_influence']
                                                 for m in layer_metrics])
                            act = np.mean([m['activation']
                                           for m in layer_metrics])

                            imgui.text(f"Field Influence: {field_inf:.4f}")
                            imgui.text(f"Neighbor Influence: {neigh_inf:.4f}")
                            imgui.text(f"Activation: {act:.4f}")

                            imgui.tree_pop()
                    imgui.tree_pop()

                if imgui.tree_node("Module Grid"):
                    # Show spatial grid structure
                    for layer_idx, layer_metrics in enumerate(self.current_metrics):
                        if imgui.tree_node(f"Layer {layer_idx} Grid"):
                            grid_size = int(round(len(layer_metrics) ** (1 / 3)))
                            for x in range(grid_size):
                                for y in range(grid_size):
                                    for z in range(grid_size):
                                        idx = x * grid_size * grid_size + y * grid_size + z
                                        metrics = layer_metrics[idx]
                                        imgui.text(
                                            f"({x},{y},{z}): "
                                            f"Act={metrics['activation']:.3f}"
                                        )
                            imgui.tree_pop()
                    imgui.tree_pop()
        imgui.end()


@dataclass
class HoloTissue(TissueConfig):
    def create(self, *args, **kwargs) -> nn.Module:
        return StableHolographicAE()

__petri__ = PetriMapping(
    tissue_class=StableHolographicAE,
    config_class=HoloTissue,
    petri_class=HolographicPetri,
)