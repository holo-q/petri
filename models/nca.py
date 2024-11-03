from abc import ABC
from dataclasses import dataclass
from typing import Any, Dict, Optional, T, Tuple

import moderngl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from imgui_bundle import imgui

import const
from core import Petri, PetriMapping, TissueConfig, TrainingStep

@dataclass
class TrainingJob:
    steps: int
    batch_size: int
    target_images: torch.Tensor
    params: dict

class NCA(nn.Module):
    """
    Some NCA tissue learning to perceive and organize with sobel, laplace and gaussian filters.
    """
    def __init__(self, channel_n=16, fire_rate=0.5):
        super().__init__()
        self.channel_n = channel_n
        self.fire_rate = fire_rate

        # Extended perception kernels
        self.register_buffer('identity', self._make_kernel([0, 1, 0]))
        self.register_buffer('dx', self._make_kernel([1, 2, 1], [-1, 0, 1]) / 8.0)
        self.register_buffer('dy', self._make_kernel([-1, 0, 1], [1, 2, 1]) / 8.0)
        # Add Laplacian
        self.register_buffer('laplace', self._make_kernel([[0, 1, 0], [1, -4, 1], [0, 1, 0]]))
        # Add Gaussian blur
        self.register_buffer('gaussian', self._make_kernel([1, 2, 1], [1, 2, 1]) / 16.0)

        self.update_net = nn.Sequential(
            nn.Conv2d(channel_n * 5, 128, 1),  # 5 filters now
            nn.ReLU(),
            nn.Conv2d(128, channel_n, 1)
        )

    def perceive(self, x, angle=0.0):
        c, s = torch.cos(torch.tensor(angle)), torch.sin(torch.tensor(angle))
        dx = c * self.dx - s * self.dy
        dy = s * self.dx + c * self.dy

        filters = torch.stack([
            self.identity,
            dx,
            dy,
            self.laplace,
            self.gaussian
        ])
        filters = filters.unsqueeze(1).repeat(1, self.channel_n, 1, 1)
        filters = filters.view(-1, 1, 3, 3)

        y = F.conv2d(
            x.view(-1, 1, x.shape[2], x.shape[3]),
            filters,
            padding=1,
            groups=self.channel_n
        )
        return y.view(x.shape[0], -1, x.shape[2], x.shape[3])

@dataclass
class NCATissue(TissueConfig):
    """Configuration for Neural CA training and visualization"""
    # tissue configuration
    channel_n: int = 16
    fire_rate: float = 0.5

    # training configuration (technically should be physics?)
    grid_size: Tuple[int, int] = (64, 64)
    target_rgba: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 1.0)
    pattern_radius: int = 10
    batch_size: int = 8
    input_noise: float = 0.1
    growth_steps_min: int = 48
    growth_steps_max: int = 64
    growth_step_mode: str = "random"  # "random" or "fixed"
    seed_size: int = 1
    seed_mode: str = "center"  # "center", "random", or "pattern"

    def create(self, params) -> T:
        NCA(self.channel_n, self.fire_rate)

class NCAPetri(Petri, ABC):
    """Petri interface for NCA models."""

    def __init__(self, model: torch.nn.Module, config: NCATissue):
        super().__init__(model, config)
        self.config: NCATissue = config  # Type hint for IDE

        # OpenGL/Visualization state
        self.state_texture: Optional[moderngl.Texture] = None
        self.current_state: Optional[torch.Tensor] = None
        self.growing_state: Optional[torch.Tensor] = None
        self.viz_mode = "state"  # "state", "alpha", "rgb", or "features"
        self.selected_feature = 0
        self.play_mode = False
        self.play_speed = 1
        self.step_count = 0

        # Initialize state
        self.reset_state()

    def setup_gl(self, ctx: moderngl.Context):
        """Setup OpenGL resources"""
        self.state_texture = ctx.texture(
            self.config.grid_size, 4,  # RGBA
            data=None
        )

    def reset_state(self):
        """Reset CA state to initial condition"""
        batch_size = 1  # For visualization
        self.current_state = torch.zeros(
            batch_size,
            self.config.channel_n,
            *self.config.grid_size,
            device=const.device
        )

        # Initialize seed based on mode
        if self.config.seed_mode == "center":
            x = self.config.grid_size[0] // 2
            y = self.config.grid_size[1] // 2
            r = self.config.seed_size
            self.current_state[:, 3:, x - r:x + r, y - r:y + r] = 1.0

        elif self.config.seed_mode == "random":
            # Random seed positions
            for _ in range(self.config.seed_size):
                x = np.random.randint(0, self.config.grid_size[0])
                y = np.random.randint(0, self.config.grid_size[1])
                self.current_state[:, 3:, x, y] = 1.0

        elif self.config.seed_mode == "pattern":
            # Create circular pattern
            center_x = self.config.grid_size[0] // 2
            center_y = self.config.grid_size[1] // 2
            r = self.config.pattern_radius

            y, x = torch.meshgrid(
                torch.arange(self.config.grid_size[0], device=const.device),
                torch.arange(self.config.grid_size[1], device=const.device),
                indexing='ij'
            )
            mask = ((x - center_x) ** 2 + (y - center_y) ** 2 <= r ** 2).float()
            self.current_state[:, 3:] = mask.unsqueeze(0).unsqueeze(0)

        self.step_count = 0
        self.growing_state = None

    def get_target_batch(self) -> torch.Tensor:
        """Generate target patterns for training"""
        targets = torch.zeros(
            self.config.batch_size,
            self.config.channel_n,
            *self.config.grid_size,
            device=const.device
        )

        # Create target patterns
        for i in range(self.config.batch_size):
            # Random position for pattern center
            cx = np.random.randint(self.config.pattern_radius, self.config.grid_size[0] - self.config.pattern_radius)
            cy = np.random.randint(self.config.pattern_radius, self.config.grid_size[1] - self.config.pattern_radius)

            # Create pattern with target RGBA
            y, x = torch.meshgrid(
                torch.arange(self.config.grid_size[0], device=const.device),
                torch.arange(self.config.grid_size[1], device=const.device),
                indexing='ij'
            )
            mask = ((x - cx) ** 2 + (y - cy) ** 2 <= self.config.pattern_radius ** 2).float()

            # Set RGBA channels
            for j, value in enumerate(self.config.target_rgba):
                targets[i, j] = mask * value

        return targets

    def train_step(self, step:TrainingStep) -> Dict[str, float]:
        """Execute single training step"""
        # Generate targets if not provided
        targets = self.get_target_batch()

        # Random number of growth steps
        if self.config.growth_step_mode == "random":
            n_steps = torch.randint(
                self.config.growth_steps_min,
                self.config.growth_steps_max,
                (1,)
            ).item()
        else:
            n_steps = self.config.growth_steps_max

        # Initialize states with noise
        states = torch.randn(
            self.config.batch_size,
            self.config.channel_n,
            *self.config.grid_size,
            device=const.device
        ) * self.config.input_noise

        # Add seed
        states[:, 3:,
        self.config.grid_size[0] // 2,
        self.config.grid_size[1] // 2] = 1.0

        # Growth steps
        for step in range(n_steps):
            states = self.tissue(states)

        # Compute loss on RGBA channels
        loss = F.mse_loss(states[:, :4], targets[:, :4])

        if self.physics is not None:
            self.physics.zero_grad()
            loss.backward()
            self.physics.step()

        return {
            'loss':         loss.item(),
            'batch_size':   self.config.batch_size,
            'growth_steps': n_steps
        }

    def step_viz_state(self):
        """Step the visualization state forward"""
        if self.current_state is not None:
            with torch.no_grad():
                self.current_state = self.tissue(self.current_state)
                self.step_count += 1

    def update_state_texture(self):
        """Update OpenGL texture from current state"""
        if self.current_state is None or self.state_texture is None:
            return

        with torch.no_grad():
            # Get visualization data based on mode
            if self.viz_mode == "state":
                # Show RGBA channels
                vis = self.current_state[0, :4]

            elif self.viz_mode == "alpha":
                # Show only alpha channel
                vis = torch.cat([
                    self.current_state[0, 3:4].repeat(3, 1, 1),
                    torch.ones_like(self.current_state[0, 0:1])
                ])

            elif self.viz_mode == "rgb":
                # Show RGB channels with full alpha
                vis = torch.cat([
                    self.current_state[0, :3],
                    torch.ones_like(self.current_state[0, 0:1])
                ])

            else:  # "features"
                # Show selected feature channel
                channel = self.selected_feature % self.config.channel_n
                vis = torch.cat([
                    self.current_state[0, channel:channel + 1].repeat(3, 1, 1),
                    torch.ones_like(self.current_state[0, 0:1])
                ])

            # Convert to numpy and update texture
            vis_np = (vis.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
            self.state_texture.write(vis_np.tobytes())

    def gui(self):
        """Render Petri controls and visualization"""
        imgui.begin("Neural CA Controls")

        # Seed controls
        if imgui.button("Reset"):
            self.reset_state()

        imgui.same_line()
        changed, self.play_mode = imgui.checkbox("Play", self.play_mode)

        if self.play_mode:
            imgui.same_line()
            changed, self.play_speed = imgui.slider_int(
                "Steps per frame", self.play_speed, 1, 10
            )

            # Auto-step when playing
            for _ in range(self.play_speed):
                self.step_viz_state()
        else:
            imgui.same_line()
            if imgui.button("Step"):
                self.step_viz_state()

        # Visualization controls
        imgui.text("Visualization Mode")
        viz_modes = ["state", "alpha", "rgb", "features"]
        changed, current_mode = imgui.combo(
            "##viz_mode",
            viz_modes.index(self.viz_mode),
            viz_modes
        )
        if changed:
            self.viz_mode = viz_modes[current_mode]

        if self.viz_mode == "features":
            changed, self.selected_feature = imgui.slider_int(
                "Feature Channel",
                self.selected_feature,
                0,
                self.config.channel_n - 1
            )

        # Seed configuration
        imgui.separator()
        imgui.text("Seed Configuration")

        seed_modes = ["center", "random", "pattern"]
        changed, current_mode = imgui.combo(
            "Seed Mode",
            seed_modes.index(self.config.seed_mode),
            seed_modes
        )
        if changed:
            self.config.seed_mode = seed_modes[current_mode]
            self.reset_state()

        changed, value = imgui.slider_int(
            "Seed Size",
            self.config.seed_size,
            1, 10
        )
        if changed:
            self.config.seed_size = value
            self.reset_state()

        # Training configuration
        imgui.separator()
        imgui.text("Training Configuration")

        changed, value = imgui.color_edit4(
            "Target RGBA",
            *self.config.target_rgba
        )
        if changed:
            self.config.target_rgba = value

        changed, value = imgui.slider_int(
            "Pattern Radius",
            self.config.pattern_radius,
            5, 30
        )
        if changed:
            self.config.pattern_radius = value

        changed, value = imgui.slider_int(
            "Batch Size",
            self.config.batch_size,
            1, 32
        )
        if changed:
            self.config.batch_size = value

        # Growth steps configuration
        growth_modes = ["random", "fixed"]
        changed, current_mode = imgui.combo(
            "Growth Step Mode",
            growth_modes.index(self.config.growth_step_mode),
            growth_modes
        )
        if changed:
            self.config.growth_step_mode = growth_modes[current_mode]

        changed, value = imgui.drag_int2(
            "Growth Steps (Min/Max)",
            self.config.growth_steps_min,
            self.config.growth_steps_max,
            1.0, 1, 200
        )
        if changed:
            self.config.growth_steps_min = value[0]
            self.config.growth_steps_max = value[1]

        imgui.end()

        # Visualization window
        imgui.begin("CA State")
        if self.state_texture is not None:
            self.update_state_texture()
            imgui.image(
                self.state_texture.glo,
                *self.config.grid_size
            )

            imgui.text(f"Step: {self.step_count}")

        imgui.end()

    def run_inference(self):
        """Run inference from current state"""
        if self.current_state is not None:
            self.growing_state = self.current_state.clone()
            for _ in range(100):  # Run for 100 steps
                self.growing_state = self.tissue(self.growing_state)

__petri__ = PetriMapping(
    tissue_class=NCA,
    config_class=NCATissue,
    petri_class=NCAPetri
)
