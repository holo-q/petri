import math

import torch
from torch.optim import Optimizer
from typing import List, Dict, Optional
import numpy as np

import const

class CronkleOptimizerDraft(Optimizer):
    def __init__(self,
                 params,
                 lr: float = 1.0,
                 pattern_size: int = 4,
                 noise_decay: float = 0.9,
                 momentum_beta: float = 0.9,
                 reference_points: Optional[Dict[str, float]] = None):
        """
        Cronkle optimizer using bisection dynamics and pattern-based reference points.

        Args:
            params: Iterable of parameters to optimize
            lr: Learning rate scale factor
            pattern_size: Size of repeating dynamic pattern
            noise_decay: Rate at which noise influence decreases
            momentum_beta: Momentum coefficient
            reference_points: Optional custom reference points
        """
        if not 0.0 <= momentum_beta < 1.0:
            raise ValueError(f"Invalid momentum_beta: {momentum_beta}")

        defaults = dict(
            lr=lr,
            noise_decay=noise_decay,
            momentum_beta=momentum_beta,
            pattern_size=pattern_size,
            step_count=0
        )
        super().__init__(params, defaults)

        # Initialize reference points for different dynamics
        self.ref_points = reference_points or {
            'V':   [0.0, 0.5, 1.0],
            'W':   [0.0, 0.25, 0.5, 0.75, 1.0],
            'phi': [0.0, 0.381966, 0.618034, 1.0],  # Golden ratio based
            'e':   [0.0, 0.367879, 0.632121, 1.0],  # e based
        }

        # Create tiled pattern
        self.pattern = self._create_dynamic_pattern(pattern_size)
        self._initialize_state()

    def _create_dynamic_pattern(self, size: int) -> List[List[str]]:
        """Create a tiled pattern of dynamics."""
        basic_pattern = ['V', 'W', 'phi', 'e']
        pattern = []
        for i in range(size):
            row = []
            for j in range(size):
                idx = (i + j) % len(basic_pattern)
                row.append(basic_pattern[idx])
            pattern.append(row)
        return pattern

    def _initialize_state(self):
        """Initialize optimizer state for all parameter groups."""
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['momentum'] = torch.zeros_like(p)
                state['noise_scale'] = 1.0

                # Assign pattern-based reference points to each parameter
                shape = p.shape
                if len(shape) >= 2:
                    # For 2D+ tensors, tile the pattern
                    state['dynamics'] = self._assign_dynamics(shape)
                else:
                    # For 1D tensors, use simple repeating pattern
                    state['dynamics'] = [self.pattern[0][i % len(self.pattern[0])]
                                         for i in range(shape[0])]

    def _assign_dynamics(self, shape):
        """Assign dynamics pattern to parameter tensor."""
        rows, cols = shape[-2:]
        dynamics = []
        for i in range(rows):
            row = []
            for j in range(cols):
                pattern_i = i % len(self.pattern)
                pattern_j = j % len(self.pattern[0])
                row.append(self.pattern[pattern_i][pattern_j])
            dynamics.append(row)
        return dynamics

    def _compute_bisection_target(self, param_val, grad, dynamic_type, noise):
        """Compute bisection target based on dynamics pattern."""
        refs = self.ref_points[dynamic_type]

        # Add noise to reference points
        noisy_refs = [r + noise * np.random.randn() for r in refs]

        # Find closest upper and lower reference points
        sorted_refs = sorted(noisy_refs)
        upper_ref = min((r for r in sorted_refs if r > param_val), default=1.0)
        lower_ref = max((r for r in sorted_refs if r < param_val), default=0.0)

        # Choose target based on gradient direction
        return upper_ref if grad > 0 else lower_ref

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum_beta = group['momentum_beta']

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                state['step_count'] = state.get('step_count', 0) + 1

                # Update noise scale
                state['noise_scale'] *= group['noise_decay']
                noise = state['noise_scale'] * 0.1  # Base noise magnitude

                # Get gradient and momentum
                grad = p.grad
                momentum = state['momentum']
                momentum.mul_(momentum_beta).add_(grad, alpha=1 - momentum_beta)

                # Compute updates based on pattern
                if len(p.shape) >= 2:
                    # For 2D+ tensors
                    for i in range(p.shape[-2]):
                        for j in range(p.shape[-1]):
                            dynamic = state['dynamics'][i][j]
                            target = self._compute_bisection_target(
                                p[..., i, j].item(),
                                momentum[..., i, j].item(),
                                dynamic,
                                noise
                            )
                            p[..., i, j].add_((target - p[..., i, j]) * lr)
                else:
                    # For 1D tensors
                    for i in range(p.shape[0]):
                        dynamic = state['dynamics'][i]
                        target = self._compute_bisection_target(
                            p[i].item(),
                            momentum[i].item(),
                            dynamic,
                            noise
                        )
                        p[i].add_((target - p[i]) * lr)

        return loss

from typing import Optional, List, Dict
from dataclasses import dataclass
import math
import torch
from torch.optim import Optimizer
from torch import Tensor

@dataclass
class CBDState:
    """State for Cronkle-Bisection Descent parameters"""
    momentum: Tensor
    pattern: Tensor
    running_grad_norm: Optional[Tensor] = None
    running_param_norm: Optional[Tensor] = None
    step_count: int = 0

class CBD(Optimizer):
    """
    Cronkle-Bisection Descent with enhanced stability features.

    Combines bisection descent with Adam optimization using dynamic control.
    Implements tissue-growth like parameter evolution with mathematical attractors.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.9,
        noise_scale: float = 0.01,
        adam_control: float = 0.0,
        noise_decay: float = 0.1,
        eps: float = 1e-8,
        clip_value: float = 100.0
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            noise_scale=noise_scale,
            adam_control=adam_control,
            noise_decay=noise_decay,
            eps=eps,
            clip_value=clip_value
        )
        super().__init__(params, defaults)

        # Mathematical constants as attractor points
        self.ref_points = torch.tensor([
            0.618033988749895,  # φ (golden ratio)
            0.367879441171442,  # 1/e
            0.318309886183791  # 1/π
        ])

        self.step_count = 0

    def _init_state(self, p: Tensor) -> CBDState:
        """Initialize optimizer state for a parameter tensor"""
        return CBDState(
            momentum=torch.zeros_like(p),
            pattern=self._generate_full_pattern(p.shape),
            running_grad_norm=torch.zeros(1, device=p.device),
            running_param_norm=torch.zeros(1, device=p.device)
        )

    def _safe_norm(self, tensor: Tensor, eps: float = 1e-8) -> Tensor:
        """Safely compute norm with epsilon"""
        return torch.sqrt(torch.sum(tensor * tensor) + eps)

    def _apply_update(self, p: Tensor, update: Tensor, clip_value: float):
        """Safely apply parameter update with checks"""
        if torch.isnan(update).any() or torch.isinf(update).any():
            return False

        norm_update = self._safe_norm(update)
        if norm_update > clip_value:
            update = update * (clip_value / norm_update)

        p.data.add_(update)
        return True

    def step(self, closure=None):
        """Performs a single optimization step with enhanced stability"""
        loss = None
        if closure is not None:
            loss = closure()

        # Ensure ref points are on correct device
        self.ref_points = self.ref_points.to(next(iter(self.param_groups[0]['params'])).device)

        # Dynamic control based on training progress
        wave = 0.5 * (1 + math.sin(self.step_count / 1000))

        for group in self.param_groups:
            # Extract hyperparameters
            lr = group['lr']
            momentum = group['momentum']
            noise_scale = group['noise_scale']
            adam_control = group['adam_control']
            noise_decay = group['noise_decay']
            eps = group['eps']
            clip_value = group['clip_value']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                w = p.data

                # Check for invalid gradients
                if torch.isnan(grad).any() or torch.isinf(grad).any():
                    print(f"Warning: Found NaN/Inf in gradients")
                    continue

                # Get or initialize state
                state = self.state[p]
                if len(state) == 0:
                    state.update(self._init_state(p).__dict__)

                # Update running norms for monitoring
                state['running_grad_norm'] = (
                    0.9 * state['running_grad_norm'] +
                    0.1 * self._safe_norm(grad)
                )
                state['running_param_norm'] = (
                    0.9 * state['running_param_norm'] +
                    0.1 * self._safe_norm(p.data)
                )

                # Get reference points for parameters
                refs = self.ref_points[state['pattern']].to(p.device)

                # Compute bisection update
                dir = torch.sign(grad)
                targets = torch.where(
                    ((dir > 0) & (p < refs)) | ((dir < 0) & (p > refs)),
                    refs,
                    torch.where(dir > 0,
                        torch.ones_like(refs),
                        torch.zeros_like(refs))
                )

                # Scale update by gradient magnitude with stability
                grad_mag = grad.abs().clamp(min=eps)
                bisect_delta = dir * (targets - w) / 2

                # Compute Adam-like update with momentum
                mom = state['momentum']
                mom.mul_(momentum).add_(grad)
                adam_update = -lr * mom

                # Blend updates with wave-based control
                blend = adam_control * wave
                update = (blend * bisect_delta + (1 - blend) * adam_update)
                # update = bisect_delta + adam_update
                #
                # Decaying noise injection
                # noise_factor = noise_scale * (noise_decay ** (self.step_count / 1000))
                noise = torch.randn_like(p) * noise_scale
                update = update + noise

                print(f'delta max: {torch.max(update)}, min: {torch.min(update)}')

                # Safe parameter update
                success = self._apply_update(p, update, clip_value)
                if not success:
                    print("Warning: Update failed due to numerical instability")

                state['step_count'] += 1

        self.step_count += 1
        return loss

    def _generate_full_pattern(self, shape: tuple) -> Tensor:
        """Generate a pattern of indices for the full tensor shape"""
        coords = [torch.arange(s) for s in shape]
        meshgrids = torch.meshgrid(*coords, indexing='ij')
        coord_sum = sum(grid for grid in meshgrids)
        return coord_sum % 3

    def get_state_info(self) -> Dict[str, float]:
        """Get diagnostic information about optimizer state"""
        info = {}
        for group in self.param_groups:
            for p in group['params']:
                if p in self.state:
                    state = self.state[p]
                    info['grad_norm'] = state['running_grad_norm'].item()
                    info['param_norm'] = state['running_param_norm'].item()
                    break
        return info
