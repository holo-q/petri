import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Type

import torch
from torch import Tensor
from torch.optim import Optimizer

from core import PhysicsConfig

@dataclass
class DivineAdamState:
    """State for Divine Adam parameters incorporating tissue growth"""
    exp_avg: Tensor  # First moment estimate (momentum)
    exp_avg_sq: Tensor  # Second moment estimate (variance)
    pattern: Tensor  # Resonance pattern for divine ratios
    running_grad_norm: Optional[Tensor] = None
    running_param_norm: Optional[Tensor] = None
    max_exp_avg_sq: Optional[Tensor] = None  # For AMSGrad variant
    step_count: int = 0

class DivineAdam(Optimizer):
    """
    Adam optimizer enhanced with divine ratio alignment and tissue growth dynamics.

    Combines Adam's adaptive moments with mathematical attractor points,
    implementing tissue-growth parameter evolution guided by data gradients.
    """

    def __init__(
        self,
        params,
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        amsgrad: bool = False,
        noise_scale: float = 0.01,
        noise_decay: float = 0.995,
        divine_momentum: float = 0.1,
        clip_value: float = 100.0
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1 parameter: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2 parameter: {betas[1]}")

        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            amsgrad=amsgrad, noise_scale=noise_scale,
            noise_decay=noise_decay, divine_momentum=divine_momentum,
            clip_value=clip_value
        )
        super().__init__(params, defaults)

        # Divine ratio attractor points (normalized to [0,1])
        self.ref_points = torch.tensor([
            0.618033988749895,  # φ (golden ratio conjugate)
            0.367879441171442,  # 1/e
            0.318309886183791,  # 1/π
            0.707106781186547,  # 1/√2
            0.693147180559945,  # ln(2)
            0.786151377757423  # √φ
        ])

        self.step_count = 0

    def _init_state(self, p: Tensor) -> DivineAdamState:
        """Initialize optimizer state for a parameter tensor"""
        state = DivineAdamState(
            exp_avg=torch.zeros_like(p),
            exp_avg_sq=torch.zeros_like(p),
            pattern=self._generate_resonance_pattern(p.shape),
            running_grad_norm=torch.zeros(1, device=p.device),
            running_param_norm=torch.zeros(1, device=p.device),
            max_exp_avg_sq=torch.zeros_like(p) if self.defaults['amsgrad'] else None
        )
        return state

    def _safe_norm(self, tensor: Tensor, eps: float = 1e-8) -> Tensor:
        """Safely compute norm with epsilon"""
        return torch.sqrt(torch.sum(tensor * tensor) + eps)

    def _generate_resonance_pattern(self, shape: tuple) -> Tensor:
        """Generate a resonance pattern using mathematical constants"""
        coords = [torch.arange(s) for s in shape]
        meshgrids = torch.meshgrid(*coords, indexing='ij')

        # Create phase shifts based on tensor dimensions
        phases = [(grid + 1) * math.pi / 6 for grid in meshgrids]

        # Combine phases with trigonometric functions
        pattern = sum(torch.sin(phase) * torch.cos(phase) for phase in phases)

        # Map to indices for ref_points
        return (pattern.abs() * (len(self.ref_points) - 1)).long() % len(self.ref_points)

    def _compute_divine_update(
        self,
        p: Tensor,
        grad: Tensor,
        state: Dict,
        group: Dict
    ) -> Tensor:
        """Compute divine gradient with harmonic noise to break grid patterns"""
        refs = self.ref_points.to(p.device)
        p_flat = p.reshape(-1)

        # Convert step count to tensor and move to correct device
        t = torch.tensor(state['step_count'], device=p.device, dtype=torch.float32)

        # Add harmonic noise to break up grid patterns
        harmonic_noise = (
                             torch.sin(t * 0.01) * torch.sin(p_flat * 2.718) +  # e-based frequency
                             torch.sin(t * 0.02) * torch.sin(p_flat * 3.141) +  # π-based frequency
                             torch.sin(t * 0.03) * torch.sin(p_flat * 1.618)  # φ-based frequency
                         ).unsqueeze(1) * 0.1  # Scale the noise

        # Distance with harmonic perturbation
        distances = torch.abs(p_flat.unsqueeze(1) - refs.unsqueeze(0))
        distances = distances + harmonic_noise

        # Temperature that oscillates with golden ratio
        base_temp = 0.5
        temp = base_temp * (1 + 0.2 * torch.sin(t * 0.618))

        # Softer attraction with oscillating bias
        attractions = torch.softmax(-(distances) / temp, dim=1)

        # Add spiral dynamics to direction field
        directions = p_flat.unsqueeze(1) - refs.unsqueeze(0)
        spiral = torch.stack([
            torch.cos(t * 0.01) * directions,
            torch.sin(t * 0.01) * directions
        ], dim=-1).mean(dim=-1)

        divine_grad = (attractions * spiral).sum(dim=1)
        divine_grad = divine_grad.reshape(p.shape)

        # Very gentle influence that pulses with golden ratio
        grad_norm = torch.norm(grad.reshape(-1)) + 1e-8
        divine_norm = torch.norm(divine_grad.reshape(-1)) + 1e-8

        # Strength oscillates with multiple frequencies
        base_strength = 0.01 * (
            0.5 + 0.2 * torch.sin(t * 0.618) +  # φ
            0.1 * torch.sin(t * 0.367) +  # 1/e
            0.1 * torch.sin(t * 0.318)  # 1/π
        )

        scale_factor = (grad_norm * base_strength) / divine_norm
        divine_grad = divine_grad * scale_factor

        return -divine_grad

    def step(self, closure=None):
        """Performs a single optimization step with divine guidance"""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('DivineAdam does not support sparse gradients')

                # Get or initialize state
                state = self.state[p]
                if len(state) == 0:
                    state.update(self._init_state(p).__dict__)

                # Get hyperparameters
                lr = group['lr']
                beta1, beta2 = group['betas']
                eps = group['eps']
                weight_decay = group['weight_decay']
                amsgrad = group['amsgrad']

                # Apply weight decay
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                # Update running norms
                state['running_grad_norm'] = (
                    0.9 * state['running_grad_norm'] +
                    0.1 * self._safe_norm(grad)
                )
                state['running_param_norm'] = (
                    0.9 * state['running_param_norm'] +
                    0.1 * self._safe_norm(p.data)
                )

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']

                # Update biased first and second moment estimates
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                if amsgrad:
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = max_exp_avg_sq.sqrt().add_(eps)
                else:
                    denom = exp_avg_sq.sqrt().add_(eps)

                # Bias correction
                step = state['step_count'] + 1
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1

                # Add divine influence to gradient
                divine_grad = self._compute_divine_update(p, grad, state, group)
                total_grad = grad + divine_grad

                # Update biased first and second moment estimates with combined gradient
                exp_avg.mul_(beta1).add_(total_grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(total_grad, total_grad, value=1 - beta2)

                if amsgrad:
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = max_exp_avg_sq.sqrt().add_(eps)
                else:
                    denom = exp_avg_sq.sqrt().add_(eps)

                # Apply Adam update with divine-influenced moments
                update = -step_size * exp_avg / denom
                p.data.add_(update)

                # Add decaying noise for exploration
                noise_factor = (
                    group['noise_scale'] *
                    (group['noise_decay'] ** (self.step_count / 33))
                )
                noise = torch.randn_like(p) * noise_factor
                update = update + noise

                # Clip update if necessary
                update_norm = self._safe_norm(update)
                if update_norm > group['clip_value']:
                    update = update * (group['clip_value'] / update_norm)

                # Apply update
                p.data.add_(update)
                state['step_count'] += 1

        self.step_count += 1
        return loss
    #
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

class DivineAdamPhysics(PhysicsConfig):
    lr: float = 0.002
    betas: Tuple[float, float] = (0.9, 0.999)
    noise_scale: float = 0.01
    noise_decay: float = 0.995
    divine_momentum: float = 0.0
    clip_value: float = 100.0

    def get_class(self) -> Type[torch.optim.Optimizer]:
        return DivineAdam
