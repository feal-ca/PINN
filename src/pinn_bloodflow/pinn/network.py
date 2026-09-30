"""The PINN itself: a plain MLP mapping (x, y[, t]) -> (u, v, p).

tanh activations are used throughout rather than ReLU because the momentum
residual needs a clean *second* derivative through the network (the
viscous term). ReLU's second derivative is zero almost everywhere, which
would silently kill the diffusion term's gradient signal during training.
"""

import math

import torch
import torch.nn as nn


class PINN(nn.Module):
    def __init__(
        self,
        in_dim: int = 2,
        hidden_layers: int = 7,
        hidden_units: int = 48,
        fourier_time_freqs: list[float] | None = None,
        input_bounds: list[tuple[float, float]] | None = None,
    ):
        """`fourier_time_freqs`, when given, replaces the raw time input
        (the last of the `in_dim` coordinates) with
        `[sin(2*pi*freq*t), cos(2*pi*freq*t)]` for each frequency `freq`
        (in cycles per unit time -- e.g. `1/period` for the fundamental,
        `2/period` for the second harmonic) in the list, before the first
        linear layer.

        This is Phase 2-specific: a plain MLP has to *discover* a periodic
        function of t purely through tanh nonlinearities and gradient
        descent, which in practice converges toward the cycle-mean first
        and only slowly sharpens the true peak-to-trough amplitude (see
        dev-log 2026-08-28's amplitude-damping finding). Handing the
        network sin/cos at the known cardiac harmonics as inputs instead
        gives it a ready-made oscillatory basis -- and since
        sin(2*pi*freq*(t+T)) = sin(2*pi*freq*t) exactly whenever
        freq = k/T for integer k, the network's output becomes *exactly*
        periodic in t by construction, rather than only approximately so
        via the soft `periodicity_loss` penalty.
        """
        super().__init__()
        # `input_bounds`, when given, holds one (low, high) pair per spatial
        # input. The network rescales those inputs to [-1, 1] as its first
        # operation, so autograd still differentiates with respect to the
        # physical coordinates the PDE is written in.
        self.use_norm = input_bounds is not None
        if self.use_norm:
            lo = torch.tensor([b[0] for b in input_bounds], dtype=torch.float32)
            hi = torch.tensor([b[1] for b in input_bounds], dtype=torch.float32)
            self.register_buffer("norm_lo", lo)
            self.register_buffer("norm_scale", 2.0 / (hi - lo))
        self.fourier_time_freqs = list(fourier_time_freqs) if fourier_time_freqs else None
        effective_in_dim = in_dim
        if self.fourier_time_freqs:
            effective_in_dim = (in_dim - 1) + 2 * len(self.fourier_time_freqs)

        dims = [effective_in_dim] + [hidden_units] * hidden_layers
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_units, 3))  # (u, v, p)
        self.net = nn.Sequential(*layers)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, *coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.fourier_time_freqs:
            *space, t = coords
            time_feats = [fn(t * (freq * 2 * math.pi)) for freq in self.fourier_time_freqs for fn in (torch.sin, torch.cos)]
            inp = torch.cat([*space, *time_feats], dim=-1)
        else:
            inp = torch.cat(coords, dim=-1)
        if self.use_norm:
            n = self.norm_lo.numel()
            inp = torch.cat([(inp[..., :n] - self.norm_lo) * self.norm_scale - 1.0, inp[..., n:]], dim=-1)
        out = self.net(inp)
        return out[..., 0:1], out[..., 1:2], out[..., 2:3]
