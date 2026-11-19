"""Short-window inference for Continuous Intelligence's ci-proto Mamba-3 blocks.

Projection and gating follow https://github.com/continuousintelligence/ci.
"""

import math
from types import MethodType

import torch
import torch.nn.functional as F
from continuousintelligence.ci_proto._mamba import Mamba3Reference, _rotate_pairs
from einops import rearrange


def recurrence(q, k, values, log_alpha, beta, gamma):
    """Evaluate the weighted state contributions through two batched matmuls."""
    batch, length, rank, heads, _ = q.shape
    time = torch.arange(length, device=q.device)
    since = time[:, None] - time[None, :]
    log_prefix = torch.cumsum(log_alpha, dim=1, dtype=torch.float32).transpose(1, 2)
    log_next = torch.cat((log_prefix[..., 1:], log_prefix[..., -1:]), dim=-1)
    beta_next = F.pad(beta[:, 1:], (0, 0, 0, 1)).transpose(1, 2)
    decay = (log_prefix[..., :, None] - log_prefix[..., None, :]).clamp_max(0).exp()
    prev_decay = (log_prefix[..., :, None] - log_next[..., None, :]).clamp_max(0).exp()
    weights = decay * gamma.transpose(1, 2)[..., None, :] * (since >= 0) + prev_decay * beta_next[
        ..., None, :
    ] * (since > 0)
    queries = rearrange(q, "b l r h n -> b h (l r) n")
    keys = rearrange(k, "b l r h n -> b h n (l r)")
    inputs = rearrange(values, "b l r h p -> b h (l r) p")
    weights = weights.repeat_interleave(rank, dim=-1).repeat_interleave(rank, dim=-2)
    mixed = ((queries @ keys) * weights) @ inputs
    return mixed.reshape(batch, heads, length, rank, -1).permute(0, 2, 3, 1, 4)


def forward(self, u):
    """Use the reference projections and apply the closed form of the recurrence."""
    width = self.d_state * self.mimo_rank
    z, x, b, c, dt, a, trap, angles = self.in_proj(u).split(
        [
            self.d_inner,
            self.d_inner,
            width,
            width,
            self.nheads,
            self.nheads,
            self.nheads,
            self.num_rope_angles,
        ],
        dim=-1,
    )
    z = rearrange(z.float(), "b l (h p) -> b l h p", p=self.headdim)
    x = rearrange(x.float(), "b l (h p) -> b l h p", p=self.headdim)
    b = rearrange(b, "b l (r n) -> b l r 1 n", r=self.mimo_rank)
    c = rearrange(c, "b l (r n) -> b l r 1 n", r=self.mimo_rank)
    dt = F.softplus(dt.float() + self.dt_bias)
    log_alpha = (-F.softplus(a.float())).clamp_max(-1e-4) * dt
    gate = trap.float().sigmoid()
    gamma = gate * dt
    beta = (1 - gate) * dt * log_alpha.exp()
    k = self.B_norm(b).float() + rearrange(self.B_bias, "h r n -> r h n")
    q = self.C_norm(c).float() + rearrange(self.C_bias, "h r n -> r h n")
    turns = angles.float().tanh()[:, :, None, :] * dt[..., None] * math.pi
    cumulative = turns.cumsum(dim=1)
    cos = cumulative.cos()[:, :, None]
    sin = cumulative.sin()[:, :, None]
    q = _rotate_pairs(q, cos, sin, interleaved=not self.is_mimo)
    k = _rotate_pairs(k, cos, sin, interleaved=not self.is_mimo)
    if self.is_mimo:
        values = torch.einsum("blhp,hrp->blrhp", x, self.mimo_x.float())
        gates = torch.einsum("blhp,hrp->blrhp", z, self.mimo_z.float())
    else:
        values, gates = x[:, :, None], z[:, :, None]
    mixed = recurrence(q, k, values, log_alpha, beta, gamma)
    mixed = (mixed + self.D[None, None, :, None] * values) * F.silu(gates)
    if self.is_mimo:
        mixed = torch.einsum("blrhp,hrp->blhp", mixed, self.mimo_o.float())
    else:
        mixed = mixed[:, :, 0]
    return self.out_proj(mixed.flatten(-2).to(u.dtype))


def vectorize_recurrence(model):
    """Use the vectorized recurrence on this instance only."""
    for block in model.modules():
        if isinstance(block, Mamba3Reference):
            block.forward = MethodType(forward, block)
