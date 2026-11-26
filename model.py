import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, dim, scale=1000.0):
    """Sinusoidal embedding of noise amounts t in [0, 1], shape (B,) -> (B, dim).
    t is scaled up so the sinusoids distinguish nearby noise levels."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = (t.float() * scale)[:, None] * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.t_proj = nn.Linear(t_dim, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.t_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class DiffusionModel(nn.Module):
    """UNet for the "corrupt" (rectified flow) approach. The noisy image is
    x_t = (1 - t) * x0 + t * noise, and the model predicts the velocity
    v = noise - x0, conditioned on the noise amount t in [0, 1]."""

    def __init__(self, in_ch=3, base_ch=64, ch_mults=(1, 2, 4, 4)):
        super().__init__()
        self.in_ch = in_ch
        self.base_ch = base_ch
        t_dim = base_ch * 4
        self.t_mlp = nn.Sequential(nn.Linear(base_ch, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim))
        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        chs = [base_ch * m for m in ch_mults]
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev = base_ch
        for i, ch in enumerate(chs):
            self.downs.append(ResBlock(prev, ch, t_dim))
            last = i == len(chs) - 1
            self.pools.append(nn.Identity() if last else nn.Conv2d(ch, ch, 3, stride=2, padding=1))
            prev = ch

        self.mid = ResBlock(prev, prev, t_dim)

        self.ups = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for i, ch in reversed(list(enumerate(chs))):
            self.ups.append(ResBlock(prev + ch, ch, t_dim))
            self.upsamples.append(
                nn.Identity() if i == 0 else nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(ch, ch, 3, padding=1),
                )
            )
            prev = ch

        self.out = nn.Sequential(
            nn.GroupNorm(8, chs[0]), nn.SiLU(), nn.Conv2d(chs[0], in_ch, 3, padding=1),
        )

    def forward(self, x, t):
        """x: noisy images (B, C, H, W); t: noise amounts (B,) in [0, 1]."""
        t_emb = self.t_mlp(timestep_embedding(t, self.base_ch))
        h = self.in_conv(x)
        skips = []
        for block, pool in zip(self.downs, self.pools):
            h = block(h, t_emb)
            skips.append(h)
            h = pool(h)
        h = self.mid(h, t_emb)
        for block, up in zip(self.ups, self.upsamples):
            h = up(block(torch.cat([h, skips.pop()], dim=1), t_emb))
        return self.out(h)

    @torch.no_grad()
    def sample(self, noise, steps=20):
        """Euler integration from t=1 (pure noise) to t=0 (clean image).
        Since dx_t/dt = noise - x0 = v, each step moves x by -v * dt."""
        x = noise
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((x.shape[0],), 1.0 - i * dt, device=x.device)
            x = x - self(x, t) * dt
        return x.clamp(-1, 1)
