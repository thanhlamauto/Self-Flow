import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def l2_normalize_tokens(x, eps=1e-6):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def log_token_magnitudes(x, eps=1e-6):
    return torch.log(x.norm(dim=-1, keepdim=True).clamp_min(eps))


def _make_mlp(dim, hidden_dim, out_dim):
    return nn.Sequential(
        nn.Linear(dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, out_dim),
    )


class ShortcutBlock(nn.Module):
    def __init__(self, width, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        hidden = int(width * mlp_ratio)
        self.norm = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, width),
        )
        self.gamma = nn.Parameter(torch.zeros(width))

    def forward(self, x):
        return x + self.gamma * self.mlp(self.norm(x))


@dataclass(frozen=True)
class PredictorConfig:
    width: int
    num_blocks: int
    mlp_ratio: float


def predictor_config_from_name(name, hidden_size):
    normalized = str(name).lower().replace("-", "_")
    if normalized in {"none", "off", "false", "0"}:
        return None
    if normalized in {"hybrid_deep_12pct", "hybrid_deep_mdm_12pct"}:
        return PredictorConfig(width=180, num_blocks=6, mlp_ratio=4.0)
    if normalized == "hybrid_deep_10":
        return PredictorConfig(width=hidden_size, num_blocks=10, mlp_ratio=4.0)
    if normalized == "hybrid_deep":
        return PredictorConfig(width=hidden_size, num_blocks=8, mlp_ratio=4.0)
    if normalized in {"tiny", "small", "base", "large"}:
        blocks = {"tiny": 2, "small": 4, "base": 6, "large": 8}[normalized]
        width = max(hidden_size // 2, 128) if normalized in {"tiny", "small"} else hidden_size
        return PredictorConfig(width=width, num_blocks=blocks, mlp_ratio=4.0)
    raise ValueError(f"Unknown shortcut predictor variant: {name!r}")


class DepthShortcutPredictor(nn.Module):
    """PyTorch depth shortcut predictor for MDM hidden states.

    Inputs and outputs use MDM's sequence-first hidden layout: [frames, batch, dim].
    The predictor learns a transition from one transformer depth to another and
    emits both a direction vector and a normalized log-magnitude delta.
    """

    def __init__(
        self,
        hidden_size,
        depth,
        variant="hybrid_deep_12pct",
        gamma_out_init=0.001,
        mag_abs_center=2.9,
        mag_abs_scale=0.6,
        use_timestep=True,
    ):
        super().__init__()
        cfg = predictor_config_from_name(variant, hidden_size)
        if cfg is None:
            raise ValueError("DepthShortcutPredictor requires a non-empty variant.")
        self.hidden_size = hidden_size
        self.depth = depth
        self.width = cfg.width
        self.mag_abs_center = float(mag_abs_center)
        self.mag_abs_scale = float(mag_abs_scale)
        self.use_timestep = bool(use_timestep)

        self.in_proj = nn.Linear(hidden_size, self.width)
        self.layer_embed = nn.Embedding(max(depth, 1), self.width)
        self.gap_embed = nn.Embedding(max(depth, 1), self.width)
        self.time_proj = _make_mlp(hidden_size, self.width, self.width)
        self.mag_proj = nn.Linear(1, self.width)
        self.blocks = nn.ModuleList(
            [ShortcutBlock(self.width, mlp_ratio=cfg.mlp_ratio) for _ in range(cfg.num_blocks)]
        )
        self.out_norm = nn.LayerNorm(self.width)
        self.dir_head = nn.Linear(self.width, hidden_size)
        self.mag_head = nn.Linear(self.width, 1)
        nn.init.zeros_(self.dir_head.weight)
        nn.init.constant_(self.dir_head.bias, 0.0)
        nn.init.constant_(self.mag_head.bias, 0.0)
        self.gamma_out = nn.Parameter(torch.tensor(float(gamma_out_init)))

    def forward(
        self,
        source_hidden,
        source_layer,
        target_layer,
        time_embed=None,
        source_log_mag=None,
    ):
        frames, batch, _ = source_hidden.shape
        device = source_hidden.device
        source_layer = int(source_layer)
        target_layer = int(target_layer)
        gap = max(min(target_layer - source_layer, self.depth - 1), 0)

        x = self.in_proj(source_hidden)
        layer_ids = torch.tensor([source_layer, target_layer], device=device, dtype=torch.long)
        cond = self.layer_embed(layer_ids).sum(dim=0) + self.gap_embed(
            torch.tensor(gap, device=device, dtype=torch.long)
        )
        x = x + cond.view(1, 1, -1)

        if self.use_timestep and time_embed is not None:
            if time_embed.dim() == 3:
                time_embed = time_embed.squeeze(0)
            x = x + self.time_proj(time_embed).view(1, batch, -1)

        if source_log_mag is not None:
            mag_cond = (source_log_mag - self.mag_abs_center) / max(self.mag_abs_scale, 1e-6)
            x = x + self.mag_proj(mag_cond)

        for block in self.blocks:
            x = block(x)
        x = self.out_norm(x)
        direction = source_hidden + self.gamma_out * self.dir_head(x)
        delta_mag = torch.tanh(self.mag_head(x))
        return direction, delta_mag


def sample_layer_pairs(
    num_layers,
    num_pairs=1,
    max_gap=None,
    gap_loc=2.0,
    gap_sigma=1.5,
    mode="trunc_normal_centered",
    device=None,
):
    if num_layers < 2:
        return []
    max_gap = int(max_gap or (num_layers - 1))
    max_gap = max(1, min(max_gap, num_layers - 1))
    pairs = []
    for _ in range(int(num_pairs)):
        if mode in {"trunc_normal", "trunc_normal_centered"}:
            gap = int(round(torch.normal(
                mean=torch.tensor(float(gap_loc), device=device),
                std=torch.tensor(float(max(gap_sigma, 1e-6)), device=device),
            ).item()))
            gap = max(1, min(max_gap, gap))
        else:
            gap = int(torch.randint(1, max_gap + 1, (), device=device).item())
        if mode.endswith("centered"):
            center = (num_layers - 1) / 2.0
            source = int(round(center - gap / 2.0))
            source = max(0, min(num_layers - gap - 1, source))
        else:
            source = int(torch.randint(0, num_layers - gap, (), device=device).item())
        pairs.append((source, source + gap))
    return pairs


def private_activation_loss(
    activations,
    max_pairs=0,
    use_residual=True,
    cosine_mode="bnd",
    pair_mode="random",
    eps=1e-8,
):
    """Common/private activation penalty over [layers, frames, batch, dim]."""
    acts = activations.float().permute(0, 2, 1, 3)
    if use_residual:
        acts_for_common = l2_normalize_tokens(acts, eps=eps)
        common = acts_for_common.mean(dim=0)
        private = acts_for_common - common.detach().unsqueeze(0)
    else:
        common = acts.mean(dim=0)
        private = acts

    if cosine_mode == "bnd":
        normed = F.normalize(private.reshape(private.shape[0], -1), dim=-1, eps=eps)
        cos = normed @ normed.t()
        i, j = torch.triu_indices(private.shape[0], private.shape[0], offset=1, device=acts.device)
        pair_cos = cos[i, j]
        pair_sq = pair_cos.square()
    elif cosine_mode == "nd":
        normed = F.normalize(private.reshape(private.shape[0], private.shape[1], -1), dim=-1, eps=eps)
        cos = torch.einsum("lbd,mbd->lmb", normed, normed)
        i, j = torch.triu_indices(private.shape[0], private.shape[0], offset=1, device=acts.device)
        pair_cos = cos[i, j].mean(dim=-1)
        pair_sq = cos[i, j].square().mean(dim=-1)
    elif cosine_mode == "token":
        normed = F.normalize(private, dim=-1, eps=eps)
        cos = torch.einsum("lbtd,mbtd->lmbt", normed, normed)
        i, j = torch.triu_indices(private.shape[0], private.shape[0], offset=1, device=acts.device)
        pair_cos = cos[i, j].mean(dim=(-1, -2))
        pair_sq = cos[i, j].square().mean(dim=(-1, -2))
    else:
        raise ValueError(f"Unknown private cosine mode: {cosine_mode!r}")

    if pair_sq.numel() == 0:
        zero = activations.new_tensor(0.0)
        return zero, zero, zero, zero
    if max_pairs and max_pairs > 0 and max_pairs < pair_sq.numel():
        if pair_mode == "random":
            idx = torch.randperm(pair_sq.numel(), device=acts.device)[:max_pairs]
        else:
            idx = torch.arange(max_pairs, device=acts.device)
        pair_sq = pair_sq[idx]
        pair_cos = pair_cos[idx]
    loss = pair_sq.mean()
    common_norm = common.reshape(common.shape[0], -1).norm(dim=-1).mean()
    private_norm = private.reshape(private.shape[0], private.shape[1], -1).norm(dim=-1).mean()
    return loss, common_norm, private_norm, pair_cos.mean()
