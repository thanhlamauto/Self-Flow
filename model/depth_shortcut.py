import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


def l2_normalize_tokens(x, eps=1e-6):
    x = x.float()
    return x / (torch.linalg.norm(x, dim=-1, keepdim=True) + eps)


def log_token_magnitudes(x, eps=1e-6):
    return torch.log(torch.linalg.norm(x.float(), dim=-1, keepdim=True) + eps)


def build_predictor_source(hidden, normalize_input=True, eps=1e-6):
    directions = l2_normalize_tokens(hidden, eps=eps)
    magnitudes = log_token_magnitudes(hidden, eps=eps)
    return (directions if normalize_input else hidden.float()), magnitudes


def magnitude_input_features(m_source, abs_center=5.5, abs_scale=1.5, key_padding_mask=None, eps=1e-6):
    m_source = m_source.float()
    m_abs = (m_source - float(abs_center)) / float(abs_scale)
    if key_padding_mask is None:
        token_mean = m_source.mean(dim=1, keepdim=True)
        token_std = m_source.std(dim=1, keepdim=True, unbiased=False)
    else:
        valid = (~key_padding_mask).to(device=m_source.device, dtype=m_source.dtype)[:, :, None]
        denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        token_mean = (m_source * valid).sum(dim=1, keepdim=True) / denom
        token_var = ((m_source - token_mean).square() * valid).sum(dim=1, keepdim=True) / denom
        token_std = torch.sqrt(token_var)
    m_spatial = (m_source - token_mean) / (token_std + eps)
    features = torch.cat([m_abs, m_spatial], dim=-1)
    if key_padding_mask is not None:
        features = features.masked_fill(key_padding_mask[:, :, None].to(device=features.device), 0.0)
    return features


def modulate(x, shift, scale):
    return x * (1.0 + scale[:, None, :]) + shift[:, None, :]


def zero_init(module):
    nn.init.zeros_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    return module


class DeepHybridShortcutBlock(nn.Module):
    """1D MDM version of the hybrid deep shortcut block.

    The image branch used 2D depthwise convolutions over a patch grid. MDM uses a
    temporal token sequence, so this block keeps the same conditional residual
    structure but applies depthwise Conv1d over the motion/time-token axis.
    """

    def __init__(self, width, mlp_ratio=4.0, dilation=1, use_attention=False, num_heads=4, adaln_zero=True):
        super().__init__()
        self.width = width
        self.use_attention = use_attention
        self.num_heads = num_heads

        self.conv_ln = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.conv_adaln = nn.Linear(width, 3 * width)
        self.dwconv = nn.Conv1d(width, width, kernel_size=3, padding=dilation, dilation=dilation, groups=width)
        self.pwconv = nn.Linear(width, width)

        if self.use_attention:
            self.attn_ln = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
            self.attn_adaln = nn.Linear(width, 3 * width)
            self.attn = nn.MultiheadAttention(width, num_heads, batch_first=False)
        else:
            self.attn_ln = None
            self.attn_adaln = None
            self.attn = None

        self.mlp_ln = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.mlp_adaln = nn.Linear(width, 3 * width)
        self.mlp = nn.Sequential(
            nn.Linear(width, int(width * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(width * mlp_ratio), width),
        )

        if adaln_zero:
            zero_init(self.conv_adaln)
            if self.attn_adaln is not None:
                zero_init(self.attn_adaln)
            zero_init(self.mlp_adaln)
        zero_init(self.mlp[-1])

    def _adaln(self, layer, cond):
        return layer(F.silu(cond)).chunk(3, dim=-1)

    def _zero_padded_tokens(self, h, key_padding_mask):
        if key_padding_mask is None:
            return h
        return h.masked_fill(key_padding_mask[:, :, None].to(device=h.device), 0.0)

    def _temporal_conv(self, x):
        if x.shape[1] <= 1:
            return x
        cond_token = x[:, :1]
        motion_tokens = x[:, 1:]
        motion_tokens = self.dwconv(motion_tokens.transpose(1, 2)).transpose(1, 2)
        return torch.cat([cond_token, motion_tokens], dim=1)

    def forward(self, h, cond, key_padding_mask=None):
        h = self._zero_padded_tokens(h, key_padding_mask)
        shift, scale, gate = self._adaln(self.conv_adaln, cond)
        x = modulate(self.conv_ln(h), shift, scale)
        x = self._temporal_conv(x)
        x = self.pwconv(x)
        h = h + gate[:, None, :] * x
        h = self._zero_padded_tokens(h, key_padding_mask)

        if self.use_attention:
            shift, scale, gate = self._adaln(self.attn_adaln, cond)
            x = modulate(self.attn_ln(h), shift, scale)
            x = x.transpose(0, 1)
            attn_out, _ = self.attn(
                x,
                x,
                x,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            h = h + gate[:, None, :] * attn_out.transpose(0, 1)
            h = self._zero_padded_tokens(h, key_padding_mask)

        shift, scale, gate = self._adaln(self.mlp_adaln, cond)
        x = modulate(self.mlp_ln(h), shift, scale)
        h = h + gate[:, None, :] * self.mlp(x)
        h = self._zero_padded_tokens(h, key_padding_mask)
        return h


class MagnitudeHead(nn.Module):
    def __init__(self, width, mag_abs_center=5.5, mag_abs_scale=1.5):
        super().__init__()
        self.mag_abs_center = mag_abs_center
        self.mag_abs_scale = mag_abs_scale
        mag_channels = width + 2
        self.ln = nn.LayerNorm(mag_channels, elementwise_affine=False, eps=1e-6)
        self.adaln = nn.Linear(width, 2 * mag_channels)
        self.dwconv = nn.Conv1d(mag_channels, mag_channels, kernel_size=3, padding=1, groups=mag_channels)
        self.pw1 = nn.Linear(mag_channels, max(width // 4, 1))
        self.pw2 = nn.Linear(max(width // 4, 1), 1)
        zero_init(self.adaln)
        zero_init(self.pw2)

    def forward(self, h, m_source, cond, key_padding_mask=None):
        m_features = magnitude_input_features(
            m_source,
            abs_center=self.mag_abs_center,
            abs_scale=self.mag_abs_scale,
            key_padding_mask=key_padding_mask,
        )
        x = torch.cat([h, m_features.to(dtype=h.dtype)], dim=-1)
        gamma, beta = self.adaln(F.gelu(cond)).chunk(2, dim=-1)
        x = self.ln(x)
        x = x * (1.0 + gamma[:, None, :]) + beta[:, None, :]
        x = self.dwconv(x.transpose(1, 2)).transpose(1, 2)
        x = F.gelu(self.pw1(x))
        out = torch.tanh(self.pw2(x))
        if key_padding_mask is not None:
            out = out.masked_fill(key_padding_mask[:, :, None].to(device=out.device), 0.0)
        return out


class DepthShortcutPredictor(nn.Module):
    """Predict target-layer hidden directions and log-magnitude deltas."""

    def __init__(
        self,
        hidden_size,
        depth,
        width=192,
        num_blocks=10,
        mlp_ratio=2.0,
        dilation_schedule=None,
        num_heads=4,
        cond_dim=None,
        residual_output=True,
        attention_every=4,
        adaln_zero=True,
        gamma_out_init=0.001,
        mag_abs_center=5.5,
        mag_abs_scale=1.5,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.depth = depth
        self.width = width
        self.num_blocks = num_blocks
        self.residual_output = residual_output
        self.attention_every = attention_every
        cond_dim = int(cond_dim or width)

        self.in_proj = nn.Linear(hidden_size, width)
        self.cond_t_proj = nn.Linear(hidden_size, cond_dim)
        self.cond_layer_embed = nn.Embedding(depth + 1, cond_dim)
        self.cond_delta_embed = nn.Embedding(depth, cond_dim)
        self.cond_out = nn.Linear(cond_dim, width)

        if dilation_schedule is None:
            dilation_schedule = (1, 2, 4, 1, 2, 4, 1, 2, 4, 1)[:num_blocks]
        if len(dilation_schedule) != num_blocks:
            raise ValueError("dilation_schedule length must match num_blocks")
        self.blocks = nn.ModuleList()
        for idx, dilation in enumerate(dilation_schedule):
            self.blocks.append(
                DeepHybridShortcutBlock(
                    width=width,
                    mlp_ratio=mlp_ratio,
                    dilation=int(dilation),
                    use_attention=attention_every > 0 and ((idx + 1) % attention_every == 0),
                    num_heads=num_heads,
                    adaln_zero=adaln_zero,
                )
            )
        self.final_ln = nn.LayerNorm(width, eps=1e-6)
        self.out_proj = nn.Linear(width, hidden_size)
        self.gamma_out = nn.Parameter(torch.tensor(float(gamma_out_init)))
        self.mag_head = MagnitudeHead(
            width,
            mag_abs_center=mag_abs_center,
            mag_abs_scale=mag_abs_scale,
        )
        zero_init(self.out_proj)

    def forward(
        self,
        u_source,
        source_layer,
        target_layer,
        timestep_embed,
        m_source=None,
        use_timestep_embed=True,
        key_padding_mask=None,
    ):
        batch_size = u_source.shape[0]
        device = u_source.device
        if not torch.is_tensor(source_layer):
            source_layer = torch.full((batch_size,), int(source_layer), device=device, dtype=torch.long)
        if not torch.is_tensor(target_layer):
            target_layer = torch.full((batch_size,), int(target_layer), device=device, dtype=torch.long)
        source_layer = source_layer.to(device=device, dtype=torch.long).clamp(0, self.depth)
        target_layer = target_layer.to(device=device, dtype=torch.long).clamp(0, self.depth)
        delta = (target_layer - source_layer).clamp(1, self.depth)

        if use_timestep_embed:
            cond = self.cond_t_proj(timestep_embed.float())
        else:
            cond = torch.zeros(batch_size, self.cond_t_proj.out_features, device=device, dtype=timestep_embed.dtype)
        cond = (
            cond
            + self.cond_layer_embed(source_layer)
            + self.cond_layer_embed(target_layer)
            + self.cond_delta_embed(delta - 1)
        )
        cond = self.cond_out(F.gelu(cond))

        h = self.in_proj(u_source.float())
        if key_padding_mask is not None:
            h = h.masked_fill(key_padding_mask[:, :, None].to(device=h.device), 0.0)
        for block in self.blocks:
            h = block(h, cond, key_padding_mask=key_padding_mask)
        h = self.final_ln(h)
        delta_y = self.out_proj(h)
        y = u_source.float() + self.gamma_out * delta_y if self.residual_output else delta_y
        if m_source is None:
            return y
        return y, self.mag_head(h, m_source, cond, key_padding_mask=key_padding_mask)


def predictor_config_from_name(name, hidden_size):
    normalized = str(name).lower().replace("-", "_")
    variants = {
        # Default for MDM-B: roughly 10-12% of an 8-layer 512-wide MDM backbone.
        "hybrid_mdm_10": {
            "width": max(96, int(round(hidden_size * 0.25))),
            "num_blocks": 8,
            "mlp_ratio": 1.5,
            "dilation_schedule": (1, 2, 4, 1, 2, 4, 1, 2),
            "num_heads": 4,
            "attention_every": 4,
            "cond_dim": max(64, int(round(hidden_size * 0.125))),
        },
        "hybrid_mdm_8": {
            "width": max(96, int(round(hidden_size * 0.21875))),
            "num_blocks": 6,
            "mlp_ratio": 1.5,
            "dilation_schedule": (1, 2, 4, 1, 2, 4),
            "num_heads": 4,
            "attention_every": 3,
            "cond_dim": max(64, int(round(hidden_size * 0.125))),
        },
        "hybrid_deep_10": {
            "width": max(96, int(round(hidden_size * 0.25))),
            "num_blocks": 8,
            "mlp_ratio": 1.5,
            "dilation_schedule": (1, 2, 4, 1, 2, 4, 1, 2),
            "num_heads": 4,
            "attention_every": 4,
            "cond_dim": max(64, int(round(hidden_size * 0.125))),
        },
        "hybrid_deep_12pct": {
            "width": max(112, int(round(hidden_size * 0.28125))),
            "num_blocks": 8,
            "mlp_ratio": 1.5,
            "dilation_schedule": (1, 2, 4, 1, 2, 4, 1, 2),
            "num_heads": 4,
            "attention_every": 4,
            "cond_dim": max(64, int(round(hidden_size * 0.125))),
        },
    }
    if normalized in {"hybrid", "default", "hybrid_deep"}:
        normalized = "hybrid_mdm_10"
    if normalized not in variants:
        raise ValueError(f"Unknown MDM depth shortcut predictor: {name!r}")
    return dict(variants[normalized])


def sample_distinct_pairs(
    num_hidden,
    max_gap,
    loc,
    sigma,
    num_pairs,
    pair_mode="trunc_normal",
    center_sigma=0.0,
):
    depth = int(num_hidden) - 1
    max_gap = max(1, min(int(max_gap), depth))
    candidates = []
    weights = []
    for a in range(depth):
        for b in range(a + 1, min(depth, a + max_gap) + 1):
            gap = b - a
            candidates.append((a, b))
            if pair_mode in {"uniform", "random"}:
                weight = 1.0
            else:
                weight = math.exp(-0.5 * ((gap - float(loc)) / max(float(sigma), 1e-6)) ** 2)
                if pair_mode in {"trunc_normal_centered", "trunc_normal_centered_to_uniform"} and center_sigma > 0:
                    midpoint = 0.5 * (a + b)
                    center = 0.5 * depth
                    weight *= math.exp(-0.5 * ((midpoint - center) / max(float(center_sigma), 1e-6)) ** 2)
            weights.append(weight)
    if not candidates:
        return [(0, min(1, depth))]
    chosen = random.choices(candidates, weights=weights, k=max(int(num_pairs), 1))
    deduped = []
    for pair in chosen:
        if pair not in deduped:
            deduped.append(pair)
    while len(deduped) < num_pairs:
        deduped.append(random.choice(candidates))
    return deduped[:num_pairs]


def sample_triplet(depth):
    depth = int(depth)
    for _ in range(32):
        a = random.randint(0, max(depth - 2, 0))
        b = random.randint(a + 1, max(a + 1, depth - 1))
        c = random.randint(b + 1, depth)
        if a < b < c:
            return a, b, c
    return 0, max(1, depth // 2), depth


def private_activation_loss(
    activations,
    max_pairs=0,
    use_residual=True,
    cosine_mode="bnd",
    pair_mode="first",
    token_mask=None,
    eps=1e-8,
):
    activations = activations.float()
    if use_residual:
        normed = l2_normalize_tokens(activations, eps=eps)
        common = normed.mean(dim=0)
        private = normed - common.detach().unsqueeze(0)
    else:
        common = activations.mean(dim=0)
        private = activations
    if token_mask is not None:
        private = private * token_mask.to(device=private.device, dtype=private.dtype)[None, :, :, None]

    num_layers = private.shape[0]
    pair_ids = [(a, b) for a in range(num_layers) for b in range(a + 1, num_layers)]
    if pair_mode == "random" and max_pairs > 0:
        random.shuffle(pair_ids)
    if max_pairs > 0:
        pair_ids = pair_ids[:max_pairs]
    if not pair_ids:
        zero = activations.new_tensor(0.0)
        return zero, zero, zero, zero

    pair_cosines = []
    pair_losses = []
    for a, b in pair_ids:
        if cosine_mode == "token":
            pa = F.normalize(private[a], dim=-1, eps=eps)
            pb = F.normalize(private[b], dim=-1, eps=eps)
            cos = (pa * pb).sum(dim=-1).mean()
        elif cosine_mode == "nd":
            pa = F.normalize(private[a].flatten(1), dim=-1, eps=eps)
            pb = F.normalize(private[b].flatten(1), dim=-1, eps=eps)
            cos = (pa * pb).sum(dim=-1).mean()
        else:
            pa = F.normalize(private[a].reshape(1, -1), dim=-1, eps=eps)
            pb = F.normalize(private[b].reshape(1, -1), dim=-1, eps=eps)
            cos = (pa * pb).sum()
        pair_cosines.append(cos)
        pair_losses.append(cos.square())

    loss = torch.stack(pair_losses).mean()
    pairwise_cosine = torch.stack(pair_cosines).mean()
    common_norm = torch.linalg.norm(common.reshape(common.shape[0], -1), dim=-1).mean()
    private_avg_norm = torch.linalg.norm(private.reshape(private.shape[0], private.shape[1], -1), dim=-1).mean()
    return loss, common_norm, private_avg_norm, pairwise_cosine
