"""Runtime monkey-patches and conditioning helpers shared by training and inference.

The backbone (`AudioFlow` / `UncertaintyHead` / `LearnedSinusoidalPosEmb`) is
patched at run time with numerically-safe forward implementations that also
support JVP (forward-mode autodiff) used by the uncertainty head. These patches
are identical for training and inference; the training-only branches
(`self.training`) are simply dormant at inference time (`model.eval()`).

`STNOProjection` and `make_slot_id` build the optional STNO conditioning signal.
"""

import math

import torch
import torch.nn.functional as F
import torch.autograd.forward_ad as fwad
from einops import rearrange, reduce
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from mousse.configs import config


# -----------------------------------------------------------------
# STNO conditioning projection (Stage 2)
# -----------------------------------------------------------------
class STNOProjection(torch.nn.Module):
    """Maps oracle STNO features (p_T, p_N, p_O, slot_id) into the backbone's
    hidden space via the existing 'condition' pathway in generic_safe_forward.
    Zero-init on the output layer so conditioning starts as identity."""
    def __init__(self, n_dim: int):
        super().__init__()
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(4, n_dim // 4),
            torch.nn.SiLU(),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(n_dim // 4, n_dim),
        )
        torch.nn.init.zeros_(self.proj[3].weight)
        torch.nn.init.zeros_(self.proj[3].bias)

    def forward(self, p_T: torch.Tensor, p_N: torch.Tensor,
                p_O: torch.Tensor, slot_id: torch.Tensor) -> torch.Tensor:
        feat = torch.stack([p_T, p_N, p_O, slot_id], dim=-1)  # [B, T, 4]
        return self.proj(feat)                                  # [B, T, n_dim]


def make_slot_id(B: int, n_src: int, T_single: int, device) -> torch.Tensor:
    slot_id = torch.zeros(B, n_src * T_single, device=device)
    for s in range(n_src):
        slot_id[:, s * T_single:(s + 1) * T_single] = s / max(n_src - 1, 1)
    return slot_id


# -----------------------------------------------------------------
# Forward-pass monkey-patches
# -----------------------------------------------------------------
def safe_sinu_forward(self, x):
    x = x.reshape(-1)
    x = rearrange(x, "b -> b 1")
    freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * math.pi
    return torch.cat((freqs.sin(), freqs.cos()), dim=-1)


def generic_safe_forward(self, *, audio, noise, condition=None, times, mask=None, target=None, mask_loss=False):
    _, _tangent = fwad.unpack_dual(noise)
    is_jvp = (audio.ndim > 3) or (_tangent is not None)

    if is_jvp:
        backend_ctx = sdpa_kernel([SDPBackend.MATH])
    elif self.training:
        backend_ctx = sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION])
    else:
        backend_ctx = sdpa_kernel([SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH])

    with backend_ctx:
        combined = torch.cat([audio, noise], dim=-1)
        if hasattr(self, "transformer_input"):
            output = self.transformer_input(combined)
        else:
            output = self.input_layer(combined)
        if condition is not None:
            output = output + condition

        orig_shape = list(output.shape)
        seq_len  = orig_shape[-2]
        feat_dim = orig_shape[-1]
        leading  = orig_shape[:-2]

        raw_sinu   = self.sinu_pos_emb(times)
        sinu_times = raw_sinu.view(leading + [1, feat_dim])

        output      = torch.cat([output, sinu_times], dim=-2)
        output_flat = output.view(-1, seq_len + 1, feat_dim)
        output_flat = self.conv_embed(output_flat) + output_flat

        if self.training and not is_jvp:
            output_flat = grad_checkpoint(self.transformer, output_flat, use_reentrant=False)
        else:
            output_flat = self.transformer(output_flat)

        output_flat = self.prediction(output_flat)
        output_flat = output_flat[..., :-1, :]
        output = output_flat.view(leading + [seq_len, config.audio.n_mels])

        if target is not None:
            loss = F.mse_loss(output, target, reduction="none")
            loss = reduce(loss, "b n d -> b n", "mean")
            if mask_loss:
                loss = loss.masked_fill(~mask, 0.)
                n_masked = mask.sum(dim=-1).clamp(min=1)
                loss = loss.sum(dim=-1) / n_masked
            else:
                loss = loss.sum(dim=-1) / target.shape[1]
            return output, loss.mean()
        return output


def apply_forward_patches():
    """Install the shared forward patches on the backbone module classes.

    Returns the patched `mousse.models.audio_flow` module so that callers can
    add further patches (training additionally sets
    `AudioFlow.sample_with_uncertainty`).
    """
    import mousse.models.audio_flow as sf_model
    sf_model.LearnedSinusoidalPosEmb.forward = safe_sinu_forward
    sf_model.AudioFlow.forward = generic_safe_forward
    sf_model.UncertaintyHead.forward = generic_safe_forward
    return sf_model
