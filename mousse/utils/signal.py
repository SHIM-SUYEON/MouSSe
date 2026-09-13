"""Waveform signal utilities shared by training and inference."""

import torch

def rms_normalize(wav: torch.Tensor, target_rms: float = 0.1) -> torch.Tensor:
    rms = wav.pow(2).mean().sqrt()
    if rms < 1e-8:
        return wav
    return wav * (target_rms / rms)
