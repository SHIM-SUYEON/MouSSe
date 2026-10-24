"""Neural vocoder loading and mel-to-waveform synthesis.

Waveform metrics use the external BigVSAN model from torch hub.
FT_VOCODER_PATH selects an optional fine-tuned checkpoint.
"""

import os

import torch
import torchaudio

from mousse.configs import config

# bigvsan output is resampled from 24 kHz to the configured 16 kHz rate.
VOCODER_SR = 24000

# Optional fine-tuned vocoder checkpoint path.
FT_VOCODER_PATH = None

_VOCODER_CACHE = None


def do_vocoder(spec, vocoder, device):
    with torch.no_grad():
        wav = vocoder.generate(spec.T.to(torch.float32).to(device)).squeeze(0)
    if VOCODER_SR != config.audio.sample_rate:
        wav = torchaudio.functional.resample(wav.unsqueeze(0), VOCODER_SR, config.audio.sample_rate).squeeze(0)
    return wav


def get_vocoder(device):
    global _VOCODER_CACHE
    if _VOCODER_CACHE is not None:
        return _VOCODER_CACHE.to(device)
    cached_hub = os.path.expanduser("~/.cache/torch/hub/ex3ndr_supervoice-vocoder_master")
    print(f"Cached hub: {cached_hub}")
    try:
        src = cached_hub if os.path.exists(cached_hub) else "ex3ndr/supervoice-vocoder"
        load_kw = dict(source="local") if os.path.exists(cached_hub) else {}
        vocoder = torch.hub.load(src, "bigvsan", **load_kw)
        if FT_VOCODER_PATH is not None and FT_VOCODER_PATH.exists():
            ft_ckpt = torch.load(str(FT_VOCODER_PATH), map_location="cpu")
            vocoder.load_state_dict(ft_ckpt["vocoder"])
            print("[VOCODER] Fine-tuned vocoder loaded.")
        else:
            print("[VOCODER] Using base vocoder weights.")
        vocoder = vocoder.eval()
        _VOCODER_CACHE = vocoder
        return vocoder.to(device)
    except Exception as e:
        print(f"[VOCODER] Unavailable: {e}")
        return None
