"""
Evaluation for the MouSSe single-stage PIT+UQ model (training/train.py).

Checkpoint keys: "backbone" + "uq_head"  (no stno_proj, no oracle STNO).
Expected SI-SNRi: ~13 dB  (matches validation performance).

Inference modes:
  --stage 1
        Backbone-only with fixed CFG (alpha=1.5). No UQ head at inference.
        This matches what the training validation loop effectively runs.
  --stage 2 --ablation no_stno
        UQ head active, condition=None (no STNO). Uses original num/den formula
        → lambda≈0 always (broken), effectively same as backbone-only.
  --stage 2 --ablation no_stno --lambda-scale 5.0
        Adaptive UQ guidance (σ_cond × 5.0).  May give slight improvement.

Usage (run from the repository root):
    CKPT=/path/to/checkpoint/large-sep-0820.best.pt

    # Backbone-only (stage 1, fixed alpha=1.5) — expected ~13 dB SI-SNRi
    nohup python3 -u inference/eval.py \
            --stage 1 --gpu 2,3 \
            --ckpt "$CKPT" --cap 1300 \
            --out results/stage1_uq/backbone_only \
            > eval-backbone.log 2>&1 &

    # Adaptive UQ guidance (no STNO, σ_cond × lambda_scale)
    nohup python3 -u inference/eval.py \
            --stage 2 --ablation no_stno --lambda-scale 5.0 --gpu 2,3 \
            --ckpt "$CKPT" --cap 1300 \
            --out results/stage1_uq/adaptive_uq \
            > eval-adaptive.log 2>&1 &

    nohup python3 -u inference/eval.py \
                --stage 2 --ablation no_stno --lambda-scale 1.0 --gpu 0,1 \
                --ckpt "$CKPT" --cap 1300 \
                --out results/stage1_uq_lambda1/main_adaptive_1_0 \
                > eval-adaptive_1_0_lambda1.log 2>&1 &

    --save-wav    Save separated audio files under out_dir/wav/

Or use the driver script (inference/eval.sh) to reproduce the full paper suite.
"""

import os
import sys
import gc
import json
import math
import argparse
import random
import subprocess
import tempfile
import pickle
import shutil
import warnings
from pathlib import Path

# Make the top-level ``mousse`` package importable when running this script
# directly (repo root is the parent of this inference/ directory).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import torch.autograd.forward_ad as fwad
from einops import rearrange
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, Subset
from torchmetrics.audio import (
    ShortTimeObjectiveIntelligibility,
    PerceptualEvaluationSpeechQuality,
    ScaleInvariantSignalNoiseRatio,
    ScaleInvariantSignalDistortionRatio,
    SignalDistortionRatio,
)
from tqdm import tqdm

from mousse.configs import config
from mousse.models import AudioFlow, UncertaintyHead

from mousse.utils.dataset import make_dataset
from mousse.models.patches import STNOProjection, make_slot_id, apply_forward_patches
from mousse.utils.vocoder import get_vocoder, do_vocoder
from mousse.utils.signal import rms_normalize
from mousse.utils.reporting import (
    LANG_CATS, GENDER_CATS, OVERLAP_CATS,
    _make_acc, _acc_add, _merge_acc, _avg,
    _load_meta, _categorize, _finalize_results,
)

random.seed(42)

DEFAULT_CKPT = Path(
    "/shared/NAS_SSD/syshim/hi-d/bluejays/speechflowuq/supervoice_separate/racoon-aug/single-stage/0826/large-sep-0820.best.pt"
)

DEFAULT_STAGE2_CKPT = Path(
    "/shared/NAS_SSD/syshim/hi-d/bluejays/speechflowuq/supervoice_separate/racoon-aug/stage2-trainable-0820-600-0821/large-uq-0820.best.pt"
)

N_SRC_LIST = [2, 3, 4]
TEST_CAP = 1300
EVAL_STEPS = 16
MAX_FRAMES_PER_SPK = 1500 # 1200
PESQ_SR = 16000

# Long sequences are truncated at MAX_FRAMES_PER_SPK; no duration-based skip.
MAX_MIX_DURATION = float("inf")

# Threshold for detecting active speech frames in the target mel spectrogram.
# Consistent with compute_stno_condition's activity detection.
ACTIVE_THRESH = 1e-4

DEFAULT_OUT_DIR = Path(
    "/shared/NAS_SSD/syshim/hi-d/bluejays/speechflowuq/supervoice_separate/racoon-aug/results/"
    "stage1-uq/single-stage-0826/"
)

os.makedirs(DEFAULT_OUT_DIR, exist_ok=True)


# ─── Category accumulator helpers ─────────────────────────────────────────────







# ─── Model / inference helpers (unchanged) ───────────────────────────────────




def fast_euler_sample(predictor, uq_head, audio, steps=16, alpha=1.5,
                      lambda_max=2.0, use_uq_guidance=True, condition=None,
                      lambda_fixed=None, lambda_scale=None):
    """ODE Euler sampler.
    lambda_fixed: if set (float), overrides both use_uq_guidance and alpha.
                  lambda_fixed=0.0 → conditioned only (no CFG, skips uncond pass).
                  lambda_fixed>0.0 → fixed CFG scale.
    lambda_scale: if set (float), enables the diet/UQ-only formula:
                  lambda* = clamp(sigma_cond.mean * lambda_scale, max=lambda_max)
                  This is the formula used by 0828-best-diet.py (no sigma_uncond, no STNO).
    """
    device = audio.device
    batch_size = audio.shape[0]
    z = torch.randn_like(audio)
    dt = 1.0 / steps
    audio_empty = torch.zeros_like(audio)
    sigma_cond_sum = None
    sigma_uncond_sum = None
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for i in range(steps):
                t_batch = torch.full((batch_size,), i * dt, device=device, dtype=audio.dtype)
                v_cond = predictor(audio=audio, noise=z, times=t_batch, condition=condition)

                if lambda_fixed is not None:
                    # Fixed-lambda sweep: UQ head skipped entirely
                    if lambda_fixed == 0.0:
                        v_guided = v_cond  # no CFG, skip uncond pass
                    else:
                        v_uncond = predictor(audio=audio_empty, noise=z, times=t_batch)
                        v_guided = v_cond + (v_cond - v_uncond) * lambda_fixed
                elif use_uq_guidance and lambda_scale is not None:
                    # Diet/UQ-only formula: lambda* = clamp(sigma_cond.mean * lambda_scale, max)
                    v_uncond = predictor(audio=audio_empty, noise=z, times=t_batch)
                    log_s_cond   = uq_head(audio=audio,       noise=z, times=t_batch)
                    log_s_uncond = uq_head(audio=audio_empty, noise=z, times=t_batch)
                    sigma_cond   = torch.exp(log_s_cond)
                    sigma_uncond = torch.exp(log_s_uncond)
                    sc = sigma_cond.float()
                    su = sigma_uncond.float()
                    sigma_cond_sum   = sc if sigma_cond_sum   is None else sigma_cond_sum   + sc
                    sigma_uncond_sum = su if sigma_uncond_sum is None else sigma_uncond_sum + su
                    lambda_star = torch.clamp(
                        sigma_cond.mean(dim=-1, keepdim=True) * lambda_scale,
                        max=lambda_max,
                    )
                    if i == 0: print(f"[UQ-diet] lambda_star step0  mean={lambda_star.mean():.3f}  max={lambda_star.max():.3f}  frac_clamped={(lambda_star >= lambda_max).float().mean():.3f}", flush=True)
                    v_guided = v_cond + (v_cond - v_uncond) * lambda_star
                elif use_uq_guidance:
                    # Original formula: lambda* = clamp(num/den, 0, lambda_max)
                    v_uncond = predictor(audio=audio_empty, noise=z, times=t_batch)
                    log_s_uncond = uq_head(audio=audio, noise=z, times=t_batch)
                    log_s_cond = uq_head(audio=audio, noise=z, times=t_batch, condition=condition)
                    sigma_uncond = torch.exp(log_s_uncond)
                    sigma_cond = torch.exp(log_s_cond)
                    sc = sigma_cond.float()
                    sigma_cond_sum = sc if sigma_cond_sum is None else sigma_cond_sum + sc
                    num = torch.sum(sigma_uncond * sigma_cond - sigma_cond ** 2, dim=-1, keepdim=True)
                    den = torch.sum((sigma_cond - sigma_uncond) ** 2, dim=-1, keepdim=True).clamp(min=1e-8)
                    lambda_star = torch.clamp(num / den, min=0.0, max=lambda_max)
                    if i == 0: print(f"[UQ] lambda_star step0  mean={lambda_star.mean():.3f}  max={lambda_star.max():.3f}  frac_clamped={( lambda_star >= lambda_max).float().mean():.3f}", flush=True)
                    v_guided = v_cond + (v_cond - v_uncond) * lambda_star
                else:
                    v_uncond = predictor(audio=audio_empty, noise=z, times=t_batch)
                    v_guided = v_cond + (v_cond - v_uncond) * alpha
                z = z + v_guided.float() * dt
    sigma_cond_mean   = sigma_cond_sum   / steps if sigma_cond_sum   is not None else None
    sigma_uncond_mean = sigma_uncond_sum / steps if sigma_uncond_sum is not None else None
    return z, sigma_cond_mean, sigma_uncond_mean


_VOCODER_CACHE = None




def _snr_val(preds: torch.Tensor, target: torch.Tensor) -> float:
    signal = float(target.pow(2).mean())
    noise  = float((preds - target).pow(2).mean())
    if noise < 1e-8:
        return 40.0
    return 10.0 * math.log10(max(signal, 1e-8) / noise)


def compute_stno_condition(target_flat, n_src, T_per_spk, stno_proj, device):
    d_st_total = (target_flat.abs().sum(dim=-1) > 1e-4).float()
    d_st = d_st_total.reshape(1, n_src, T_per_spk)
    p_S = torch.prod(1.0 - d_st, dim=1, keepdim=True)
    p_T_list, p_N_list, p_O_list = [], [], []
    for s_k in range(n_src):
        d_sk = d_st[:, s_k:s_k+1, :]
        mo = torch.ones_like(d_st); mo[:, s_k, :] = 0.0
        p_oth = torch.prod(1.0 - (d_st * mo), dim=1, keepdim=True)
        p_T_list.append(d_sk * p_oth)
        p_O_list.append(d_sk * (1.0 - p_oth))
        p_N_list.append((1.0 - p_S) - d_sk * p_oth - d_sk * (1.0 - p_oth))
    p_T = torch.cat(p_T_list, dim=1).reshape(1, n_src * T_per_spk)
    p_N = torch.cat(p_N_list, dim=1).reshape(1, n_src * T_per_spk)
    p_O = torch.cat(p_O_list, dim=1).reshape(1, n_src * T_per_spk)
    slot_id = make_slot_id(1, n_src, T_per_spk, device)
    return stno_proj(p_T, p_N, p_O, slot_id)


# ─── Print helpers ────────────────────────────────────────────────────────────





# ─── Core evaluation loop ─────────────────────────────────────────────────────
def _eval_loop(backbone, uq_head, stno_proj, test_loaders, device,
               stage=1, out_dir=None, save_wav=False, ablation='full', eval_steps=None,
               lambda_fixed=None, lambda_scale=None):
    """Run the evaluation loop and return raw accumulators (no printing of final tables)."""
    # ── Ablation mode banner ──────────────────────────────────────────────────
    _ABLATION_VALID = ("full", "no_uq", "no_stno", "no_entropy", "no_stno_no_uq")
    if ablation not in _ABLATION_VALID:
        raise ValueError(f"Unknown ablation mode '{ablation}'. Choose from {_ABLATION_VALID}")
    use_uq_guidance = (stage == 2) and (ablation not in ("no_uq", "no_stno_no_uq"))
    use_stno        = (stage == 2) and (ablation not in ("no_stno", "no_stno_no_uq"))
    print(f"\n{'='*70}")
    print(f"  ABLATION MODE : {ablation.upper()}")
    if lambda_fixed is not None:
        print(f"  UQ guidance   : NO  — fixed λ={lambda_fixed:.2f} (lambda sweep override)")
    elif use_uq_guidance and lambda_scale is not None:
        print(f"  UQ guidance   : YES — λ*=σ_cond×{lambda_scale} (diet/UQ-only formula, max={2.0})")
    else:
        print(f"  UQ guidance   : {'YES — λ* adaptive (num/den)' if use_uq_guidance else 'NO  — fixed α=1.5'}")
    print(f"  STNO cond.    : {'YES — oracle' if use_stno else 'NO  — condition=None'}")
    print(f"  Entropy loss  : {'separate no-entropy checkpoint' if ablation == 'no_entropy' else 'standard training'}")
    print(f"{'='*70}\n")

    backbone.eval()
    uq_head.eval()
    if stno_proj is not None:
        stno_proj.eval()

    sr = config.audio.sample_rate
    stoi_m  = ShortTimeObjectiveIntelligibility(fs=sr, extended=False).to(device)
    estoi_m = ShortTimeObjectiveIntelligibility(fs=sr, extended=True).to(device)
    pesq_m  = PerceptualEvaluationSpeechQuality(fs=PESQ_SR, mode="wb").to(device)
    sisnr_m = ScaleInvariantSignalNoiseRatio().to(device)
    sisdr_m = ScaleInvariantSignalDistortionRatio().to(device)
    sdr_m   = SignalDistortionRatio(filter_length=1, zero_mean=False).to(device)

    vocoder = get_vocoder(device)
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)

    # Accumulators
    nsrc_acc = {n: _make_acc() for n in N_SRC_LIST}
    cat_acc = {
        n: {
            "lang": {c: _make_acc() for c in LANG_CATS},
            "gender": {c: _make_acc() for c in GENDER_CATS},
            "overlap": {c: _make_acc() for c in OVERLAP_CATS},
        }
        for n in N_SRC_LIST
    }
    per_sample_records = []  # one dict per mixture

    for n_src, loader in test_loaders.items():
        print(f"\n[EVAL] n_src={n_src}, {len(loader.dataset)} samples  (MAX_MIX_DURATION={MAX_MIX_DURATION}s)")
        pesq_skip_cnt = 0
        dur_skip_cnt  = 0

        wav_dir = None
        if out_dir is not None and save_wav:
            wav_dir = Path(out_dir) / "wav" / f"{n_src}spk"
            wav_dir.mkdir(parents=True, exist_ok=True)

        for item_idx, batch in enumerate(tqdm(loader, desc=f"n_src={n_src}")):
            cond_spec, flow_target, mix_paths, src_paths = batch

            single_cond = cond_spec[0:1].to(device)
            single_target = flow_target[0:1].to(device)
            item_src_paths = src_paths[0] if src_paths else []

            T_per_spk = single_cond.shape[1] // n_src
            if T_per_spk > MAX_FRAMES_PER_SPK:
                T_per_spk = MAX_FRAMES_PER_SPK
                seq_len = T_per_spk * n_src
                single_cond = single_cond[:, :seq_len, :]
                single_target = single_target[:, :seq_len, :]

            # Build conditioning
            norm_cond = (single_cond - config.audio.norm_mean) / config.audio.norm_std
            stno_cond = None
            if use_stno and stno_proj is not None:
                with torch.no_grad():
                    stno_cond = compute_stno_condition(
                        single_target, n_src, T_per_spk, stno_proj, device
                    )

            mix_input = norm_cond.to(dtype=torch.bfloat16)
            sep_spec, sigma_cond_mean, sigma_uncond_mean = fast_euler_sample(
                backbone, uq_head, mix_input,
                steps=(eval_steps or EVAL_STEPS), alpha=1.5, lambda_max=2.0,
                use_uq_guidance=use_uq_guidance,
                condition=stno_cond.to(dtype=torch.bfloat16) if stno_cond is not None else None,
                lambda_fixed=lambda_fixed,
                lambda_scale=lambda_scale,
            )

            sep_flat = sep_spec.detach()[0].to(torch.float32)
            tgt_flat = single_target[0].to(torch.float32)
            cond_flat = single_cond[0].to(torch.float32)

            denorm_sep = sep_flat * config.audio.norm_std + config.audio.norm_mean
            denorm_sep = torch.nan_to_num(denorm_sep, nan=0.0, posinf=1.0, neginf=-1.0)
            denorm_tgt = tgt_flat

            pred_slots = [denorm_sep[s*T_per_spk:(s+1)*T_per_spk, :].flatten() for s in range(n_src)]
            tgt_slots = [denorm_tgt[s*T_per_spk:(s+1)*T_per_spk, :].flatten() for s in range(n_src)]
            mix_slot = cond_flat[0:T_per_spk, :].flatten()

            # PIT assignment
            cost_mat = np.zeros((n_src, n_src))
            for pi in range(n_src):
                for ti in range(n_src):
                    cost_mat[pi, ti] = -sisnr_m(pred_slots[pi], tgt_slots[ti]).item()
            pred_ind, tgt_ind = linear_sum_assignment(cost_mat)

            # Load metadata for this sample
            meta = _load_meta(mix_paths[0])
            if meta is not None and meta["total_length_sec"] > MAX_MIX_DURATION:
                dur_skip_cnt += 1
                continue
            lang, gender, overlap = _categorize(meta)
            stem = Path(mix_paths[0]).stem

            # Load mix wav (needed for SI-SDRi baseline and optional save)
            sample_dir = wav_dir / stem if wav_dir is not None else None
            wav_mix = None
            try:
                mix_wav_raw, mix_sr_raw = torchaudio.load(mix_paths[0])
                if mix_sr_raw != sr:
                    mix_wav_raw = torchaudio.functional.resample(mix_wav_raw, mix_sr_raw, sr)
                wav_mix = mix_wav_raw.mean(0).to(device)
                if sample_dir is not None:
                    sample_dir.mkdir(parents=True, exist_ok=True)
                    torchaudio.save(str(sample_dir / "mix.wav"), mix_wav_raw.mean(0, keepdim=True).cpu(), sr)
            except Exception as e:
                print(f"  [WAV] mix load error: {e}")

            # Increment sample counters
            nsrc_acc[n_src]["n_samples"] += 1
            if lang and lang in LANG_CATS:     cat_acc[n_src]["lang"][lang]["n_samples"]       += 1
            if gender and gender in GENDER_CATS: cat_acc[n_src]["gender"][gender]["n_samples"] += 1
            if overlap and overlap in OVERLAP_CATS: cat_acc[n_src]["overlap"][overlap]["n_samples"] += 1

            # Per-sample record for JSON
            record = {
                "stem": stem, "n_src": n_src,
                "lang": lang, "gender": gender,
                "overlap": f"{overlap}%" if overlap else None,
            }

            # Accumulate per speaker slot
            for pi, ti in zip(pred_ind, tgt_ind):
                p_vec   = pred_slots[pi]
                t_vec   = tgt_slots[ti]
                spk_key = f"s{ti + 1}"
                uq_val = None
                uq_active_val = None
                if sigma_cond_mean is not None:
                    slot_sigma = sigma_cond_mean[0, pi * T_per_spk:(pi + 1) * T_per_spk, :]
                    # --- All-frames UQ (existing) ---
                    uq_val = round(float(slot_sigma.mean().item()), 6)
                    _acc_add(nsrc_acc[n_src], uq=uq_val)
                    if lang    and lang    in LANG_CATS:    _acc_add(cat_acc[n_src]["lang"][lang],       uq=uq_val)
                    if gender  and gender  in GENDER_CATS:  _acc_add(cat_acc[n_src]["gender"][gender],   uq=uq_val)
                    if overlap and overlap in OVERLAP_CATS: _acc_add(cat_acc[n_src]["overlap"][overlap], uq=uq_val)
                    # --- Active-frames-only UQ ---
                    # Active mask: frames where target mel energy exceeds ACTIVE_THRESH
                    # (same criterion as compute_stno_condition's activity detection)
                    tgt_mel = tgt_slots[ti].view(T_per_spk, -1)       # [T, n_mels]
                    active_mask = tgt_mel.abs().sum(dim=-1) > ACTIVE_THRESH  # [T]
                    n_active = active_mask.sum().item()
                    if n_active > 0:
                        uq_active_val = round(float(slot_sigma[active_mask].mean().item()), 6)
                        _acc_add(nsrc_acc[n_src], uq_active=uq_active_val)
                        if lang    and lang    in LANG_CATS:    _acc_add(cat_acc[n_src]["lang"][lang],       uq_active=uq_active_val)
                        if gender  and gender  in GENDER_CATS:  _acc_add(cat_acc[n_src]["gender"][gender],   uq_active=uq_active_val)
                        if overlap and overlap in OVERLAP_CATS: _acc_add(cat_acc[n_src]["overlap"][overlap], uq_active=uq_active_val)
                    # --- STO region sigma (Silence / Target-only / Overlap) ---
                    other_mels = [tgt_slots[oi].view(T_per_spk, -1) for oi in range(n_src) if oi != ti]
                    if other_mels:
                        others_active = torch.stack(
                            [m.abs().sum(dim=-1) > ACTIVE_THRESH for m in other_mels]
                        ).any(dim=0)
                    else:
                        others_active = torch.zeros(T_per_spk, dtype=torch.bool, device=slot_sigma.device)
                    silence_mask     = ~active_mask
                    target_only_mask = active_mask & ~others_active
                    overlap_mask     = active_mask &  others_active
                    slot_su = (sigma_uncond_mean[0, pi*T_per_spk:(pi+1)*T_per_spk, :]
                               if sigma_uncond_mean is not None else None)
                    if silence_mask.any():
                        sc = float(slot_sigma[silence_mask].mean().item())
                        _acc_add(nsrc_acc[n_src], uq_silence=sc)
                        if slot_su is not None:
                            _acc_add(nsrc_acc[n_src], uq_uncond_silence=float(slot_su[silence_mask].mean().item()))
                    if target_only_mask.any():
                        sc = float(slot_sigma[target_only_mask].mean().item())
                        _acc_add(nsrc_acc[n_src], uq_target=sc)
                        if slot_su is not None:
                            _acc_add(nsrc_acc[n_src], uq_uncond_target=float(slot_su[target_only_mask].mean().item()))
                    if overlap_mask.any():
                        sc = float(slot_sigma[overlap_mask].mean().item())
                        _acc_add(nsrc_acc[n_src], uq_overlap=sc)
                        if slot_su is not None:
                            _acc_add(nsrc_acc[n_src], uq_uncond_overlap=float(slot_su[overlap_mask].mean().item()))
                spk_rec = {"si_snr_mix": None, "si_snri": None, "si_snr_out": None,
                           "sisdri_vo": None, "si_sdr_out_vo": None, "si_sdr_mix_vo": None,
                           "sdri_vo": None,
                           "sisdri_orig": None, "sdri_orig": None,
                           "estoi": None, "estoii": None,
                           "stoi": None, "stoii": None, "pesq": None, "dur": None,
                           "stoi_orig": None, "stoii_orig": None,
                           "estoi_orig": None, "estoii_orig": None, "pesq_orig": None,
                           "uq_sigma_mean": uq_val, "uq_sigma_active": uq_active_val}

                # Speech duration from JSON metadata (target speaker ti)
                if meta is not None and ti < len(meta["durations"]):
                    dur_val = meta["durations"][ti]
                    spk_rec["dur"] = round(dur_val, 4)
                    _acc_add(nsrc_acc[n_src], dur=dur_val)
                    if lang    and lang    in LANG_CATS:    _acc_add(cat_acc[n_src]["lang"][lang],       dur=dur_val)
                    if gender  and gender  in GENDER_CATS:  _acc_add(cat_acc[n_src]["gender"][gender],   dur=dur_val)
                    if overlap and overlap in OVERLAP_CATS: _acc_add(cat_acc[n_src]["overlap"][overlap], dur=dur_val)

                # SI-SNRi (mel-domain) + absolute SI-SNR_out and SI-SNR_mix
                try:
                    sisnr_sep     = float(-cost_mat[pi, ti])   # SI-SNR(output, target)
                    sisnr_mix_val = sisnr_m(mix_slot, t_vec).item()  # SI-SNR(mixture, target)
                    sisnri_val    = sisnr_sep - sisnr_mix_val
                    spk_rec["si_snr_mix"] = round(sisnr_mix_val, 4)
                    spk_rec["si_snri"]    = round(sisnri_val, 4)
                    spk_rec["si_snr_out"] = round(sisnr_sep, 4)
                    _acc_add(nsrc_acc[n_src], sisnri=sisnri_val,
                             si_snr_out=sisnr_sep, si_snr_mix=sisnr_mix_val)
                    if lang and lang in LANG_CATS:
                        _acc_add(cat_acc[n_src]["lang"][lang],
                                 sisnri=sisnri_val, si_snr_out=sisnr_sep, si_snr_mix=sisnr_mix_val)
                    if gender and gender in GENDER_CATS:
                        _acc_add(cat_acc[n_src]["gender"][gender],
                                 sisnri=sisnri_val, si_snr_out=sisnr_sep, si_snr_mix=sisnr_mix_val)
                    if overlap and overlap in OVERLAP_CATS:
                        _acc_add(cat_acc[n_src]["overlap"][overlap],
                                 sisnri=sisnri_val, si_snr_out=sisnr_sep, si_snr_mix=sisnr_mix_val)
                except Exception as e:
                    print(f"  [SI-SNRi] error spk={pi}: {e}")

                # Vocoder → wav save + SI-SDRi + SDRi (waveform) + STOI + PESQ
                if vocoder is not None:
                    src_mel_frames = T_per_spk
                    if ti < len(item_src_paths):
                        try:
                            info = torchaudio.info(item_src_paths[ti])
                            actual_wav_frames = int(info.num_frames * sr / info.sample_rate)
                            src_mel_frames = min(int(actual_wav_frames / config.audio.hop_size), T_per_spk)
                        except Exception:
                            pass

                    spec_shape = (T_per_spk, config.audio.n_mels)
                    try:
                        p_spec = p_vec.view(spec_shape)[:src_mel_frames]
                        t_spec = t_vec.view(spec_shape)[:src_mel_frames]
                        with torch.no_grad():
                            wav_p = do_vocoder(p_spec, vocoder, device)
                            wav_t = do_vocoder(t_spec, vocoder, device)
                    except Exception as e:
                        print(f"  [Vocoder] error spk={pi}: {e}")
                        record[spk_key] = spk_rec
                        continue

                    # Load original source waveform for _orig metrics
                    wav_orig = None
                    if ti < len(item_src_paths):
                        try:
                            wav_orig_raw, orig_sr = torchaudio.load(item_src_paths[ti])
                            if orig_sr != sr:
                                wav_orig_raw = torchaudio.functional.resample(wav_orig_raw, orig_sr, sr)
                            wav_orig = wav_orig_raw.mean(0).to(device)
                        except Exception as e:
                            print(f"  [WAV] orig load error spk={ti}: {e}")

                    # Save separated wav
                    if sample_dir is not None:
                        try:
                            torchaudio.save(str(sample_dir / f"{spk_key}.wav"),
                                            wav_p.unsqueeze(0).cpu(), sr)
                        except Exception as e:
                            print(f"  [WAV] {spk_key} save error: {e}")

                    wav_p_n = rms_normalize(wav_p)
                    wav_t_n = rms_normalize(wav_t)
                    _MIN_RMS = 1e-2
                    has_speech = (
                        wav_p_n.pow(2).mean().sqrt().item() >= _MIN_RMS and
                        wav_t_n.pow(2).mean().sqrt().item() >= _MIN_RMS
                    )

                    # SI-SDRi, SI-SDR_out, SI-SDR_mix, SDRi vs vocoded target (waveform domain)
                    if has_speech and wav_mix is not None:
                        try:
                            tgt_len = wav_t_n.shape[-1]
                            wav_mix_trim = wav_mix[:tgt_len]
                            if wav_mix_trim.shape[-1] < tgt_len:
                                wav_mix_trim = F.pad(wav_mix_trim, (0, tgt_len - wav_mix_trim.shape[-1]))
                            wav_mix_n = rms_normalize(wav_mix_trim)
                            min_len = min(wav_p_n.shape[-1], tgt_len)

                            sisdr_sep = sisdr_m(wav_p_n[:min_len], wav_t_n[:min_len]).item()
                            sisdr_mix = sisdr_m(wav_mix_n[:min_len], wav_t_n[:min_len]).item()
                            sisdri_vo_val = sisdr_sep - sisdr_mix

                            # SI-SDRi (improvement)
                            if not (math.isnan(sisdri_vo_val) or math.isinf(sisdri_vo_val)):
                                spk_rec["sisdri_vo"] = round(sisdri_vo_val, 4)
                                _acc_add(nsrc_acc[n_src], sisdri_vo=sisdri_vo_val)
                                if lang    and lang    in LANG_CATS:    _acc_add(cat_acc[n_src]["lang"][lang],       sisdri_vo=sisdri_vo_val)
                                if gender  and gender  in GENDER_CATS:  _acc_add(cat_acc[n_src]["gender"][gender],   sisdri_vo=sisdri_vo_val)
                                if overlap and overlap in OVERLAP_CATS: _acc_add(cat_acc[n_src]["overlap"][overlap], sisdri_vo=sisdri_vo_val)

                            # SI-SDR_out absolute (baseline-free, for fair n_src comparison)
                            if not (math.isnan(sisdr_sep) or math.isinf(sisdr_sep)):
                                spk_rec["si_sdr_out_vo"] = round(sisdr_sep, 4)
                                _acc_add(nsrc_acc[n_src], si_sdr_out_vo=sisdr_sep)
                                if lang    and lang    in LANG_CATS:    _acc_add(cat_acc[n_src]["lang"][lang],       si_sdr_out_vo=sisdr_sep)
                                if gender  and gender  in GENDER_CATS:  _acc_add(cat_acc[n_src]["gender"][gender],   si_sdr_out_vo=sisdr_sep)
                                if overlap and overlap in OVERLAP_CATS: _acc_add(cat_acc[n_src]["overlap"][overlap], si_sdr_out_vo=sisdr_sep)

                            # SI-SDR_mix baseline (reveals per-n_src inflation)
                            if not (math.isnan(sisdr_mix) or math.isinf(sisdr_mix)):
                                spk_rec["si_sdr_mix_vo"] = round(sisdr_mix, 4)
                                _acc_add(nsrc_acc[n_src], si_sdr_mix_vo=sisdr_mix)
                                if lang    and lang    in LANG_CATS:    _acc_add(cat_acc[n_src]["lang"][lang],       si_sdr_mix_vo=sisdr_mix)
                                if gender  and gender  in GENDER_CATS:  _acc_add(cat_acc[n_src]["gender"][gender],   si_sdr_mix_vo=sisdr_mix)
                                if overlap and overlap in OVERLAP_CATS: _acc_add(cat_acc[n_src]["overlap"][overlap], si_sdr_mix_vo=sisdr_mix)

                            sdr_sep_vo = sdr_m(wav_p_n[:min_len], wav_t_n[:min_len]).item()
                            sdr_mix_vo = sdr_m(wav_mix_n[:min_len], wav_t_n[:min_len]).item()
                            sdri_vo_val = sdr_sep_vo - sdr_mix_vo
                            if not (math.isnan(sdri_vo_val) or math.isinf(sdri_vo_val)):
                                spk_rec["sdri_vo"] = round(sdri_vo_val, 4)
                                _acc_add(nsrc_acc[n_src], sdri_vo=sdri_vo_val)
                                if lang    and lang    in LANG_CATS:    _acc_add(cat_acc[n_src]["lang"][lang],       sdri_vo=sdri_vo_val)
                                if gender  and gender  in GENDER_CATS:  _acc_add(cat_acc[n_src]["gender"][gender],   sdri_vo=sdri_vo_val)
                                if overlap and overlap in OVERLAP_CATS: _acc_add(cat_acc[n_src]["overlap"][overlap], sdri_vo=sdri_vo_val)

                            snr_sep_vo = _snr_val(wav_p_n[:min_len], wav_t_n[:min_len])
                            snr_mix_vo = _snr_val(wav_mix_n[:min_len], wav_t_n[:min_len])
                            snri_vo_val = snr_sep_vo - snr_mix_vo
                            if not (math.isnan(snri_vo_val) or math.isinf(snri_vo_val)):
                                spk_rec["snri_vo"] = round(snri_vo_val, 4)
                                _acc_add(nsrc_acc[n_src], snri_vo=snri_vo_val)
                                if lang    and lang    in LANG_CATS:    _acc_add(cat_acc[n_src]["lang"][lang],       snri_vo=snri_vo_val)
                                if gender  and gender  in GENDER_CATS:  _acc_add(cat_acc[n_src]["gender"][gender],   snri_vo=snri_vo_val)
                                if overlap and overlap in OVERLAP_CATS: _acc_add(cat_acc[n_src]["overlap"][overlap], snri_vo=snri_vo_val)
                        except Exception as e:
                            print(f"  [SI-SDRi/SDRi vo] error spk={pi}: {e}")

                    # SI-SDRi and SDRi vs original target (waveform domain)
                    if has_speech and wav_mix is not None and wav_orig is not None:
                        try:
                            wav_orig_n = rms_normalize(wav_orig)
                            if wav_orig_n.pow(2).mean().sqrt().item() >= _MIN_RMS:
                                orig_len = wav_orig_n.shape[-1]
                                wav_mix_trim_o = wav_mix[:orig_len]
                                if wav_mix_trim_o.shape[-1] < orig_len:
                                    wav_mix_trim_o = F.pad(wav_mix_trim_o, (0, orig_len - wav_mix_trim_o.shape[-1]))
                                wav_mix_n_o = rms_normalize(wav_mix_trim_o)
                                min_len_o = min(wav_p_n.shape[-1], orig_len)

                                sisdr_sep_o = sisdr_m(wav_p_n[:min_len_o], wav_orig_n[:min_len_o]).item()
                                sisdr_mix_o = sisdr_m(wav_mix_n_o[:min_len_o], wav_orig_n[:min_len_o]).item()
                                sisdri_orig_val = sisdr_sep_o - sisdr_mix_o
                                if not (math.isnan(sisdri_orig_val) or math.isinf(sisdri_orig_val)):
                                    spk_rec["sisdri_orig"] = round(sisdri_orig_val, 4)
                                    _acc_add(nsrc_acc[n_src], sisdri_orig=sisdri_orig_val)
                                    if lang    and lang    in LANG_CATS:    _acc_add(cat_acc[n_src]["lang"][lang],       sisdri_orig=sisdri_orig_val)
                                    if gender  and gender  in GENDER_CATS:  _acc_add(cat_acc[n_src]["gender"][gender],   sisdri_orig=sisdri_orig_val)
                                    if overlap and overlap in OVERLAP_CATS: _acc_add(cat_acc[n_src]["overlap"][overlap], sisdri_orig=sisdri_orig_val)

                                sdr_sep_o = sdr_m(wav_p_n[:min_len_o], wav_orig_n[:min_len_o]).item()
                                sdr_mix_o = sdr_m(wav_mix_n_o[:min_len_o], wav_orig_n[:min_len_o]).item()
                                sdri_orig_val = sdr_sep_o - sdr_mix_o
                                if not (math.isnan(sdri_orig_val) or math.isinf(sdri_orig_val)):
                                    spk_rec["sdri_orig"] = round(sdri_orig_val, 4)
                                    _acc_add(nsrc_acc[n_src], sdri_orig=sdri_orig_val)
                                    if lang    and lang    in LANG_CATS:    _acc_add(cat_acc[n_src]["lang"][lang],       sdri_orig=sdri_orig_val)
                                    if gender  and gender  in GENDER_CATS:  _acc_add(cat_acc[n_src]["gender"][gender],   sdri_orig=sdri_orig_val)
                                    if overlap and overlap in OVERLAP_CATS: _acc_add(cat_acc[n_src]["overlap"][overlap], sdri_orig=sdri_orig_val)
                        except Exception as e:
                            print(f"  [SI-SDRi/SDRi orig] error spk={pi}: {e}")

                    if has_speech and wav_p_n.shape[-1] >= int(sr * 0.5):
                        try:
                            stoi_val = stoi_m(wav_p_n, wav_t_n).item()
                            spk_rec["stoi"] = round(stoi_val, 4)
                            _acc_add(nsrc_acc[n_src], stoi=stoi_val)
                            if lang    and lang    in LANG_CATS:    _acc_add(cat_acc[n_src]["lang"][lang],       stoi=stoi_val)
                            if gender  and gender  in GENDER_CATS:  _acc_add(cat_acc[n_src]["gender"][gender],   stoi=stoi_val)
                            if overlap and overlap in OVERLAP_CATS: _acc_add(cat_acc[n_src]["overlap"][overlap], stoi=stoi_val)
                            # ESTOi (Extended STOI, absolute)
                            estoi_val = estoi_m(wav_p_n, wav_t_n).item()
                            if not (math.isnan(estoi_val) or math.isinf(estoi_val)):
                                spk_rec["estoi"] = round(estoi_val, 4)
                                _acc_add(nsrc_acc[n_src], estoi=estoi_val)
                                if lang    and lang    in LANG_CATS:    _acc_add(cat_acc[n_src]["lang"][lang],       estoi=estoi_val)
                                if gender  and gender  in GENDER_CATS:  _acc_add(cat_acc[n_src]["gender"][gender],   estoi=estoi_val)
                                if overlap and overlap in OVERLAP_CATS: _acc_add(cat_acc[n_src]["overlap"][overlap], estoi=estoi_val)

                            # STOIi / ESTOii: improvement over mixture-signal baseline
                            # Trim mixture to match the vocoded target length for a fair comparison
                            if wav_mix is not None:
                                tgt_w_len = wav_t_n.shape[-1]
                                mix_for_stoi = wav_mix[:tgt_w_len]
                                if mix_for_stoi.shape[-1] < tgt_w_len:
                                    mix_for_stoi = F.pad(mix_for_stoi, (0, tgt_w_len - mix_for_stoi.shape[-1]))
                                mix_stoi_n = rms_normalize(mix_for_stoi)
                                if mix_stoi_n.shape[-1] >= int(sr * 0.5):
                                    stoi_mix_val = stoi_m(mix_stoi_n, wav_t_n).item()
                                    stoii_val = stoi_val - stoi_mix_val
                                    if not (math.isnan(stoii_val) or math.isinf(stoii_val)):
                                        spk_rec["stoii"] = round(stoii_val, 4)
                                        _acc_add(nsrc_acc[n_src], stoii=stoii_val)
                                        if lang    and lang    in LANG_CATS:    _acc_add(cat_acc[n_src]["lang"][lang],       stoii=stoii_val)
                                        if gender  and gender  in GENDER_CATS:  _acc_add(cat_acc[n_src]["gender"][gender],   stoii=stoii_val)
                                        if overlap and overlap in OVERLAP_CATS: _acc_add(cat_acc[n_src]["overlap"][overlap], stoii=stoii_val)

                                    estoi_mix_val = estoi_m(mix_stoi_n, wav_t_n).item()
                                    estoii_val = estoi_val - estoi_mix_val
                                    if not (math.isnan(estoii_val) or math.isinf(estoii_val)):
                                        spk_rec["estoii"] = round(estoii_val, 4)
                                        _acc_add(nsrc_acc[n_src], estoii=estoii_val)
                                        if lang    and lang    in LANG_CATS:    _acc_add(cat_acc[n_src]["lang"][lang],       estoii=estoii_val)
                                        if gender  and gender  in GENDER_CATS:  _acc_add(cat_acc[n_src]["gender"][gender],   estoii=estoii_val)
                                        if overlap and overlap in OVERLAP_CATS: _acc_add(cat_acc[n_src]["overlap"][overlap], estoii=estoii_val)
                        except Exception as e:
                            print(f"  [STOI/ESTOi] error spk={pi}: {e}")

                    if has_speech:
                        try:
                            p_16k = torchaudio.functional.resample(wav_p_n.unsqueeze(0), sr, PESQ_SR).squeeze(0).to(device)
                            t_16k = torchaudio.functional.resample(wav_t_n.unsqueeze(0), sr, PESQ_SR).squeeze(0).to(device)
                            p_16k = p_16k.clamp(-1.0, 1.0)
                            t_16k = t_16k.clamp(-1.0, 1.0)
                            if p_16k.shape[-1] >= int(PESQ_SR * 0.5) and t_16k.shape[-1] >= int(PESQ_SR * 0.5):
                                pesq_val = pesq_m(p_16k, t_16k).item()
                                spk_rec["pesq"] = round(pesq_val, 4)
                                _acc_add(nsrc_acc[n_src], pesq=pesq_val)
                                if lang    and lang    in LANG_CATS:    _acc_add(cat_acc[n_src]["lang"][lang],       pesq=pesq_val)
                                if gender  and gender  in GENDER_CATS:  _acc_add(cat_acc[n_src]["gender"][gender],   pesq=pesq_val)
                                if overlap and overlap in OVERLAP_CATS: _acc_add(cat_acc[n_src]["overlap"][overlap], pesq=pesq_val)

                                # PESQi: improvement over mixture baseline
                                if wav_mix is not None:
                                    tgt_w_len = wav_t_n.shape[-1]
                                    mix_for_pesq = wav_mix[:tgt_w_len]
                                    if mix_for_pesq.shape[-1] < tgt_w_len:
                                        mix_for_pesq = F.pad(mix_for_pesq, (0, tgt_w_len - mix_for_pesq.shape[-1]))
                                    mix_pesq_n = rms_normalize(mix_for_pesq)
                                    mx_16k = torchaudio.functional.resample(mix_pesq_n.unsqueeze(0), sr, PESQ_SR).squeeze(0).to(device)
                                    mx_16k = mx_16k.clamp(-1.0, 1.0)
                                    if mx_16k.shape[-1] >= int(PESQ_SR * 0.5):
                                        pesq_mix_val = pesq_m(mx_16k, t_16k).item()
                                        pesqi_val = pesq_val - pesq_mix_val
                                        if not (math.isnan(pesqi_val) or math.isinf(pesqi_val)):
                                            spk_rec["pesqi"] = round(pesqi_val, 4)
                                            _acc_add(nsrc_acc[n_src], pesqi=pesqi_val)
                                            if lang    and lang    in LANG_CATS:    _acc_add(cat_acc[n_src]["lang"][lang],       pesqi=pesqi_val)
                                            if gender  and gender  in GENDER_CATS:  _acc_add(cat_acc[n_src]["gender"][gender],   pesqi=pesqi_val)
                                            if overlap and overlap in OVERLAP_CATS: _acc_add(cat_acc[n_src]["overlap"][overlap], pesqi=pesqi_val)
                        except Exception as e:
                            if "No utterances" in str(e):
                                pesq_skip_cnt += 1
                            else:
                                print(f"  [PESQ] error spk={pi}: {e}")

                    # STOI / ESTOI / PESQ vs original wav (paper-standard: no vocoder in reference chain)
                    if has_speech and wav_orig is not None:
                        try:
                            wav_orig_n2 = rms_normalize(wav_orig)
                            if (wav_orig_n2.pow(2).mean().sqrt().item() >= _MIN_RMS
                                    and wav_p_n.shape[-1] >= int(sr * 0.5)):
                                orig_len2 = wav_orig_n2.shape[-1]
                                min_len_o2 = min(wav_p_n.shape[-1], orig_len2)
                                if min_len_o2 >= int(sr * 0.5):
                                    stoi_orig_val  = stoi_m(wav_p_n[:min_len_o2], wav_orig_n2[:min_len_o2]).item()
                                    estoi_orig_val = estoi_m(wav_p_n[:min_len_o2], wav_orig_n2[:min_len_o2]).item()
                                    spk_rec["stoi_orig"] = round(stoi_orig_val, 4)
                                    _acc_add(nsrc_acc[n_src], stoi_orig=stoi_orig_val)
                                    if lang    and lang    in LANG_CATS:    _acc_add(cat_acc[n_src]["lang"][lang],       stoi_orig=stoi_orig_val)
                                    if gender  and gender  in GENDER_CATS:  _acc_add(cat_acc[n_src]["gender"][gender],   stoi_orig=stoi_orig_val)
                                    if overlap and overlap in OVERLAP_CATS: _acc_add(cat_acc[n_src]["overlap"][overlap], stoi_orig=stoi_orig_val)
                                    if not (math.isnan(estoi_orig_val) or math.isinf(estoi_orig_val)):
                                        spk_rec["estoi_orig"] = round(estoi_orig_val, 4)
                                        _acc_add(nsrc_acc[n_src], estoi_orig=estoi_orig_val)
                                        if lang    and lang    in LANG_CATS:    _acc_add(cat_acc[n_src]["lang"][lang],       estoi_orig=estoi_orig_val)
                                        if gender  and gender  in GENDER_CATS:  _acc_add(cat_acc[n_src]["gender"][gender],   estoi_orig=estoi_orig_val)
                                        if overlap and overlap in OVERLAP_CATS: _acc_add(cat_acc[n_src]["overlap"][overlap], estoi_orig=estoi_orig_val)
                                    # Improvement over mixture baseline (vs original wav)
                                    if wav_mix is not None:
                                        mix_for_o = wav_mix[:min_len_o2]
                                        if mix_for_o.shape[-1] < min_len_o2:
                                            mix_for_o = F.pad(mix_for_o, (0, min_len_o2 - mix_for_o.shape[-1]))
                                        mix_o_n = rms_normalize(mix_for_o)
                                        if mix_o_n.shape[-1] >= int(sr * 0.5):
                                            stoi_mix_o  = stoi_m(mix_o_n, wav_orig_n2[:min_len_o2]).item()
                                            estoi_mix_o = estoi_m(mix_o_n, wav_orig_n2[:min_len_o2]).item()
                                            stoii_orig_val  = stoi_orig_val  - stoi_mix_o
                                            estoii_orig_val = estoi_orig_val - estoi_mix_o
                                            if not (math.isnan(stoii_orig_val) or math.isinf(stoii_orig_val)):
                                                spk_rec["stoii_orig"] = round(stoii_orig_val, 4)
                                                _acc_add(nsrc_acc[n_src], stoii_orig=stoii_orig_val)
                                                if lang    and lang    in LANG_CATS:    _acc_add(cat_acc[n_src]["lang"][lang],       stoii_orig=stoii_orig_val)
                                                if gender  and gender  in GENDER_CATS:  _acc_add(cat_acc[n_src]["gender"][gender],   stoii_orig=stoii_orig_val)
                                                if overlap and overlap in OVERLAP_CATS: _acc_add(cat_acc[n_src]["overlap"][overlap], stoii_orig=stoii_orig_val)
                                            if not (math.isnan(estoii_orig_val) or math.isinf(estoii_orig_val)):
                                                spk_rec["estoii_orig"] = round(estoii_orig_val, 4)
                                                _acc_add(nsrc_acc[n_src], estoii_orig=estoii_orig_val)
                                                if lang    and lang    in LANG_CATS:    _acc_add(cat_acc[n_src]["lang"][lang],       estoii_orig=estoii_orig_val)
                                                if gender  and gender  in GENDER_CATS:  _acc_add(cat_acc[n_src]["gender"][gender],   estoii_orig=estoii_orig_val)
                                                if overlap and overlap in OVERLAP_CATS: _acc_add(cat_acc[n_src]["overlap"][overlap], estoii_orig=estoii_orig_val)
                                    # PESQ vs original
                                    try:
                                        p_16k_o = torchaudio.functional.resample(wav_p_n[:min_len_o2].unsqueeze(0), sr, PESQ_SR).squeeze(0).to(device)
                                        o_16k   = torchaudio.functional.resample(wav_orig_n2[:min_len_o2].unsqueeze(0), sr, PESQ_SR).squeeze(0).to(device)
                                        p_16k_o = p_16k_o.clamp(-1.0, 1.0)
                                        o_16k   = o_16k.clamp(-1.0, 1.0)
                                        if p_16k_o.shape[-1] >= int(PESQ_SR * 0.5) and o_16k.shape[-1] >= int(PESQ_SR * 0.5):
                                            pesq_orig_val = pesq_m(p_16k_o, o_16k).item()
                                            spk_rec["pesq_orig"] = round(pesq_orig_val, 4)
                                            _acc_add(nsrc_acc[n_src], pesq_orig=pesq_orig_val)
                                            if lang    and lang    in LANG_CATS:    _acc_add(cat_acc[n_src]["lang"][lang],       pesq_orig=pesq_orig_val)
                                            if gender  and gender  in GENDER_CATS:  _acc_add(cat_acc[n_src]["gender"][gender],   pesq_orig=pesq_orig_val)
                                            if overlap and overlap in OVERLAP_CATS: _acc_add(cat_acc[n_src]["overlap"][overlap], pesq_orig=pesq_orig_val)
                                    except Exception as e:
                                        if "No utterances" not in str(e):
                                            print(f"  [PESQ orig] error spk={pi}: {e}")
                        except Exception as e:
                            print(f"  [STOI/PESQ orig] error spk={pi}: {e}")

                record[spk_key] = spk_rec

            per_sample_records.append(record)

        # Per-n_src progress report
        acc = nsrc_acc[n_src]
        print(
            f"[n_src={n_src}]  "
            f"SI-SDRi(vo)={_avg(acc,'sisdri_vo'):.2f} dB (n={acc['n_sisdri_vo']})  "
            f"SI-SDR_out(vo)={_avg(acc,'si_sdr_out_vo'):.2f} dB  "
            f"SI-SDR_mix(vo)={_avg(acc,'si_sdr_mix_vo'):.2f} dB  "
            f"ESTOi={_avg(acc,'estoi'):.4f} (n={acc['n_estoi']})  "
            f"ESTOii={_avg(acc,'estoii'):.4f} (n={acc['n_estoii']})  "
            f"SI-SNRi={_avg(acc,'sisnri'):.2f} dB (n={acc['n_sisnri']})  "
            f"SDRi(vo)={_avg(acc,'sdri_vo'):.2f} dB (n={acc['n_sdri_vo']})  "
            f"SNRi(vo)={_avg(acc,'snri_vo'):.2f} dB (n={acc['n_snri_vo']})  "
            f"STOI={_avg(acc,'stoi'):.4f} (n={acc['n_stoi']})  "
            f"PESQ={_avg(acc,'pesq'):.4f} (n={acc['n_pesq']})  "
            f"PESQi={_avg(acc,'pesqi'):.4f} (n={acc['n_pesqi']})  "
            f"Dur={_avg(acc,'dur'):.2f} s (n={acc['n_dur']})  "
            f"[dur_skipped(>{MAX_MIX_DURATION}s): {dur_skip_cnt}  PESQ skipped (no utterances): {pesq_skip_cnt}]"
        )
        if acc["n_uq_silence"] > 0:
            _fs = lambda k: f"{_avg(acc, k):.4f}" if acc[f'n_{k}'] > 0 else "N/A"
            print(
                f"  [STO-sigma n_src={n_src}]  "
                f"S(cond)={_fs('uq_silence')}  T(cond)={_fs('uq_target')}  O(cond)={_fs('uq_overlap')}  ||  "
                f"S(uncond)={_fs('uq_uncond_silence')}  T(uncond)={_fs('uq_uncond_target')}  O(uncond)={_fs('uq_uncond_overlap')}"
            )

    del stoi_m, estoi_m, pesq_m, sisnr_m, sisdr_m, sdr_m
    gc.collect()
    torch.cuda.empty_cache()

    return nsrc_acc, cat_acc, per_sample_records


# ─── Finalize: print tables + build results dict ──────────────────────────────



# ─── Checkpoint loading ───────────────────────────────────────────────────────
def load_checkpoint(ckpt_path, device):
    try:
        torch.serialization.add_safe_globals([np.core.multiarray.scalar])
    except AttributeError:
        pass
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    print(f"[CKPT] Keys: {list(ckpt.keys())}")
    print(f"[CKPT] Step: {ckpt.get('step', 'N/A')}")

    backbone = AudioFlow(config).to(device)
    uq_head = UncertaintyHead(config).to(device)

    def _strip_flow_prefix(sd):
        if all(k.startswith("flow.") for k in sd):
            return {k[len("flow."):]: v for k, v in sd.items()}
        return sd

    bb_key = next((k for k in ("backbone", "backbone_frozen", "model") if k in ckpt), None)
    if bb_key:
        backbone.load_state_dict(_strip_flow_prefix(ckpt[bb_key]))
        print(f"[CKPT] Backbone loaded from key '{bb_key}'")
    else:
        raise KeyError(f"No backbone key found in checkpoint. Keys: {list(ckpt.keys())}")

    if "uq_head" in ckpt:
        uq_head.load_state_dict(ckpt["uq_head"])
        print("[CKPT] UQ head loaded")

    stno_proj = None
    if "stno_proj" in ckpt and ckpt["stno_proj"]:
        stno_proj = STNOProjection(config.model.n_dim).to(device)
        stno_proj.load_state_dict(ckpt["stno_proj"])
        print("[CKPT] STNOProjection loaded")

    return backbone, uq_head, stno_proj


# ─── Result saving ────────────────────────────────────────────────────────────
def _save_results(results, out_dir, ckpt_path, args):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_sample = results.pop("per_sample", [])
    per_sample_path = out_dir / "per_sample_metrics.json"
    with open(per_sample_path, "w") as f:
        json.dump(per_sample, f, indent=2)
    print(f"[EVAL] Per-sample metrics saved to {per_sample_path}  ({len(per_sample)} records)")

    uq_json_records = []
    for rec in per_sample:
        for s in range(1, rec["n_src"] + 1):
            spk_key = f"s{s}"
            if spk_key not in rec:
                continue
            sd = rec[spk_key]
            uq_json_records.append({
                "stem": rec["stem"], "n_src": rec["n_src"],
                "lang": rec["lang"], "gender": rec["gender"], "overlap": rec["overlap"],
                "speaker_slot": spk_key,
                "sisdri_vo":       sd.get("sisdri_vo"),
                "si_sdr_out_vo":   sd.get("si_sdr_out_vo"),
                "si_sdr_mix_vo":   sd.get("si_sdr_mix_vo"),
                "estoi":           sd.get("estoi"),
                "estoii":          sd.get("estoii"),
                "si_snri":         sd.get("si_snri"),
                "si_snr_out":      sd.get("si_snr_out"),
                "si_snr_mix":      sd.get("si_snr_mix"),
                "sdri_vo":         sd.get("sdri_vo"),
                "snri_vo":         sd.get("snri_vo"),
                "sisdri_orig":     sd.get("sisdri_orig"),
                "sdri_orig":       sd.get("sdri_orig"),
                "stoi":            sd.get("stoi"),
                "stoii":           sd.get("stoii"),
                "pesq":            sd.get("pesq"),
                "pesqi":           sd.get("pesqi"),
                "stoi_orig":       sd.get("stoi_orig"),
                "stoii_orig":      sd.get("stoii_orig"),
                "estoi_orig":      sd.get("estoi_orig"),
                "estoii_orig":     sd.get("estoii_orig"),
                "pesq_orig":       sd.get("pesq_orig"),
                "dur":             sd.get("dur"),
                "uq_sigma_mean":   sd.get("uq_sigma_mean"),
                "uq_sigma_active": sd.get("uq_sigma_active"),
            })
    uq_json_path = out_dir / "per-sample-uncertainty.json"
    with open(uq_json_path, "w") as f:
        json.dump(uq_json_records, f, indent=2)
    print(f"[EVAL] Per-sample uncertainty JSON saved to {uq_json_path}  ({len(uq_json_records)} records)")

    results_path = out_dir / "test_metrics_analyze.json"
    results["checkpoint"] = str(ckpt_path)
    results["stage"] = args.stage
    results["ablation"] = args.ablation
    results["cap"] = args.cap
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[EVAL] Aggregate results saved to {results_path}")


# ─── Multi-GPU helpers ────────────────────────────────────────────────────────
def _build_test_loaders(args, shard_rank=0, shard_world=1):
    """Build per-n_src DataLoaders, sharded for multi-GPU workers."""
    test_loaders = {}
    for n in N_SRC_LIST:
        ds = make_dataset(setting=f"librittskor{n}", cache_path="/tmp", subset="test")
        n_total = len(ds)
        n_use = min(args.cap, n_total)
        all_indices = list(range(n_use))[:TEST_CAP + 1]
        if shard_world > 1:
            indices = all_indices[shard_rank::shard_world]
        else:
            indices = all_indices
        ds_use = Subset(ds, indices)
        test_loaders[n] = DataLoader(
            ds_use, batch_size=1, shuffle=False,
            num_workers=2, collate_fn=ds.collate,
        )
        print(f"  n_src={n}: {len(indices)} samples"
              + (f" (shard {shard_rank+1}/{shard_world})" if shard_world > 1 else f" / {n_total} total"))
    return test_loaders


def _multi_gpu_eval(ckpt_path, args, gpu_list, out_dir):
    """Spawn one subprocess per GPU; each evaluates its data shard; main merges."""
    temp_dir = Path(tempfile.mkdtemp(prefix="eval_mgpu_"))
    procs, temp_files = [], []
    try:
        for rank, gid in enumerate(gpu_list):
            tmp_pkl = str(temp_dir / f"rank{rank}.pkl")
            temp_files.append(tmp_pkl)
            cmd = [
                sys.executable, __file__,
                "--ckpt",        str(ckpt_path),
                "--gpu",         str(gid),
                "--stage",       str(args.stage),
                "--ablation",    args.ablation,
                "--cap",         str(args.cap),
                "--out",         str(out_dir),
                f"--shard-rank={rank}",
                f"--shard-world={len(gpu_list)}",
                f"--temp-out={tmp_pkl}",
            ] + (["--save-wav"] if args.save_wav else []) \
              + (["--steps", str(args.steps)] if getattr(args, 'steps', None) else []) \
              + (["--lambda-fixed", str(args.lambda_fixed)] if getattr(args, 'lambda_fixed', None) is not None else []) \
              + (["--lambda-scale", str(args.lambda_scale)] if getattr(args, 'lambda_scale', None) is not None else [])
            print(f"[EVAL] Spawning worker rank={rank} on GPU {gid}")
            procs.append(subprocess.Popen(cmd))

        exit_codes = [p.wait() for p in procs]
        if any(c != 0 for c in exit_codes):
            print(f"[ERROR] Some workers failed: exit codes = {exit_codes}")
            sys.exit(1)

        print("\n[EVAL] All workers done. Merging results...")
        merged_nsrc = {n: _make_acc() for n in N_SRC_LIST}
        merged_cat = {
            n: {
                "lang":    {c: _make_acc() for c in LANG_CATS},
                "gender":  {c: _make_acc() for c in GENDER_CATS},
                "overlap": {c: _make_acc() for c in OVERLAP_CATS},
            }
            for n in N_SRC_LIST
        }
        merged_per_sample = []
        for pkl in temp_files:
            with open(pkl, "rb") as f:
                nsrc_acc, cat_acc, per_sample = pickle.load(f)
            for n in N_SRC_LIST:
                _merge_acc(merged_nsrc[n], nsrc_acc[n])
                for c in LANG_CATS:    _merge_acc(merged_cat[n]["lang"][c],    cat_acc[n]["lang"][c])
                for c in GENDER_CATS:  _merge_acc(merged_cat[n]["gender"][c],  cat_acc[n]["gender"][c])
                for c in OVERLAP_CATS: _merge_acc(merged_cat[n]["overlap"][c], cat_acc[n]["overlap"][c])
            merged_per_sample.extend(per_sample)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    results = _finalize_results(merged_nsrc, merged_cat, merged_per_sample, N_SRC_LIST)
    _save_results(results, out_dir, ckpt_path, args)


def _single_gpu_eval(ckpt_path, args, gpu_id, out_dir, shard_rank=0, shard_world=1):
    """Run evaluation on one GPU (possibly a shard for multi-GPU mode)."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[EVAL] Device: {device}  GPU={gpu_id}")

    apply_forward_patches()
    print(f"[EVAL] Loading {ckpt_path} for stage {args.stage}  ablation={args.ablation}")
    backbone, uq_head, stno_proj = load_checkpoint(ckpt_path, device)

    print(f"\n[EVAL] Building test loaders (cap={args.cap} per n_src)...")
    test_loaders = _build_test_loaders(args, shard_rank=shard_rank, shard_world=shard_world)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[EVAL] Output directory: {out_dir}")

    return _eval_loop(
        backbone, uq_head, stno_proj, test_loaders, device,
        stage=args.stage, out_dir=str(out_dir), save_wav=args.save_wav,
        ablation=args.ablation, eval_steps=getattr(args, 'steps', None),
        lambda_fixed=getattr(args, 'lambda_fixed', None),
        lambda_scale=getattr(args, 'lambda_scale', None),
    )


def _eval_one_ckpt(ckpt_path, args, gpu_list, out_dir):
    """Evaluate a single checkpoint (dispatches single- or multi-GPU)."""
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    out_dir = Path(out_dir)
    is_worker = args.shard_world > 1 and args.temp_out is not None

    if len(gpu_list) > 1 and not is_worker:
        _multi_gpu_eval(ckpt_path, args, gpu_list, out_dir)
    else:
        gpu_id = gpu_list[0]
        nsrc_acc, cat_acc, per_sample = _single_gpu_eval(
            ckpt_path, args, gpu_id, out_dir,
            shard_rank=args.shard_rank, shard_world=args.shard_world,
        )
        if is_worker:
            # Worker mode: dump accumulators for the main process to merge
            with open(args.temp_out, "wb") as f:
                pickle.dump((nsrc_acc, cat_acc, per_sample), f)
            print(f"[WORKER rank={args.shard_rank}] Saved shard results to {args.temp_out}")
        else:
            results = _finalize_results(nsrc_acc, cat_acc, per_sample, N_SRC_LIST)
            _save_results(results, out_dir, ckpt_path, args)


# ─── Entry point ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate stage-1 separation model on LibriTTS-KOR test set."
    )
    parser.add_argument("--ckpt", default=str(DEFAULT_CKPT),
                        help="Path to .pt checkpoint, OR a directory to evaluate all .pt files inside")
    parser.add_argument("--gpu", default="4",
                        help="Single GPU id ('4') or comma-separated list for multi-GPU ('0,1,2,3')")
    parser.add_argument("--stage",   default=2, type=int, choices=[1, 2])
    parser.add_argument("--ablation", default="full",
                        choices=["full", "no_uq", "no_stno", "no_entropy", "no_stno_no_uq"],
                        help=(
                            "Ablation mode for stage-2 evaluation:\n"
                            "  full          — oracle STNO + UQ guidance (λ* adaptive)  [full model]\n"
                            "  no_uq         — oracle STNO, no UQ guidance (fixed α=1.5) [same ckpt]\n"
                            "  no_stno       — UQ guidance on, STNO condition=None       [same ckpt]\n"
                            "  no_entropy    — oracle STNO + UQ guidance, but separate   [no-entropy ckpt]\n"
                            "  no_stno_no_uq — Stage2 backbone only, no STNO, fixed α=1.5 [same ckpt]\n"
                        ))
    parser.add_argument("--cap",     default=TEST_CAP, type=int,
                        help="Max test samples per n_src")
    parser.add_argument("--out",     default=None,
                        help="Output directory (default: DEFAULT_OUT_DIR)")
    parser.add_argument("--save-wav", action="store_true",
                        help="Save separated and mixture wav files under out_dir/wav/")
    parser.add_argument("--steps", type=int, default=None,
                        help=f"ODE Euler steps (default: EVAL_STEPS={EVAL_STEPS})")
    parser.add_argument("--lambda-fixed", type=float, default=None, dest="lambda_fixed",
                        help=(
                            "Fixed CFG guidance scale for lambda sweep. "
                            "Overrides both UQ guidance and alpha. "
                            "0.0=conditioned only (no CFG), 1.5=same as default alpha. "
                            "E.g.: --lambda-fixed 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0"
                        ))
    parser.add_argument("--lambda-scale", type=float, default=None, dest="lambda_scale",
                        help=(
                            "Enable diet/UQ-only guidance formula: λ*=clamp(σ_cond.mean×scale, max=2.0). "
                            "Use with --ablation no_stno for 0828-best-diet.py checkpoints. "
                            "Typical value: 5.0  (matches training default lambda_scale=5.0)"
                        ))
    # Hidden args used internally by multi-GPU subprocess workers
    parser.add_argument("--shard-rank",  type=int, default=0,  help=argparse.SUPPRESS)
    parser.add_argument("--shard-world", type=int, default=1,  help=argparse.SUPPRESS)
    parser.add_argument("--temp-out",    default=None,          help=argparse.SUPPRESS)
    args = parser.parse_args()

    gpu_list = [int(x.strip()) for x in args.gpu.split(",")]
    ckpt_path = Path(args.ckpt)

    # Directory mode: evaluate every .pt file found inside
    if ckpt_path.is_dir() and args.shard_world == 1:
        ckpt_files = sorted(ckpt_path.glob("*.pt"))
        if not ckpt_files:
            raise FileNotFoundError(f"No .pt files found in {ckpt_path}")
        print(f"[EVAL] Found {len(ckpt_files)} checkpoints in {ckpt_path}:")
        for f in ckpt_files:
            print(f"  {f.name}")
        base_out = Path(args.out) if args.out else DEFAULT_OUT_DIR
        for cf in ckpt_files:
            out_sub = base_out / cf.stem
            print(f"\n{'='*80}")
            print(f"[EVAL] === Checkpoint: {cf.name} ===")
            _eval_one_ckpt(cf, args, gpu_list, out_sub)
        return

    out_dir = Path(args.out) if args.out else DEFAULT_OUT_DIR
    _eval_one_ckpt(ckpt_path, args, gpu_list, out_dir)


if __name__ == "__main__":
    main()
