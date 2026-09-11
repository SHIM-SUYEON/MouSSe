import os
import sys

# Make the top-level ``mousse`` package importable when running this script
# directly (repo root is the parent of this training/ directory).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import shutil
import time
import math
import random
import gc
from itertools import permutations
from pathlib import Path
from typing import Dict, Optional, Any

import numpy as np
import torch
import torch.nn.functional as F
import torch.autograd.forward_ad as fwad
from einops import rearrange, reduce
from tqdm import tqdm
from accelerate import Accelerator, DistributedDataParallelKwargs
import wandb
import torchaudio
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, Subset
from torchmetrics.audio import (
    ShortTimeObjectiveIntelligibility,
    PerceptualEvaluationSpeechQuality,
    ScaleInvariantSignalNoiseRatio,
)

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
except ImportError:
    raise RuntimeError("Install optuna: pip install optuna")

from mousse.configs import config
from mousse.models import AudioFlow, UncertaintyHead
from mousse.utils.dataset import make_dataset
from mousse.models.patches import apply_forward_patches
from mousse.utils.vocoder import get_vocoder, do_vocoder


# from pyannote.audio import Pipeline

# -----------------------------------------------------------------
# Paths  (same as train_b1_v2_libri.py)
# -----------------------------------------------------------------
PRETRAINED_BACKBONE_PATH = Path(
    "/shared/NAS_SSD/syshim/hi-d/bluejays/ss-librimix/librimix-sep-01.pt"
)
STAGE1_SAVE_PATH = Path(
    "/shared/NAS_SSD/syshim/hi-d/bluejays/speechflowuq/supervoice_separate/racoon-aug/single-stage/0826/"
)
HP_SEARCH_PATH = Path(
    "/shared/NAS_SSD/syshim/hi-d/bluejays/speechflowuq/supervoice_separate/racoon-aug/hyper/stage1/"
)
# FT_VOCODER_PATH = Path(
#     "/shared/NAS_SSD/syshim/hi-d/bluejays/speechflowuq/vocoder_ft/vocoder_ft.pt"
# )
FT_VOCODER_PATH = None
OPTUNA_DB = f"sqlite:///{HP_SEARCH_PATH}/optuna.db"

for _p in [STAGE1_SAVE_PATH, HP_SEARCH_PATH]:
    os.makedirs(_p, exist_ok=True)

print(f"STAGE1_SAVE_PATH: {STAGE1_SAVE_PATH}")

# -----------------------------------------------------------------
# HP search control
# -----------------------------------------------------------------
HP_SEARCH_MODE = False #False  # search done — trial #14 won (11.03 dB mel-SI-SNRi)

STAGE1_SEARCH_STEPS      = 10000
STAGE1_SEARCH_EVAL_EVERY = 2500
STAGE1_N_TRIALS          = 20

HP_SEARCH_EVAL_N = 50   # val samples per n_src for fast mel proxy

STAGE1_FULL_STEPS    = 120000
STAGE1_FULL_SAVE_EV  = [30000, 47500, 60000] # 10000
STAGE1_FULL_EVAL_EV  = 1000
TRAIN_LOG_EVERY      = 1

# Early stopping (Stage 1 full-training only)
# Fires after EARLY_STOP_PATIENCE_EVALS consecutive evals without ≥ EARLY_STOP_MIN_DELTA_DB gain.
# At eval_every=1000 and patience=20 this means 20 000 steps without meaningful improvement.
EARLY_STOP_PATIENCE_EVALS = 20
EARLY_STOP_MIN_DELTA_DB   = 0.2

STAGE1_EXPERIMENT = "large-sep-0820"

MANUAL_STAGE1_HP: Dict[str, Any] = {
    "lr_max": 1.40251473845546e-05,
    "lr_start": 2.6475736370673437e-08,
    "warmup_steps": 4348,
    "clip_norm": 0.08034961599688825,
    "sigma": 3.2629595908661235e-06,
    "batch_size": 4,
    "grad_accum": 4,
}

# -----------------------------------------------------------------
# Dataset / training constants
# -----------------------------------------------------------------
N_SRC_LIST            = [2, 3, 4]
VAL_SAMPLES_PER_NSRC  = 100 #1300    # 3 x 200 = 600 total val (was 1300, reduced for speed)
TRAIN_LOADER_WORKERS  = 5
TRAIN_MIXED_PRECISION = "bf16"
MAX_FRAMES_PER_SPK    = 600 #300
MAX_EVAL_FRAMES_PER_SPK = 1500 #1200

train_project = "mousse-stage1"

# -----------------------------------------------------------------
# Pyannote 3.1
# -----------------------------------------------------------------

# pipeline = Pipeline.from_pretrained(
#   "pyannote/speaker-diarization-3.1",
#   use_auth_token="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")
# pipeline.to(torch.device("cuda"))
# diarization = pipeline("audio.wav", min_speakers=2, max_speakers=4)

# -----------------------------------------------------------------
# Loss functions
# -----------------------------------------------------------------
def pit_mse_loss(pred, target, n_src, mask=None):
    B, L, D = pred.shape
    T = L // n_src
    pred_slots   = [pred[:, s*T:(s+1)*T, :]   for s in range(n_src)]
    target_slots = [target[:, s*T:(s+1)*T, :] for s in range(n_src)]
    mask_slots   = (
        [mask[:, s*T:(s+1)*T] for s in range(n_src)]
        if mask is not None else [None] * n_src
    )
    best_loss = None
    for perm in permutations(range(n_src)):
        perm_terms = []
        for i in range(n_src):
            slot_loss = F.mse_loss(pred_slots[i], target_slots[perm[i]], reduction="none").mean(dim=-1)
            m = mask_slots[perm[i]]
            if m is not None:
                nvalid = m.float().sum().clamp(min=1)
                slot_loss = (slot_loss * m.float()).sum() / nvalid
            else:
                slot_loss = slot_loss.mean()
            perm_terms.append(slot_loss)
        perm_loss = torch.stack(perm_terms).mean()
        if best_loss is None or perm_loss < best_loss:
            best_loss = perm_loss
    return best_loss


def pit_nll_loss(pred_mean, pred_log_sigma, target, n_src, mask=None):
    B, L, D = pred_mean.shape
    T = L // n_src
    mean_slots = [pred_mean[:, s*T:(s+1)*T, :]      for s in range(n_src)]
    lsig_slots = [pred_log_sigma[:, s*T:(s+1)*T, :] for s in range(n_src)]
    tgt_slots  = [target[:, s*T:(s+1)*T, :]         for s in range(n_src)]
    mask_slots = (
        [mask[:, s*T:(s+1)*T] for s in range(n_src)]
        if mask is not None else [None] * n_src
    )
    var_slots = [torch.exp(ls) ** 2 for ls in lsig_slots]
    best_loss = None
    for perm in permutations(range(n_src)):
        perm_terms = []
        for i in range(n_src):
            j = perm[i]
            error_sq = (mean_slots[i] - tgt_slots[j]) ** 2
            nll = error_sq / (2 * var_slots[i]) + lsig_slots[i]
            m = mask_slots[j]
            if m is not None:
                m_exp = m.unsqueeze(-1).expand_as(nll)
                nvalid = m_exp.float().sum().clamp(min=1)
                perm_terms.append((nll * m_exp.float()).sum() / nvalid)
            else:
                perm_terms.append(nll.mean())
        perm_loss = torch.stack(perm_terms).mean()
        if best_loss is None or perm_loss < best_loss:
            best_loss = perm_loss
    return best_loss







def fast_euler_sample_uq(
    predictor, uq_head, audio: torch.Tensor,
    steps: int = 16, alpha: float = 1.5,
    lambda_max: float = 2.0, use_uq_guidance: bool = True,
    condition=None,
) -> torch.Tensor:
    """Fixed-step Euler ODE integration for evaluation.
    Skips JVP and adaptive stepping (U is discarded by callers anyway),
    reducing cost to 4 forward passes/step (use_uq_guidance=True)
    or 2 passes/step (use_uq_guidance=False, alpha!=0)
    vs. 6 passes * up to 128 steps in sample_with_uncertainty."""
    device = audio.device
    batch_size = audio.shape[0]
    z = torch.randn_like(audio)
    dt = 1.0 / steps
    audio_empty = torch.zeros_like(audio)

    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for i in range(steps):
                t_batch = torch.full((batch_size,), i * dt, device=device, dtype=audio.dtype)
                v_cond   = predictor(audio=audio,       noise=z, times=t_batch, condition=condition)
                v_uncond = predictor(audio=audio_empty, noise=z, times=t_batch)

                if use_uq_guidance:
                    log_s_uncond = uq_head(audio=audio_empty, noise=z, times=t_batch)
                    log_s_cond   = uq_head(audio=audio,       noise=z, times=t_batch, condition=condition)
                    sigma_uncond = torch.exp(log_s_uncond)
                    sigma_cond   = torch.exp(log_s_cond)
                    num = torch.sum(sigma_uncond * sigma_cond - sigma_cond ** 2, dim=-1, keepdim=True)
                    den = torch.sum((sigma_cond - sigma_uncond) ** 2, dim=-1, keepdim=True).clamp(min=1e-8)
                    lambda_star = torch.clamp(num / den, min=0.0, max=lambda_max)
                    v_guided = v_cond + (v_cond - v_uncond) * lambda_star
                else:
                    v_guided = v_cond + (v_cond - v_uncond) * alpha

                z = z + v_guided.float() * dt
    return z


# -----------------------------------------------------------------
# Eval stats for wandb
# -----------------------------------------------------------------
def compute_eval_stats(
    predictor, val_loaders_dict: dict, device, sigma: float = 1e-5, n_per_nsrc: int = 5,
) -> Dict[str, float]:
    predictor.eval()
    total_loss = 0.0
    n_done = 0
    pred_bufs: list = []
    tgt_bufs:  list = []

    with torch.no_grad():
        for n_src, val_loader in val_loaders_dict.items():
            val_iter = iter(val_loader)
            for _ in range(min(n_per_nsrc, len(val_loader.dataset))):
                try:
                    batch = next(val_iter)
                except StopIteration:
                    break
                cond = batch[0].to(device)
                tgt  = batch[1].to(device)
                
                # Dynamic determination of single speaker frames
                if tgt.ndim == 4:
                    T_single = tgt.shape[2]
                else:
                    T_single = cond.shape[1] // n_src
                    
                # Enforce frame length truncation rule
                if T_single > MAX_FRAMES_PER_SPK:
                    T_single = MAX_FRAMES_PER_SPK

                T_total = T_single * n_src

                if tgt.ndim == 4:
                    tgt_flat = rearrange(tgt, "b s t d -> b (s t) d")
                    cond_flat = torch.repeat_interleave(cond, repeats=n_src, dim=1)
                else:
                    tgt_flat = tgt
                    cond_flat = cond

                # Slice time dimensions to the target T_total limit
                tgt_flat = tgt_flat[:, :T_total, :]
                cond_flat = cond_flat[:, :T_total, :]

                norm_cond = (cond_flat - config.audio.norm_mean) / config.audio.norm_std
                norm_tgt  = (tgt_flat  - config.audio.norm_mean) / config.audio.norm_std

                B_orig = norm_tgt.shape[0]
                # Fixed t=0.5 gives deterministic, low-variance eval_loss across calls.
                # Random t (old) caused 2× oscillation because MSE(t≈0) >> MSE(t≈1).
                u       = torch.full((B_orig,), 0.5, dtype=torch.float32, device=device)
                t_bc    = rearrange(u, "b -> b 1 1").expand_as(norm_tgt)
                noise   = torch.randn_like(norm_tgt)
                z       = (1.0 - (1.0 - sigma) * t_bc) * noise + t_bc * norm_tgt
                vel_tgt = norm_tgt - (1.0 - sigma) * noise

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    pred = predictor(audio=norm_cond, noise=z, times=u)
                
                # Compute MSE loss based on time-concatenated 1:1 Target Speaker Extraction
                loss = F.mse_loss(pred.float(), vel_tgt.float())
                total_loss += loss.item()
                
                recon = (pred + (1.0 - sigma) * noise).detach().cpu()
                pred_bufs.append(recon.flatten())
                tgt_bufs.append(norm_tgt.cpu().flatten())
                n_done += 1

    pred_all = torch.cat(pred_bufs) if pred_bufs else torch.tensor([0.0])
    tgt_all  = torch.cat(tgt_bufs)  if tgt_bufs  else torch.tensor([0.0])
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "eval_loss":      total_loss / max(1, n_done),
        "predicted/max":  pred_all.max().item(),
        "predicted/mean": pred_all.mean().item(),
        "predicted/min":  pred_all.min().item(),
        "target/max":     tgt_all.max().item(),
        "target/mean":    tgt_all.mean().item(),
        "target/min":     tgt_all.min().item(),
    }


# -----------------------------------------------------------------
# Full waveform evaluation
# -----------------------------------------------------------------


# -----------------------------------------------------------------
# Model patches
# -----------------------------------------------------------------


# -----------------------------------------------------------------
# Model patches
# -----------------------------------------------------------------
def patched_sample_with_uncertainty(self, *, audio, uq_head, steps, alpha=1.5, lambda_max=2.0, use_uq_guidance=False):
    device     = audio.device
    batch_size = audio.shape[0]
    z          = torch.randn_like(audio)
    U          = torch.zeros_like(audio)
    dt_nominal = 1.0 / steps
    gamma      = 10.0
    t_curr     = 0.0
    MAX_ITERS  = steps * 8
    n_iter     = 0

    while t_curr < 1.0 - 1e-6 and n_iter < MAX_ITERS:
        n_iter += 1
        t_batch     = torch.full((batch_size,), t_curr, device=device, dtype=audio.dtype)
        audio_empty = torch.zeros_like(audio)

        with torch.no_grad():
            v_uncond     = self.forward(audio=audio_empty, noise=z, times=t_batch)
            v_cond       = self.forward(audio=audio,       noise=z, times=t_batch)
            log_s_uncond = uq_head(audio=audio_empty, noise=z, times=t_batch)
            log_s_cond   = uq_head(audio=audio,       noise=z, times=t_batch)

        sigma_uncond = torch.exp(log_s_uncond)
        sigma_cond   = torch.exp(log_s_cond)

        num = torch.sum(sigma_uncond * sigma_cond - sigma_cond ** 2, dim=-1, keepdim=True)
        den = torch.sum((sigma_cond - sigma_uncond) ** 2, dim=-1, keepdim=True).clamp(min=1e-8)
        lambda_star  = torch.clamp(num / den, min=0.0, max=lambda_max)
        var_velocity = ((1.0 + lambda_star) * sigma_cond - lambda_star * sigma_uncond) ** 2

        uq_mean_var = torch.mean(var_velocity).item()
        dt_dynamic  = dt_nominal / (1.0 + gamma * uq_mean_var)
        dt_dynamic  = max(dt_dynamic, dt_nominal / 4)
        dt_dynamic  = min(dt_dynamic, 1.0 - t_curr)

        if use_uq_guidance:
            v_guided = v_cond + (v_cond - v_uncond) * lambda_star
        else:
            v_guided = v_cond + (v_cond - v_uncond) * alpha

        rademacher = torch.randint(0, 2, z.shape, device=device, dtype=z.dtype) * 2 - 1
        std_state  = torch.sqrt(U.clamp(min=1e-8))
        v_vector   = std_state * rademacher

        def local_velocity_fn(z_input):
            t_in = torch.full((batch_size,), t_curr, device=device, dtype=audio.dtype)
            v_u = self.forward(audio=torch.zeros_like(audio), noise=z_input, times=t_in)
            v_c = self.forward(audio=audio, noise=z_input, times=t_in)
            return v_c + (v_c - v_u) * alpha

        with fwad.dual_level():
            z_dual   = fwad.make_dual(z, v_vector)
            out_dual = local_velocity_fn(z_dual)
            _, jvp_out = fwad.unpack_dual(out_dual)

        if jvp_out is None:
            jvp_out = torch.zeros_like(z)

        U      = U + (var_velocity * dt_dynamic) ** 2 + 2.0 * dt_dynamic * (v_vector * jvp_out)
        z      = z + v_guided * dt_dynamic
        t_curr = t_curr + dt_dynamic

    return z, torch.mean(U, dim=-1)


def apply_training_patches():
    """Install the shared forward patches, plus the training-only
    uncertainty-aware sampler (AudioFlow.sample_with_uncertainty)."""
    sf_model = apply_forward_patches()
    sf_model.AudioFlow.sample_with_uncertainty = patched_sample_with_uncertainty


def cycle(dl):
    while True:
        for data in dl:
            yield data


def _make_train_collate(base_collate, max_frames_per_spk, hop_size):
    """Wraps dataset collate to randomly crop each waveform to max_frames_per_spk
    before spectrogram computation.  This (a) exposes different overlap regions on
    each pass through the data for clips longer than the crop window, and (b) prevents
    large zero-padded tensors from being created and transferred to the GPU only to be
    immediately discarded in the training loop."""
    max_samples = max_frames_per_spk * hop_size

    def collate_with_crop(batch):
        cropped = []
        for item in batch:
            wav = item["wav"]        # [1, T]
            tgt = item["targets"]    # [n_src, T]
            T = wav.shape[-1]
            if T > max_samples:
                start = random.randint(0, T - max_samples)
                wav = wav[:, start:start + max_samples]
                tgt = tgt[:, start:start + max_samples]
                item = {**item, "wav": wav, "targets": tgt}
            cropped.append(item)
        return base_collate(cropped)

    return collate_with_crop


def load_datasets(batch_size: int, val_cap: int = VAL_SAMPLES_PER_NSRC):
    train_loaders: Dict[int, DataLoader] = {}
    val_loaders:   Dict[int, DataLoader] = {}
    for n in N_SRC_LIST:
        setting = f"mussed{n}"
        ds_tr = make_dataset(setting=setting, cache_path="/tmp", subset="train")
        ds_va = make_dataset(setting=setting, cache_path="/tmp", subset="dev")
        if len(ds_va) > val_cap:
            ds_va_use  = Subset(ds_va, list(range(val_cap)))
            collate_fn = ds_va.collate
        else:
            ds_va_use  = ds_va
            collate_fn = ds_va.collate
        train_loaders[n] = DataLoader(
            ds_tr, batch_size=batch_size, shuffle=True,
            num_workers=TRAIN_LOADER_WORKERS,
            collate_fn=_make_train_collate(ds_tr.collate, MAX_FRAMES_PER_SPK, config.audio.hop_size),
            pin_memory=True,
        )
        val_loaders[n] = DataLoader(
            ds_va_use, batch_size=batch_size, shuffle=False,
            num_workers=2, collate_fn=collate_fn,
        )
        print(f"  n_src={n}: train={len(ds_tr)}  val={len(ds_va_use)}")
    return train_loaders, val_loaders


def _load_pretrained_backbone(backbone, path: Path):
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    backbone_key = next((k for k in ("backbone", "backbone_frozen", "model") if k in ckpt), None)
    if backbone_key:
        raw_sd = ckpt[backbone_key]
        if all(k.startswith("flow.") for k in raw_sd):
            raw_sd = {k[len("flow."):]: v for k, v in raw_sd.items()}
        backbone.load_state_dict(raw_sd)
        print(f"  Backbone loaded from key '{backbone_key}' in {path.name}")
    else:
        print(f"[WARN] No recognised backbone key in {path.name}")
    return backbone


def eval_mel_sisnri(backbone, val_loaders: dict, device, n_eval: int = HP_SEARCH_EVAL_N) -> float:
    backbone.eval()
    sisnr_m      = ScaleInvariantSignalNoiseRatio().to(device)
    total_sisnri = 0.0
    n_total      = 0
    T_cap        = 300

    with torch.no_grad():
        for n_src, val_loader in val_loaders.items():
            val_iter = iter(val_loader)
            for _ in range(min(n_eval, len(val_loader.dataset))):
                try:
                    condition_spec, flow_target, *_ = next(val_iter)
                except StopIteration:
                    break
                cond = condition_spec[0:1].to(device)
                tgt  = flow_target[0:1].to(device)
                T    = min(cond.shape[1] // n_src, T_cap)
                cond = cond[:, :T * n_src, :]
                tgt  = tgt[:, :T * n_src, :]
                norm_cond = (cond - config.audio.norm_mean) / config.audio.norm_std
                norm_tgt  = (tgt  - config.audio.norm_mean) / config.audio.norm_std
                z = torch.randn_like(norm_cond).to(device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    for step_i in range(16):
                        t_val = step_i / 16
                        t_b   = torch.full((1,), t_val, device=device, dtype=torch.bfloat16)
                        v     = backbone(audio=norm_cond.to(torch.bfloat16), noise=z.to(torch.bfloat16), times=t_b)
                        z = z + v.float() * (1.0 / 16)
                pred_slots = [z[0, s*T:(s+1)*T, :].flatten() for s in range(n_src)]
                tgt_slots  = [norm_tgt[0, s*T:(s+1)*T, :].float().flatten() for s in range(n_src)]
                mix_ref    = norm_cond[0, :T, :].float().flatten()
                cost = np.zeros((n_src, n_src))
                for pi in range(n_src):
                    for ti in range(n_src):
                        cost[pi, ti] = -sisnr_m(pred_slots[pi], tgt_slots[ti]).item()
                pi_idx, ti_idx = linear_sum_assignment(cost)
                for pi, ti in zip(pi_idx, ti_idx):
                    total_sisnri += (sisnr_m(pred_slots[pi], tgt_slots[ti]).item() - sisnr_m(mix_ref, tgt_slots[ti]).item())
                    n_total += 1

    del sisnr_m
    gc.collect()
    torch.cuda.empty_cache()
    return total_sisnri / max(1, n_total)


# -----------------------------------------------------------------
# Core training loop
# -----------------------------------------------------------------
def run_training(
    hp: Dict[str, Any], steps: int, eval_every: int, save_every: int,
    train_loaders: dict, val_loaders: dict, device: torch.device,
    experiment_name: str, save_path: Path, accelerator=None, trial=None,
    full_eval: bool = False, pretrained_path: Optional[Path] = None,
) -> float:
    try:
        torch.serialization.add_safe_globals([np.core.multiarray.scalar])
    except AttributeError:
        pass

    backbone = AudioFlow(config).to(device)
    uq_head  = UncertaintyHead(config).to(device)

    init_path = pretrained_path or PRETRAINED_BACKBONE_PATH
    if Path(init_path).exists():
        _load_pretrained_backbone(backbone, Path(init_path))

    backbone.train()
    uq_head.train()
    print(f"Backbone: {sum(p.numel() for p in backbone.parameters()):,}, "
          f"UQ head: {sum(p.numel() for p in uq_head.parameters()):,}")

    lr_max       = hp["lr_max"]
    lr_start     = hp.get("lr_start", lr_max * 0.01)
    grad_accum   = hp.get("grad_accum", 2)
    clip_norm    = hp.get("clip_norm", 0.2)
    sigma        = hp.get("sigma", 1e-5)
    warmup_steps = hp.get("warmup_steps", 3000)

    if accelerator is not None:
        lr_max   *= accelerator.num_processes
        lr_start *= accelerator.num_processes

    optim = torch.optim.AdamW(
        [
            {"params": list(backbone.parameters()), "lr": lr_max},
            {"params": list(uq_head.parameters()),  "lr": lr_max},
        ],
        betas=[0.9, 0.99], weight_decay=0.01, eps=1e-7,
    )
    _warmup = torch.optim.lr_scheduler.LinearLR(optim, start_factor=max(lr_start / lr_max, 1e-9), end_factor=1.0, total_iters=warmup_steps)
    _plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="max", factor=0.5, patience=10, min_lr=lr_start * 10, threshold=0.1)

    if accelerator is not None:
        all_loaders = list(train_loaders.values())
        backbone, uq_head, optim, *prep = accelerator.prepare(backbone, uq_head, optim, *all_loaders)
        train_loaders = {n: prep[i] for i, n in enumerate(N_SRC_LIST)}

    train_cycles   = {n: cycle(ldr) for n, ldr in train_loaders.items()}
    best_si_snri   = float("-inf")
    no_improve_evals = 0             # early stopping counter
    warmup_done    = 0
    step           = 0

    if full_eval and save_path is not None:
        for resume_name in [f"{experiment_name}.last.pt", f"{experiment_name}.pt"]:
            resume_path = save_path / resume_name
            if resume_path.exists():
                print(f"Auto-resuming from {resume_path}")
                ckpt = torch.load(str(resume_path), map_location="cpu", weights_only=False)
                raw_bb = backbone if accelerator is None else accelerator.unwrap_model(backbone)
                raw_uq = uq_head  if accelerator is None else accelerator.unwrap_model(uq_head)
                raw_bb.load_state_dict(ckpt.get("backbone", raw_bb.state_dict()))
                raw_uq.load_state_dict(ckpt.get("uq_head",  raw_uq.state_dict()))
                optim.load_state_dict(ckpt["optimizer"])
                try:
                    _warmup.load_state_dict(ckpt.get("scheduler", {}))
                except Exception:
                    pass
                if "plateau" in ckpt and _plateau is not None:
                    _plateau.load_state_dict(ckpt["plateau"])
                step           = ckpt.get("step", 0)
                best_si_snri   = ckpt.get("best_si_snri",   float("-inf"))
                warmup_done    = min(step, warmup_steps)
                print(f"  Resumed at step {step}, best SI-SNRi: {best_si_snri:.2f} dB")
                break

    pbar = tqdm(initial=step, total=steps, desc=f"[{experiment_name}]")

    while step < steps:
        backbone.train()
        uq_head.train()

        start = time.time()
        last_predicted = None
        last_vel_target = None
        running_loss = 0.0
        accum_done   = 0
        optim.zero_grad()

        while accum_done < grad_accum:
            n_src = random.choice(N_SRC_LIST)

            if accelerator is not None:
                accum_ctx = accelerator.accumulate(backbone, uq_head)
                cast_ctx  = accelerator.autocast()
            else:
                import contextlib
                accum_ctx = contextlib.nullcontext()
                cast_ctx  = torch.amp.autocast("cuda", dtype=torch.bfloat16)

            with accum_ctx:
                with cast_ctx:
                    condition_spec, flow_target, batch_mix_paths, batch_src_paths = next(train_cycles[n_src])
                    condition_spec = condition_spec.to(device)
                    flow_target = flow_target.to(device)

                    if flow_target.ndim == 4:
                        T_single = flow_target.shape[2]
                        flow_target_flat = rearrange(flow_target, "b s t d -> (b s) t d")
                        condition_spec_flat = torch.repeat_interleave(condition_spec, repeats=n_src, dim=0)
                    else:
                        T_single = condition_spec.shape[1] // n_src
                        flow_target_flat = flow_target
                        condition_spec_flat = condition_spec

                    if T_single > MAX_FRAMES_PER_SPK:
                        T_single = MAX_FRAMES_PER_SPK
                        
                    T_total = T_single * n_src
                    flow_target_flat = flow_target_flat[:, :T_total, :]
                    condition_spec_flat = condition_spec_flat[:, :T_total, :]

                    loss_mask = (flow_target_flat.abs().sum(dim=-1) > 1e-5)
                    norm_cond   = (condition_spec_flat - config.audio.norm_mean) / config.audio.norm_std
                    norm_target = (flow_target_flat    - config.audio.norm_mean) / config.audio.norm_std

                    B_orig = norm_target.shape[0]
                    times_raw = torch.rand(B_orig, dtype=condition_spec.dtype, device=device)
                    t_bcast   = rearrange(times_raw, "b -> b 1 1").expand_as(norm_target)
                    source_noise = torch.randn_like(norm_target)
                    noise_state  = ((1.0 - (1.0 - sigma) * t_bcast) * source_noise + t_bcast * norm_target)
                    vel_target = norm_target - (1.0 - sigma) * source_noise

                    # Entropy map: up-weight frames near speaker turn transitions
                    with torch.no_grad():
                        act = (flow_target_flat.abs().sum(dim=-1) > 1e-5).float()
                        diff = torch.zeros_like(act)
                        diff[:, 1:] = (act[:, 1:] - act[:, :-1]).abs()
                        entropy_map = F.max_pool1d(
                            diff.unsqueeze(1), kernel_size=15, stride=1, padding=7
                        ).squeeze(1)
                    loss_weight = loss_mask.float() * (1.0 + 1.0 * entropy_map)

                    # Backbone: entropy-weighted PIT MSE
                    predicted = backbone(audio=norm_cond, noise=noise_state, times=times_raw)
                    bb_loss = pit_mse_loss(predicted, vel_target, n_src, mask=loss_weight)

                    # UQ head: PIT NLL — detached backbone output keeps gradient paths clean
                    pred_log_sigma = uq_head(audio=norm_cond, noise=noise_state, times=times_raw)
                    pred_log_sigma = torch.clamp(pred_log_sigma, min=-2.0, max=2.0)
                    nll_loss = pit_nll_loss(predicted.detach(), pred_log_sigma, vel_target, n_src, mask=loss_weight)

                    loss = (bb_loss + 0.2 * nll_loss) / grad_accum

                    if accelerator is not None:
                        accelerator.backward(loss)
                    else:
                        loss.backward()
                    loss_val = loss.item() * grad_accum

                    # Capture stats from first micro-batch for logging
                    if last_predicted is None:
                        last_predicted = predicted.detach().float().cpu()
                        last_vel_target = vel_target.detach().float().cpu()

                sync = accelerator.sync_gradients if accelerator is not None else True
                if sync:
                    params_to_clip = list(backbone.parameters()) + list(uq_head.parameters())
                    if accelerator is not None:
                        accelerator.clip_grad_norm_(params_to_clip, clip_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(params_to_clip, clip_norm)
                    optim.step()
                    warmup_done += 1
                    if warmup_done <= warmup_steps:
                        _warmup.step()
                    optim.zero_grad()
                    running_loss += loss_val
                    accum_done   += 1

        end = time.time()
        step += 1
        pbar.update(1)
        cur_lr = optim.param_groups[0]["lr"]
        pbar.set_postfix(
            loss=f"{running_loss / grad_accum:.4f}",
            lr=f"{cur_lr:.2e}",
            step_s=f"{end - start:.2f}s",
        )

        if full_eval and accelerator is not None and accelerator.is_main_process:
            log_dict = {
                "train_loss": running_loss / grad_accum,
                "learning_rate": cur_lr,
            }
            if last_predicted is not None:
                log_dict.update({
                    "predicted/mean": last_predicted.mean().item(),
                    "predicted/max":  last_predicted.max().item(),
                    "predicted/min":  last_predicted.min().item(),
                    "target/mean":    last_vel_target.mean().item(),
                    "target/max":     last_vel_target.max().item(),
                    "target/min":     last_vel_target.min().item(),
                })
            accelerator.log(log_dict, step=step)

        if step % eval_every == 0:
            raw_bb   = backbone   if accelerator is None else accelerator.unwrap_model(backbone)
            raw_uq   = uq_head    if accelerator is None else accelerator.unwrap_model(uq_head)

            if full_eval and accelerator is not None:
                eval_stats = compute_eval_stats(raw_bb, val_loaders, device, sigma=sigma, n_per_nsrc=20)
                cur_loss   = eval_stats["eval_loss"]

                if accelerator.is_main_process:
                    accelerator.log(eval_stats, step=step)

                    si_snri = eval_mel_sisnri(raw_bb, val_loaders, device, n_eval=100)
                    pbar.write(f"Step {step}: eval_loss={cur_loss:.4f}  mel-SI-SNRi={si_snri:.2f} dB")
                    accelerator.log({"val/si_snri": si_snri}, step=step)

                    prev_best = best_si_snri
                    if si_snri > best_si_snri:
                        best_si_snri = si_snri
                        best_path = save_path / f"{experiment_name}.best.pt"
                        torch.save({
                            "backbone":        raw_bb.state_dict(),
                            "uq_head":         raw_uq.state_dict(),
                            "step":            step,
                            "best_si_snri":    float(best_si_snri),
                            "optimizer":       optim.state_dict(),
                            "scheduler":       _warmup.state_dict(),
                            "plateau":         _plateau.state_dict() if _plateau is not None else {},
                            "hp":              hp,
                        }, str(best_path), _use_new_zipfile_serialization=False)
                        accelerator.print(f"[BEST] SI-SNRi={best_si_snri:.2f} dB at step {step} -> {best_path}")

                    # RC3+RC4 fix: step plateau with actual SI-SNRi (Stage 1 only; _plateau is None for Stage 2)
                    if _plateau is not None:
                        _plateau.step(si_snri)

                    # Early stopping (full-training only)
                    if full_eval:
                        if si_snri >= prev_best + EARLY_STOP_MIN_DELTA_DB:
                            no_improve_evals = 0
                        else:
                            no_improve_evals += 1
                            accelerator.print(
                                f"[ES] No ≥{EARLY_STOP_MIN_DELTA_DB} dB gain: "
                                f"{no_improve_evals}/{EARLY_STOP_PATIENCE_EVALS}"
                            )
                            if no_improve_evals >= EARLY_STOP_PATIENCE_EVALS:
                                accelerator.print(
                                    f"[EARLY STOP] Triggered at step {step} — "
                                    f"no substantial gain for {EARLY_STOP_PATIENCE_EVALS} evals. "
                                    f"Best: {best_si_snri:.2f} dB. Use .best.pt checkpoint."
                                )
                                pbar.close()
                                return best_si_snri

                    last_path = save_path / f"{experiment_name}.last.pt"
                    torch.save({
                        "backbone":        raw_bb.state_dict(),
                        "uq_head":         raw_uq.state_dict(),
                        "step":            step,
                        "best_si_snri":    float(best_si_snri),
                        "optimizer":       optim.state_dict(),
                        "scheduler":       _warmup.state_dict(),
                        "plateau":         _plateau.state_dict() if _plateau is not None else {},
                        "hp":              hp,
                    }, str(last_path), _use_new_zipfile_serialization=False)
                    pbar.write(f"Step {step}: SI-SNRi={si_snri:.2f} dB  eval_loss={cur_loss:.4f}")
                    accelerator.print(f"[LAST] Step {step} -> {last_path}")

            else:
                si_snri = eval_mel_sisnri(raw_bb, val_loaders, device)
                print(f"  [step {step}/{steps}] SI-SNRi={si_snri:.2f} dB  loss={running_loss / grad_accum:.4f}  lr={optim.param_groups[0]['lr']:.2e}")
                if _plateau is not None:
                    _plateau.step(si_snri)

                if si_snri > best_si_snri:
                    best_si_snri = si_snri
                    if save_path is not None:
                        best_path = Path(save_path) / f"{experiment_name}.best.pt"
                        torch.save({
                            "backbone":     raw_bb.state_dict(),
                            "uq_head":      raw_uq.state_dict(),
                            "step":         step,
                            "best_si_snri": float(best_si_snri),
                            "optimizer":    optim.state_dict(),
                        }, str(best_path), _use_new_zipfile_serialization=False)
                        print(f"[BEST] mel-SI-SNRi={best_si_snri:.2f} dB -> {best_path}")

            if trial is not None:
                trial_val = si_snri  # mel-SI-SNRi
                trial.report(trial_val, step)
                if trial.should_prune():
                    pbar.close()
                    raise optuna.exceptions.TrialPruned()

            gc.collect()
            torch.cuda.empty_cache()

        torch.cuda.empty_cache()

    pbar.close()
    return best_si_snri


def search_stage1(train_loaders: dict, val_loaders: dict, device: torch.device) -> Dict[str, Any]:
    def objective(trial: optuna.Trial) -> float:
        hp = {
            "lr_max":       trial.suggest_float("lr_max",       1e-7, 1e-4, log=True),
            "lr_start":     trial.suggest_float("lr_start",     1e-9, 1e-6, log=True),
            "warmup_steps": trial.suggest_int  ("warmup_steps", 500,  5000),
            "clip_norm":    trial.suggest_float("clip_norm",    0.05, 2.0,  log=True),
            "sigma":        trial.suggest_float("sigma",        1e-6, 1e-3, log=True),
            "batch_size":   4,
            "grad_accum":   trial.suggest_categorical("grad_accum", [2, 4]),
        }
        trial_save = HP_SEARCH_PATH / f"stage1_trial{trial.number}"
        os.makedirs(trial_save, exist_ok=True)
        try:
            return run_training(
                hp=hp, steps=STAGE1_SEARCH_STEPS, eval_every=STAGE1_SEARCH_EVAL_EVERY,
                save_every=STAGE1_SEARCH_STEPS + 1, train_loaders=train_loaders, val_loaders=val_loaders, device=device,
                experiment_name=f"trial_{trial.number}", save_path=trial_save, trial=trial, full_eval=False,
            )
        except optuna.exceptions.TrialPruned:
            raise
        except Exception as e:
            print(f"[Trial {trial.number}] FAILED: {e}")
            return float("-inf")
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    study = optuna.create_study(
        study_name="stage1_hp_search_libri", direction="maximize", sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=STAGE1_SEARCH_EVAL_EVERY),
        storage=OPTUNA_DB, load_if_exists=True,
    )
    study.enqueue_trial({k: v for k, v in MANUAL_STAGE1_HP.items() if k != "batch_size"})
    remaining = max(0, STAGE1_N_TRIALS - len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]))
    if remaining > 0:
        study.optimize(objective, n_trials=remaining, show_progress_bar=True)
    best_trial = study.best_trial
    print(f"\n[Stage 1 Search] Best trial #{best_trial.number}: mel-SI-SNRi={best_trial.value:.2f} dB")
    best_hp = dict(best_trial.params)
    best_hp["batch_size"] = 4
    return best_hp


def main():
    try:
        torch.serialization.add_safe_globals([np.core.multiarray.scalar])
    except AttributeError:
        pass

    apply_training_patches()

    ddp_kwargs  = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(log_with="wandb", kwargs_handlers=[ddp_kwargs], mixed_precision=TRAIN_MIXED_PRECISION)
    device = accelerator.device

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    accelerator.print("Loading datasets...")
    train_loaders, val_loaders = load_datasets(batch_size=4)

    # # -- Stage 1 --
    if HP_SEARCH_MODE:
        accelerator.print("\n" + "=" * 60)
        accelerator.print("Stage 1 HP search")
        accelerator.print("=" * 60)
        best_s1_hp = search_stage1(train_loaders, val_loaders, device)
        accelerator.print(f"\nBest Stage 1 HPs: {best_s1_hp}")
    else:
        best_s1_hp = MANUAL_STAGE1_HP
        accelerator.print(f"\nUsing manual Stage 1 HPs: {best_s1_hp}")

    accelerator.print("\n" + "=" * 60)
    accelerator.print("Stage 1 full training")
    accelerator.print("=" * 60)
    accelerator.init_trackers(train_project, config={"stage": 1, **best_s1_hp})
    run_training(
        hp=best_s1_hp, steps=STAGE1_FULL_STEPS, eval_every=STAGE1_FULL_EVAL_EV, save_every=STAGE1_FULL_SAVE_EV,
        train_loaders=train_loaders, val_loaders=val_loaders, device=device,
        experiment_name=STAGE1_EXPERIMENT, save_path=STAGE1_SAVE_PATH, accelerator=accelerator, full_eval=True, pretrained_path=PRETRAINED_BACKBONE_PATH,
    )
    accelerator.end_training()
    accelerator.print("\nTraining complete!")


if __name__ == "__main__":
    main()