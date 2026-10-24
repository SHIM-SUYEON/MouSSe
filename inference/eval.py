import os
import sys
import gc
import json
import math
import argparse
import random
import warnings
from pathlib import Path

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
from mousse.models.patches import apply_forward_patches
from mousse.utils.vocoder import get_vocoder, do_vocoder
from mousse.utils.signal import rms_normalize
from mousse.utils.reporting import (
    LANG_CATS, GENDER_CATS, OVERLAP_CATS,
    _make_acc, _acc_add, _avg,
    _load_meta, _categorize, _finalize_results,
)

random.seed(42)

DEFAULT_CKPT = Path(
    "./mousse/mousse-checkpoint.pt"
)

N_SRC_LIST = [2, 3, 4]
TEST_CAP = 1300
EVAL_STEPS = 16
MAX_FRAMES_PER_SPK = 1500
PESQ_SR = 16000

# Long sequences are truncated at MAX_FRAMES_PER_SPK; no duration-based skip.
MAX_MIX_DURATION = float("inf")

# Threshold for detecting active speech frames in the target mel spectrogram.
ACTIVE_THRESH = 1e-4

DEFAULT_OUT_DIR = Path(
    "./results"
)

os.makedirs(DEFAULT_OUT_DIR, exist_ok=True)

def fast_euler_sample(predictor, uq_head, audio, steps=16, alpha=1.5,
                      lambda_max=2.0, use_uq_guidance=True, condition=None,
                      lambda_scale=None):
    """ODE Euler sampler.

    A lambda scale enables guidance based only on conditional uncertainty.
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

                if use_uq_guidance and lambda_scale is not None:
                    v_uncond = predictor(audio=audio_empty, noise=z, times=t_batch)
                    log_s_cond = uq_head(audio=audio, noise=z, times=t_batch)
                    log_s_uncond = uq_head(audio=audio_empty, noise=z, times=t_batch)
                    sigma_cond = torch.exp(log_s_cond)
                    sigma_uncond = torch.exp(log_s_uncond)
                    sc = sigma_cond.float()
                    su = sigma_uncond.float()
                    sigma_cond_sum = sc if sigma_cond_sum is None else sigma_cond_sum + sc
                    sigma_uncond_sum = su if sigma_uncond_sum is None else sigma_uncond_sum + su
                    lambda_star = torch.clamp(
                        sigma_cond.mean(dim=-1, keepdim=True) * lambda_scale,
                        max=lambda_max,
                    )
                    if i == 0:
                        print(f"lambda_star step0  mean={lambda_star.mean():.3f}  max={lambda_star.max():.3f}  frac_clamped={(lambda_star >= lambda_max).float().mean():.3f}", flush=True)
                    v_guided = v_cond + (v_cond - v_uncond) * lambda_star
                elif use_uq_guidance:
                    # Adaptive guidance weight
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
                    if i == 0:
                        print(f"lambda_star step0  mean={lambda_star.mean():.3f}  max={lambda_star.max():.3f}  frac_clamped={( lambda_star >= lambda_max).float().mean():.3f}", flush=True)
                    v_guided = v_cond + (v_cond - v_uncond) * lambda_star
                else:
                    v_uncond = predictor(audio=audio_empty, noise=z, times=t_batch)
                    v_guided = v_cond + (v_cond - v_uncond) * alpha
                z = z + v_guided.float() * dt
    sigma_cond_mean = sigma_cond_sum / steps if sigma_cond_sum is not None else None
    sigma_uncond_mean = sigma_uncond_sum / steps if sigma_uncond_sum is not None else None
    return z, sigma_cond_mean, sigma_uncond_mean


_VOCODER_CACHE = None

def _snr_val(preds: torch.Tensor, target: torch.Tensor) -> float:
    signal = float(target.pow(2).mean())
    noise = float((preds - target).pow(2).mean())
    if noise < 1e-8:
        return 40.0
    return 10.0 * math.log10(max(signal, 1e-8) / noise)

def _eval_loop(backbone, uq_head, test_loaders, device,
               out_dir=None, save_wav=False, ablation='full', eval_steps=None,
               lambda_scale=None):
    """Run evaluation and return metric accumulators."""

    _ABLATION_VALID = ("full", "no_uq", "no_entropy")
    if ablation not in _ABLATION_VALID:
        raise ValueError(f"Unknown ablation mode '{ablation}'. Choose from {_ABLATION_VALID}")
    use_uq_guidance = (ablation != "no_uq")
    print(f"\n{'='*70}")
    print(f"  ABLATION MODE : {ablation.upper()}")
    if use_uq_guidance and lambda_scale is not None:
        print(f"  UQ guidance   : YES — λ*=σ_cond×{lambda_scale} (diet/UQ-only formula, max={2.0})")
    else:
        print(f"  UQ guidance   : {'YES — λ* adaptive (num/den)' if use_uq_guidance else 'NO  — fixed α=1.5'}")
    print(f"  Entropy loss  : {'separate no-entropy checkpoint' if ablation == 'no_entropy' else 'standard training'}")
    print(f"{'='*70}\n")

    backbone.eval()
    uq_head.eval()

    sr = config.audio.sample_rate
    estoi_m = ShortTimeObjectiveIntelligibility(fs=sr, extended=True).to(device)
    sisnr_m = ScaleInvariantSignalNoiseRatio().to(device)
    sisdr_m = ScaleInvariantSignalDistortionRatio().to(device)
    sdr_m = SignalDistortionRatio(filter_length=1, zero_mean=False).to(device)

    vocoder = get_vocoder(device)
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)

    # Metric accumulators
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

            mix_input = norm_cond.to(dtype=torch.bfloat16)
            sep_spec, sigma_cond_mean, sigma_uncond_mean = fast_euler_sample(
                backbone, uq_head, mix_input,
                steps=(eval_steps or EVAL_STEPS), alpha=1.5, lambda_max=2.0,
                use_uq_guidance=use_uq_guidance,
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

            # Mixture waveform for SI-SDRi and output
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
            if lang and lang in LANG_CATS: cat_acc[n_src]["lang"][lang]["n_samples"]       += 1
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
                if sigma_cond_mean is not None:
                    slot_sigma = sigma_cond_mean[0, pi * T_per_spk:(pi + 1) * T_per_spk, :]
                    uq_val = round(float(slot_sigma.mean().item()), 6)
                    _acc_add(nsrc_acc[n_src], uq=uq_val)
                    if lang    and lang    in LANG_CATS: _acc_add(cat_acc[n_src]["lang"][lang], uq=uq_val)
                    if gender  and gender  in GENDER_CATS: _acc_add(cat_acc[n_src]["gender"][gender], uq=uq_val)
                    if overlap and overlap in OVERLAP_CATS: _acc_add(cat_acc[n_src]["overlap"][overlap], uq=uq_val)
                spk_rec = {"sisdri": None, "sdri": None, "estoii": None, "uq_sigma_mean": uq_val}

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

                    # SI-SDRi and SDRi against the mixture baseline
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
                            sisdri_val = sisdr_sep - sisdr_mix
                            if not (math.isnan(sisdri_val) or math.isinf(sisdri_val)):
                                spk_rec["sisdri"] = round(sisdri_val, 4)
                                _acc_add(nsrc_acc[n_src], sisdri=sisdri_val)
                                if lang and lang in LANG_CATS: _acc_add(cat_acc[n_src]["lang"][lang], sisdri=sisdri_val)
                                if gender and gender in GENDER_CATS: _acc_add(cat_acc[n_src]["gender"][gender], sisdri=sisdri_val)
                                if overlap and overlap in OVERLAP_CATS: _acc_add(cat_acc[n_src]["overlap"][overlap], sisdri=sisdri_val)

                            sdr_sep = sdr_m(wav_p_n[:min_len], wav_t_n[:min_len]).item()
                            sdr_mix = sdr_m(wav_mix_n[:min_len], wav_t_n[:min_len]).item()
                            sdri_val = sdr_sep - sdr_mix
                            if not (math.isnan(sdri_val) or math.isinf(sdri_val)):
                                spk_rec["sdri"] = round(sdri_val, 4)
                                _acc_add(nsrc_acc[n_src], sdri=sdri_val)
                                if lang and lang in LANG_CATS: _acc_add(cat_acc[n_src]["lang"][lang], sdri=sdri_val)
                                if gender and gender in GENDER_CATS: _acc_add(cat_acc[n_src]["gender"][gender], sdri=sdri_val)
                                if overlap and overlap in OVERLAP_CATS: _acc_add(cat_acc[n_src]["overlap"][overlap], sdri=sdri_val)
                        except Exception as e:
                            print(f"  [SI-SDRi/SDRi] error spk={pi}: {e}")

                    # ESTOI improvement over the mixture baseline
                    if has_speech and wav_p_n.shape[-1] >= int(sr * 0.5):
                        try:
                            estoi_sep_val = estoi_m(wav_p_n, wav_t_n).item()
                            if not (math.isnan(estoi_sep_val) or math.isinf(estoi_sep_val)):
                                if wav_mix is not None:
                                    tgt_w_len = wav_t_n.shape[-1]
                                    mix_for_stoi = wav_mix[:tgt_w_len]
                                    if mix_for_stoi.shape[-1] < tgt_w_len:
                                        mix_for_stoi = F.pad(mix_for_stoi, (0, tgt_w_len - mix_for_stoi.shape[-1]))
                                    mix_stoi_n = rms_normalize(mix_for_stoi)
                                    if mix_stoi_n.shape[-1] >= int(sr * 0.5):
                                        estoi_mix_val = estoi_m(mix_stoi_n, wav_t_n).item()
                                        estoii_val = estoi_sep_val - estoi_mix_val
                                        if not (math.isnan(estoii_val) or math.isinf(estoii_val)):
                                            spk_rec["estoii"] = round(estoii_val, 4)
                                            _acc_add(nsrc_acc[n_src], estoii=estoii_val)
                                            if lang and lang in LANG_CATS: _acc_add(cat_acc[n_src]["lang"][lang], estoii=estoii_val)
                                            if gender and gender in GENDER_CATS: _acc_add(cat_acc[n_src]["gender"][gender], estoii=estoii_val)
                                            if overlap and overlap in OVERLAP_CATS: _acc_add(cat_acc[n_src]["overlap"][overlap], estoii=estoii_val)
                        except Exception as e:
                            print(f"  [ESTOIi] error spk={pi}: {e}")

                record[spk_key] = spk_rec

            per_sample_records.append(record)

        # Speaker count summary
        acc = nsrc_acc[n_src]
        print(
            f"[n_src={n_src}]  "
            f"SDRi={_avg(acc,'sdri'):.2f} dB (n={acc['n_sdri']})  "
            f"SI-SDRi={_avg(acc,'sisdri'):.2f} dB (n={acc['n_sisdri']})  "
            f"ESTOIi={_avg(acc,'estoii'):.4f} (n={acc['n_estoii']})  "
            f"UQ={_avg(acc,'uq'):.6f} (n={acc['n_uq']})  "
            f"[dur_skipped(>{MAX_MIX_DURATION}s): {dur_skip_cnt}]"
        )

    del estoi_m, sisnr_m, sisdr_m, sdr_m
    gc.collect()
    torch.cuda.empty_cache()

    return nsrc_acc, cat_acc, per_sample_records


# Checkpoint loading 
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

    return backbone, uq_head


# Result saving 
def _save_results(results, out_dir, ckpt_path, args):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_sample = results.pop("per_sample", [])
    # Metrics kept per speaker slot.
    KEEP_SPK = ("sisdri", "sdri", "estoii", "uq_sigma_mean")
    per_sample_trim = []
    for rec in per_sample:
        out = {k: rec.get(k) for k in ("stem", "n_src", "lang", "gender", "overlap")}
        for s in range(1, rec["n_src"] + 1):
            spk_key = f"s{s}"
            if spk_key in rec:
                out[spk_key] = {k: rec[spk_key].get(k) for k in KEEP_SPK}
        per_sample_trim.append(out)
    per_sample_path = out_dir / "per_sample_metrics.json"
    with open(per_sample_path, "w") as f:
        json.dump(per_sample_trim, f, indent=2)
    print(f"[EVAL] Per-sample metrics saved to {per_sample_path}  ({len(per_sample_trim)} records)")

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
                "sisdri": sd.get("sisdri"),
                "sdri": sd.get("sdri"),
                "estoii": sd.get("estoii"),
                "uq_sigma_mean": sd.get("uq_sigma_mean"),
            })
    uq_json_path = out_dir / "per-sample-uncertainty.json"
    with open(uq_json_path, "w") as f:
        json.dump(uq_json_records, f, indent=2)
    print(f"[EVAL] Per-sample uncertainty JSON saved to {uq_json_path}  ({len(uq_json_records)} records)")

    results_path = out_dir / "test_metrics_analyze.json"
    results["checkpoint"] = str(ckpt_path)
    results["ablation"] = args.ablation
    results["cap"] = args.cap
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[EVAL] Aggregate results saved to {results_path}")


def _build_test_loaders(args):
    """Build test loaders for each speaker count."""
    test_loaders = {}
    for n in N_SRC_LIST:
        ds = make_dataset(setting=f"mussed{n}", cache_path="/tmp", subset="test")
        n_total = len(ds)
        n_use = min(args.cap, n_total)
        indices = list(range(n_use))[:TEST_CAP + 1]
        ds_use = Subset(ds, indices)
        test_loaders[n] = DataLoader(
            ds_use, batch_size=1, shuffle=False,
            num_workers=2, collate_fn=ds.collate,
        )
        print(f"  n_src={n}: {len(indices)} samples / {n_total} total")
    return test_loaders


def _single_gpu_eval(ckpt_path, args, gpu_id, out_dir):
    """Run evaluation on one GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[EVAL] Device: {device}  GPU={gpu_id}")

    apply_forward_patches()
    print(f"[EVAL] Loading {ckpt_path}  ablation={args.ablation}")
    backbone, uq_head = load_checkpoint(ckpt_path, device)

    print(f"\n[EVAL] Building test loaders (cap={args.cap} per n_src)...")
    test_loaders = _build_test_loaders(args)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[EVAL] Output directory: {out_dir}")

    return _eval_loop(
        backbone, uq_head, test_loaders, device,
        out_dir=str(out_dir), save_wav=args.save_wav,
        ablation=args.ablation, eval_steps=getattr(args, 'steps', None),
        lambda_scale=getattr(args, 'lambda_scale', None),
    )


def _eval_one_ckpt(ckpt_path, args, gpu_id, out_dir):
    """Evaluate one checkpoint on the selected GPU."""
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    out_dir = Path(out_dir)
    nsrc_acc, cat_acc, per_sample = _single_gpu_eval(
        ckpt_path, args, gpu_id, out_dir,
    )
    results = _finalize_results(nsrc_acc, cat_acc, per_sample, N_SRC_LIST)
    _save_results(results, out_dir, ckpt_path, args)

# Entry point 
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the MUSSE separation model on the MUSSED test set."
    )
    parser.add_argument("--ckpt", default=str(DEFAULT_CKPT),
                        help="Path to .pt checkpoint, OR a directory to evaluate all .pt files inside")
    parser.add_argument("--gpu", default=0, type=int,
                        help="GPU id")
    parser.add_argument("--ablation", default="full",
                        choices=["full", "no_uq", "no_entropy"],
                        help=(
                            "Ablation mode:\n"
                            "  full          — UQ guidance (λ* adaptive)          [full model]\n"
                            "  no_uq         — no UQ guidance (fixed α=1.5)        [same ckpt]\n"
                            "  no_entropy    — UQ guidance, separate no-entropy    [no-entropy ckpt]\n"
                        ))
    parser.add_argument("--cap", default=TEST_CAP, type=int,
                        help="Max test samples per n_src")
    parser.add_argument("--out", default=None,
                        help="Output directory (default: DEFAULT_OUT_DIR)")
    parser.add_argument("--save-wav", action="store_true",
                        help="Save separated and mixture wav files under out_dir/wav/")
    parser.add_argument("--steps", type=int, default=None,
                        help=f"ODE Euler steps (default: EVAL_STEPS={EVAL_STEPS})")
    parser.add_argument("--lambda-scale", type=float, default=None, dest="lambda_scale",
                        help=(
                            "Enable diet/UQ-only guidance formula: λ*=clamp(σ_cond.mean×scale, max=2.0). "
                            "Typical value: 5.0  (matches training default lambda_scale=5.0)"
                        ))
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)

    # Directory mode: evaluate every .pt file found inside
    if ckpt_path.is_dir():
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
            _eval_one_ckpt(cf, args, args.gpu, out_sub)
        return

    out_dir = Path(args.out) if args.out else DEFAULT_OUT_DIR
    _eval_one_ckpt(ckpt_path, args, args.gpu, out_dir)

if __name__ == "__main__":
    main()
