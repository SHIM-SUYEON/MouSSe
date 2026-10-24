# MouSSe

**Mo**del of **U**ncertainty for **S**peech **Se**paration — a single-stage,
flow-matching speech separator with a per-frame uncertainty (UQ) head and
adaptive UQ-guided classifier-free guidance.

This repository contains the code needed to **train** and **evaluate** the
model reported in the paper.

## Examples

Each panel shows the input **mixture** (top) and, per source, the
**ground-truth** (left) vs. the model's **separated** mel-spectrogram (right),
across increasing speaker counts and overlap ratios.

| 2 speakers · 25% overlap | 3 speakers · 50% overlap | 4 speakers · 75% overlap |
|:---:|:---:|:---:|
| ![2-speaker separation example](assets/demos/2spk_ov25.png) | ![3-speaker separation example](assets/demos/3spk_ov50.png) | ![4-speaker separation example](assets/demos/4spk_ov75.png) |

## Repository layout

```
MOUSSE/
├── mousse/
│   ├── configs/
│   │   └── config.py
│   ├── models/
│   │   ├── audio_flow.py
│   │   ├── transformer.py
│   │   └── tensors.py
│   └── utils/
│       ├── audio.py
│       ├── misc.py
│       └── dataset/
│           ├── mussed.py
│           ├── generate_mussed.py
│           └── scp/
├── training/
│   └── train.py
├── inference/
│   ├── eval.py
│   └── eval.sh
├── requirements.txt
└── LICENSE
```

All scripts are meant to be **run from the repository root** so that the
top-level `mousse` package is importable.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Versions in `requirements.txt` are pinned to those used for the paper. The
waveform-domain metrics load an external vocoder at runtime via
`torch.hub.load("ex3ndr/supervoice-vocoder", "bigvsan")`, so network access is
required the first time evaluation runs.

## Data

Training and evaluation read Kaldi-style `.scp` manifests, one line per audio
file. The test-set manifests (1300 mixtures per speaker count) ship in
[mousse/utils/dataset/scp/](mousse/utils/dataset/scp/):

```
scp/<N>spk/
  tt_mix_clean.scp   # mixtures; tt test, cv dev, tr train
  tt_s1.scp ...      # per-source targets
```

The paths inside the manifests point to the private MUSSED mixture set
(`/data/...`). Reproduction on another machine requires regenerated mixtures
and updated paths in the `.scp` files (or an `scp_base_dir` setting/symlink that
resolves those locations). Speaker counts of 2, 3 and 4 are selected via the
dataset key `mussed2` / `mussed3` / `mussed4`.

### Regenerating the mixtures

MUSSED is built from LibriTTS-R (English) and an internal Korean multi-speaker
corpus by [mousse/utils/dataset/generate_mussed.py](mousse/utils/dataset/generate_mussed.py).
The `Dataset paths` block at the top of the script defines the source and
output locations. Mixture generation runs from the repository root:

```bash
python mousse/utils/dataset/generate_mussed.py
```

It writes `<N>spk/{raw,json}/` — the mixture `.wav` files plus a per-mixture
metadata `.json` (speakers, overlap ratio, timings). Regenerate the `.scp`
manifests from those outputs (or update the paths in the shipped `.scp` files).
Requires `pydub` + `ffmpeg` (see `requirements.txt`).

## Configuration

All audio and model hyper-parameters live in
[mousse/configs/config.py](mousse/configs/config.py) and are the single source
of truth for both training and evaluation:

| field | value |
|-------|-------|
| sample_rate | 16000 |
| n_mels / n_fft / hop / win | 100 / 1024 / 256 / 1024 |
| mel_norm / mel_scale | slaney / slaney |
| norm_mean / norm_std | -5.6395 / 5.0133 |
| transformer | 24 layers, 16 heads, dim 1024, ffn 4096 |

Evaluation seeds Python's RNG with `random.seed(42)`.

## Training

```bash
python training/train.py
```

> Note: `training/train.py` pins `CUDA_VISIBLE_DEVICES = "0"` at the top of the
> file (as used for the paper run). Change this value when selecting another GPU.
> Runs are logged to Weights & Biases under the project `mousse`.

## Evaluation

Set the checkpoint path before running the driver from the repository root.
The driver uses one GPU and runs each evaluation sequentially. The `GPU`
setting in `inference/eval.sh` selects the device.

```bash
bash inference/eval.sh main       # backbone + adaptive
bash inference/eval.sh adaptive   # adaptive lambda-scale sweep
bash inference/eval.sh summary    # print summary tables from existing logs
bash inference/eval.sh all        # everything
```

Or call the evaluator directly:

```bash
python inference/eval.py \
    --ckpt /path/to/mousse-checkpoint.pt \
    --gpu 0 --cap 1300 \
    --out results/backbone_only
```

Key flags: `--ablation {full,no_uq,no_entropy}`, `--lambda-scale`, `--steps`,
`--cap`, `--gpu`, `--out`, `--save-wav`. See the module docstring in
[inference/eval.py](inference/eval.py) for details.
