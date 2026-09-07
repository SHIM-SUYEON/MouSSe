import os
import torch
import torchaudio
from torch.utils.data import Dataset
from pathlib import Path

import torch.nn.functional as F

from mousse.configs import config


_LOCAL_SCP_BASE = str(
    Path(__file__).resolve().parent / "scp"
)


class LibriTTSKORDataset(Dataset):
    def __init__(
        self,
        cache_path: str,
        n_src: int = 2,
        subset: str = "test",
        mode: str = "min",
        collate_fn=None,
        scp_base_dir: str = _LOCAL_SCP_BASE,
    ):
        self.n_src = n_src
        self.processor = collate_fn
        self.visual = False
        folder_name = f"{n_src}spk"

        subset_map = {"train": "tr", "dev": "cv", "test": "tt"}
        prefix = subset_map.get(subset, "tt")

        self.scp_dir = os.path.join(scp_base_dir, folder_name)

        mix_scp_path = os.path.join(self.scp_dir, f"{prefix}_mix_clean.scp")
        if not os.path.exists(mix_scp_path):
            mix_scp_path = os.path.join(self.scp_dir, "tr_mix_clean.scp")

        print(f"[DATA] Loading mixture manifest: {mix_scp_path}")
        if not os.path.exists(mix_scp_path):
            raise FileNotFoundError(f"Mixture manifest not found: {mix_scp_path}")

        with open(mix_scp_path, "r", encoding="utf-8") as f:
            self.mix_lines = [line.strip() for line in f if line.strip()]

        self.src_scp_lists = []
        for i in range(n_src):
            scp_path = os.path.join(self.scp_dir, f"{prefix}_s{i+1}.scp")
            if not os.path.exists(scp_path):
                scp_path = os.path.join(self.scp_dir, f"tr_s{i+1}.scp")
            if not os.path.exists(scp_path):
                raise FileNotFoundError(f"Source manifest not found: {scp_path}")
            with open(scp_path, "r", encoding="utf-8") as f:
                self.src_scp_lists.append([line.strip() for line in f if line.strip()])

        self.total_samples = len(self.mix_lines)
        for lst in self.src_scp_lists:
            self.total_samples = min(self.total_samples, len(lst))

        print(f"[DATA] {self.total_samples} samples (n_src={n_src}, subset={subset})")

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        mix_path = self.mix_lines[idx]
        wav, sr = torchaudio.load(mix_path)
        if sr != config.audio.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, config.audio.sample_rate)
        wav = wav.mean(0, keepdim=True)

        targets = []
        for src_idx in range(self.n_src):
            src_path = self.src_scp_lists[src_idx][idx]
            src_wav, src_sr = torchaudio.load(src_path)
            if src_sr != config.audio.sample_rate:
                src_wav = torchaudio.functional.resample(
                    src_wav, src_sr, config.audio.sample_rate
                )
            if src_wav.shape[0] > 1:
                src_wav = src_wav.mean(0, keepdim=True)
            targets.append(src_wav)

        mix_len = wav.shape[-1]
        synced = []
        for t in targets:
            if t.shape[-1] < mix_len:
                t = F.pad(t, (0, mix_len - t.shape[-1]))
            else:
                t = t[..., :mix_len]
            synced.append(t)

        return {
            "wav": wav,
            "targets": torch.cat(synced, dim=0),
            "n_src": self.n_src,
            "mix_path": mix_path,
            "src_paths": [self.src_scp_lists[s][idx] for s in range(self.n_src)],
        }

    def _do_spectogram(self, src):
        from mousse.utils.audio import spectogram
        return spectogram(
            src,
            config.audio.n_fft,
            config.audio.n_mels,
            config.audio.hop_size,
            config.audio.win_size,
            config.audio.mel_norm,
            config.audio.mel_scale,
            config.audio.sample_rate,
        )

    def collate(self, batch):
        batch = [b for b in batch if b is not None]
        if not batch:
            return None

        max_frames = 0
        mix_list, tgt_list = [], []

        for item in batch:
            mix_wav = item["wav"]        # [1, T]
            tgt_wavs = item["targets"]   # [n_src, T]

            mix_spec = self._do_spectogram(mix_wav).squeeze(0).transpose(-1, -2)  # [F, n_mels]

            tgt_specs = []
            for s in range(self.n_src):
                t_spec = self._do_spectogram(
                    tgt_wavs[s].unsqueeze(0)
                ).squeeze(0).transpose(-1, -2)  # [F, n_mels]
                tgt_specs.append(t_spec)

            repeated_mix = mix_spec.repeat(self.n_src, 1)          # [n_src*F, n_mels]
            concat_tgt = torch.cat(tgt_specs, dim=0)               # [n_src*F, n_mels]

            max_frames = max(max_frames, repeated_mix.shape[0])
            mix_list.append(repeated_mix)
            tgt_list.append(concat_tgt)

        padded_mix, padded_tgt = [], []
        for m, t in zip(mix_list, tgt_list):
            pad = max_frames - m.shape[0]
            if pad > 0:
                m = F.pad(m, (0, 0, 0, pad))
                t = F.pad(t, (0, 0, 0, pad))
            padded_mix.append(m)
            padded_tgt.append(t)

        mix_tensor = torch.stack(padded_mix, dim=0)     # [B, n_src*F, n_mels]
        tgt_tensor = torch.stack(padded_tgt, dim=0)     # [B, n_src*F, n_mels]

        mix_paths = [item.get("mix_path", "") for item in batch]
        src_paths = [item.get("src_paths", []) for item in batch]

        return mix_tensor, tgt_tensor, mix_paths, src_paths
