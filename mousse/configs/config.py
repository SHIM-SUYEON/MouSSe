from mousse.utils.misc import dict_to_object

config = dict_to_object({

    # Shared audio parameters
    "audio": {
        "sample_rate": 16000, #24000,
        "n_mels": 100,
        "n_fft": 1024,
        "hop_size": 256,
        "win_size": 256 * 4,
        "mel_norm": "slaney",
        "mel_scale": "slaney",
        # RC1 fix: norm_mean/norm_std were swapped (norm_stat.py: norm_mean=-5.6395, norm_std=5.0133)
        # was: "norm_std": -5.6395, #2.2615,
        # was: "norm_mean": 5.0133 #-5.8843
        "norm_mean": -5.6395,
        "norm_std": 5.0133
    },

    # Audio predictor
    "model": {
        "n_embeddings": 1024,
        "n_heads": 16,
        "n_layers": 24,
        "n_dim": 1024,
        "n_dim_head": 64,
        "n_dim_ffn": 4096,
    }
})