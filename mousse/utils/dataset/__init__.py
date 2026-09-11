from typing import Callable, Optional
from pathlib import Path
from .mussed import MussedDataset, _LOCAL_SCP_BASE

SETTINGS = {
    "mussed2": (MussedDataset, {"n_src": 2, "mode": "min"}),
    "mussed3": (MussedDataset, {"n_src": 3, "mode": "min"}),
    "mussed4": (MussedDataset, {"n_src": 4, "mode": "min"}),
}


def make_dataset(
    setting: str,
    cache_path: str,
    subset: str = "test",
    collate_fn: Optional[Callable] = None,
    scp_base_dir: str = _LOCAL_SCP_BASE,
):
    cls, kwargs = SETTINGS[setting]
    return cls(
        cache_path=cache_path,
        collate_fn=collate_fn,
        subset=subset,
        scp_base_dir=scp_base_dir,
        **kwargs,
    )
