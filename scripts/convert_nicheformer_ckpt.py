"""One-time conversion of the Nicheformer PyTorch-Lightning checkpoint to
pure-PyTorch format suitable for the spatialFMs inference adapter.

Requirements for this one-time conversion (not needed at runtime):
    pip install torch pytorch-lightning
The spatialFMs virtualenv already has torch; only pytorch-lightning may need
a temporary install:
    pip install pytorch-lightning && pip uninstall pytorch-lightning -y

Usage:
    python scripts/convert_nicheformer_ckpt.py

Reads:
    references/nicheformer/nicheformer.ckpt
    references/nicheformer/data/model_means/*.npy
    references/nicheformer/data/model_means/model.h5ad

Writes:
    model_weights/nicheformer/model_state_dict.pt
    model_weights/nicheformer/hparams.json
    model_weights/nicheformer/model.h5ad
    model_weights/nicheformer/*_mean_script.npy
"""

import json
import shutil
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CKPT_PATH = PROJECT_ROOT / "references" / "nicheformer" / "nicheformer.ckpt"
MEANS_DIR = PROJECT_ROOT / "references" / "nicheformer" / "data" / "model_means"
OUT_DIR = PROJECT_ROOT / "model_weights" / "nicheformer"


def main():
    assert CKPT_PATH.exists(), f"Checkpoint not found: {CKPT_PATH}"
    assert MEANS_DIR.exists(), f"Model means directory not found: {MEANS_DIR}"

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load Lightning checkpoint ---
    print(f"Loading checkpoint from {CKPT_PATH} ...")
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)

    # --- Extract and save hyperparameters ---
    hparams = ckpt["hyper_parameters"]
    hparams_path = OUT_DIR / "hparams.json"
    with open(hparams_path, "w") as f:
        json.dump(hparams, f, indent=2, default=str)
    print(f"Saved hyperparameters to {hparams_path}")
    print(f"  dim_model={hparams['dim_model']}, nlayers={hparams['nlayers']}, "
          f"nheads={hparams['nheads']}, n_tokens={hparams['n_tokens']}, "
          f"context_length={hparams['context_length']}")

    # --- Extract and save state dict ---
    state_dict = ckpt["state_dict"]
    sd_path = OUT_DIR / "model_state_dict.pt"
    torch.save(state_dict, sd_path)
    print(f"Saved state dict ({len(state_dict)} keys) to {sd_path}")

    # --- Copy technology mean files ---
    for npy_file in sorted(MEANS_DIR.glob("*_mean_script.npy")):
        dst = OUT_DIR / npy_file.name
        shutil.copy2(npy_file, dst)
        print(f"Copied {npy_file.name}")

    # --- Copy gene vocabulary (model.h5ad) ---
    model_h5ad = MEANS_DIR / "model.h5ad"
    if model_h5ad.exists():
        shutil.copy2(model_h5ad, OUT_DIR / "model.h5ad")
        print("Copied model.h5ad (gene vocabulary)")
    else:
        print("WARNING: model.h5ad not found — gene alignment will not work")

    print(f"\nConversion complete. Output directory: {OUT_DIR}")
    print("Contents:")
    for p in sorted(OUT_DIR.iterdir()):
        size_mb = p.stat().st_size / (1024 * 1024)
        print(f"  {p.name:40s} {size_mb:8.1f} MB")


if __name__ == "__main__":
    main()
