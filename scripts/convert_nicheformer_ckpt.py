"""One-time conversion of the Nicheformer PyTorch-Lightning checkpoint to
pure-PyTorch format suitable for the spatialFMs inference adapter.

Requirements for this one-time conversion (not needed at runtime):
    pip install torch pytorch-lightning

Usage:
    python scripts/convert_nicheformer_ckpt.py \\
        --input-dir downloaded_nicheformer \\
        --output-dir model_weights/nicheformer

Reads:
    {input-dir}/nicheformer.ckpt
    {input-dir}/data/model_means/*.npy
    {input-dir}/data/model_means/model.h5ad

Writes:
    {output-dir}/model_state_dict.pt
    {output-dir}/hparams.json
    {output-dir}/model.h5ad
    {output-dir}/*_mean_script.npy
"""

import argparse
import json
import shutil
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = PROJECT_ROOT / "references" / "nicheformer"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "model_weights" / "nicheformer"


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing nicheformer.ckpt and data/model_means/ "
             f"(default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to write converted weights and metadata "
             f"(default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help="Override path to the Lightning checkpoint "
             "(default: {input-dir}/nicheformer.ckpt)",
    )
    parser.add_argument(
        "--means-dir",
        type=Path,
        default=None,
        help="Override path to the model_means directory "
             "(default: {input-dir}/data/model_means)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    ckpt_path = args.ckpt if args.ckpt is not None else args.input_dir / "nicheformer.ckpt"
    means_dir = args.means_dir if args.means_dir is not None else args.input_dir / "data" / "model_means"
    out_dir = args.output_dir

    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"
    assert means_dir.exists(), f"Model means directory not found: {means_dir}"

    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load Lightning checkpoint ---
    print(f"Loading checkpoint from {ckpt_path} ...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # --- Extract and save hyperparameters ---
    hparams = ckpt["hyper_parameters"]
    hparams_path = out_dir / "hparams.json"
    with open(hparams_path, "w") as f:
        json.dump(hparams, f, indent=2, default=str)
    print(f"Saved hyperparameters to {hparams_path}")
    print(f"  dim_model={hparams['dim_model']}, nlayers={hparams['nlayers']}, "
          f"nheads={hparams['nheads']}, n_tokens={hparams['n_tokens']}, "
          f"context_length={hparams['context_length']}")

    # --- Extract and save state dict ---
    state_dict = ckpt["state_dict"]
    sd_path = out_dir / "model_state_dict.pt"
    torch.save(state_dict, sd_path)
    print(f"Saved state dict ({len(state_dict)} keys) to {sd_path}")

    # --- Copy technology mean files ---
    for npy_file in sorted(means_dir.glob("*_mean_script.npy")):
        dst = out_dir / npy_file.name
        shutil.copy2(npy_file, dst)
        print(f"Copied {npy_file.name}")

    # --- Copy gene vocabulary (model.h5ad) ---
    model_h5ad = means_dir / "model.h5ad"
    if model_h5ad.exists():
        shutil.copy2(model_h5ad, out_dir / "model.h5ad")
        print("Copied model.h5ad (gene vocabulary)")
    else:
        print("WARNING: model.h5ad not found — gene alignment will not work")

    print(f"\nConversion complete. Output directory: {out_dir}")
    print("Contents:")
    for p in sorted(out_dir.iterdir()):
        size_mb = p.stat().st_size / (1024 * 1024)
        print(f"  {p.name:40s} {size_mb:8.1f} MB")


if __name__ == "__main__":
    main()
