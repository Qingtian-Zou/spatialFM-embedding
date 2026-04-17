"""Loki (OmiCLIP) adapter for the unified embedding framework.

Produces L2-normalized 768-dim embeddings via two paths sharing one model load:
- text:  top-50 expressed gene names per spot, encoded by OmiCLIP text tower
- image: H&E patches around each spot, encoded by OmiCLIP image tower

The image path is auto-enabled when spatial info is available, either embedded
in the AnnData (`obsm['spatial']` + `uns['spatial'][lib]['images']['hires']` +
`tissue_hires_scalef`) or supplied via ``spatial_dir`` pointing at a standard
Visium ``spatial/`` folder.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from anndata import AnnData
from PIL import Image

from src.models.loki.preprocess import generate_gene_df, segment_patches
from src.models.loki.utils import encode_images, encode_texts, load_model

# PIL refuses very large H&E images by default; lift the cap.
Image.MAX_IMAGE_PIXELS = None


def _attach_visium_spatial(
    adata: AnnData,
    spatial_dir: str,
    library_id: str = "loki",
) -> None:
    """Load a standard Visium ``spatial/`` folder and attach to ``adata`` in place.

    Populates ``adata.obsm['spatial']`` with full-resolution pixel coords and
    ``adata.uns['spatial'][library_id]`` with the hires image and scalefactors.
    Intersects on barcode; warns about dropouts.
    """
    spatial_dir = Path(spatial_dir)
    scalef = json.loads((spatial_dir / "scalefactors_json.json").read_text())
    hires = np.asarray(Image.open(spatial_dir / "tissue_hires_image.png"))

    pos = pd.read_csv(spatial_dir / "tissue_positions.csv")
    pos = pos.set_index("barcode")

    common = adata.obs_names.intersection(pos.index)
    missing = len(adata.obs_names) - len(common)
    if missing:
        print(f"[loki] {missing} barcodes in h5ad lack spatial coords; dropping them.")
    adata._inplace_subset_obs(common)
    pos = pos.loc[adata.obs_names]

    adata.obsm["spatial"] = pos[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy()
    adata.uns["spatial"] = {
        library_id: {
            "images": {"hires": hires},
            "scalefactors": scalef,
        }
    }


def _resolve_spatial(
    adata: AnnData,
    spatial_dir: Optional[str],
    library_id: Optional[str],
):
    """Return (img_array, coord_df, library_id) or None if image path unavailable."""
    if "spatial" not in adata.obsm or "spatial" not in adata.uns:
        if spatial_dir is None:
            return None
        _attach_visium_spatial(adata, spatial_dir, library_id=library_id or "loki")

    lib = library_id or next(iter(adata.uns["spatial"]))
    entry = adata.uns["spatial"][lib]
    img = np.asarray(entry["images"]["hires"])
    scalef = entry["scalefactors"]["tissue_hires_scalef"]

    coords = adata.obsm["spatial"]  # columns: (col, row) in full-res pixels
    coord_df = pd.DataFrame(
        {
            "pixel_x": coords[:, 0] * scalef,
            "pixel_y": coords[:, 1] * scalef,
        },
        index=adata.obs_names,
    )
    return img, coord_df, lib


def run(
    input_path: str,
    output_dir: str,
    model_dir: str,
    spatial_dir: Optional[str] = None,
    housekeeping_genes_path: Optional[str] = None,
    library_id: Optional[str] = None,
    patch_size: int = 16,
    device: str = "cuda",
) -> AnnData:
    """Run Loki text (always) and image (when spatial info available) embedding.

    Args:
        input_path: Path to input .h5ad.
        output_dir: Directory for output files.
        model_dir: Directory containing ``checkpoint.pt`` (OmiCLIP weights).
        spatial_dir: Optional Visium ``spatial/`` folder; used when the h5ad
            does not already carry spatial metadata.
        housekeeping_genes_path: Optional CSV with a ``genesymbol`` column;
            those genes are excluded from the top-50 selection.
        library_id: Library key under ``adata.uns['spatial']`` (default: first).
        patch_size: H&E patch side length in pixels (post-scaling).
        device: "cuda" or "cpu".

    Returns:
        AnnData with embeddings in ``obsm['X_loki_text']`` and, when applicable,
        ``obsm['X_loki_image']``.
    """
    # sc.read_h5ad may emit "Variable names are not unique" UserWarning when the
    # input H5AD has duplicate gene symbols; var_names_make_unique() below
    # resolves it. The warning is harmless and cannot be suppressed at the source.
    adata = sc.read_h5ad(input_path)
    adata.var_names_make_unique()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Detect image availability up front (this may rewrite adata.obs ordering).
    spatial_resolved = _resolve_spatial(adata, spatial_dir, library_id)
    if spatial_resolved is None:
        print("[loki] No spatial metadata found; running text-only.")

    # Single model load shared by both modalities.
    model, preprocess, tokenizer = load_model(str(Path(model_dir) / "checkpoint.pt"), device)

    # --- Text path -------------------------------------------------------
    if housekeeping_genes_path:
        hk_df = pd.read_csv(housekeeping_genes_path)
    else:
        hk_df = pd.DataFrame({"genesymbol": []})

    text_df = generate_gene_df(
        adata, hk_df, todense=sp.issparse(adata.X)
    )
    text_emb = encode_texts(model, tokenizer, text_df["label"].tolist(), device)
    text_emb = text_emb.cpu().numpy()
    adata.obsm["X_loki_text"] = text_emb

    pd.DataFrame(text_emb, index=adata.obs_names).to_csv(out / "embeddings_text.csv")
    pd.DataFrame(text_emb, index=adata.obs_names).to_csv(out / "embeddings_text.tsv", sep="\t")
    np.save(out / "embeddings_text.npy", text_emb)

    # --- Image path ------------------------------------------------------
    if spatial_resolved is not None:
        img_array, coord_df, _ = spatial_resolved
        patch_dir = out / "patches"
        segment_patches(img_array, coord_df, str(patch_dir), height=patch_size, width=patch_size)

        valid_paths, valid_idx = [], []
        for i, spot in enumerate(adata.obs_names):
            p = patch_dir / f"{spot}_hires.png"
            if p.exists():
                valid_paths.append(str(p))
                valid_idx.append(i)
        n_dropped = len(adata.obs_names) - len(valid_paths)
        if n_dropped:
            print(f"[loki] {n_dropped} patches were out-of-range and will be NaN.")

        img_emb_valid = encode_images(model, preprocess, valid_paths, device).cpu().numpy()
        img_emb = np.full((adata.n_obs, img_emb_valid.shape[1]), np.nan, dtype=np.float32)
        img_emb[valid_idx] = img_emb_valid
        adata.obsm["X_loki_image"] = img_emb

        pd.DataFrame(img_emb, index=adata.obs_names).to_csv(out / "embeddings_image.csv")
        pd.DataFrame(img_emb, index=adata.obs_names).to_csv(out / "embeddings_image.tsv", sep="\t")
        np.save(out / "embeddings_image.npy", img_emb)

    adata.write_h5ad(out / "embeddings.h5ad")

    print(f"Saved Loki embeddings to {out}/")
    print(f"  text: {text_emb.shape}")
    if spatial_resolved is not None:
        print(f"  image: {adata.obsm['X_loki_image'].shape}")

    return adata
