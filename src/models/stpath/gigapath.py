"""Gigapath feature extraction for the STPath workflow.

STPath consumes 1536-d Prov-Gigapath tile-encoder features as one of its inputs.
The upstream STPath repo treats this as an external prerequisite; this module
brings the precomputation step in-process so the STPath CLI can compute features
inline (and cache them) when a precomputed sidecar is not provided.

Three pieces:
- ``resolve_he_inputs``: turn ``--fullres-image`` / ``--spatial-dir`` /
  in-h5ad spatial metadata into a ``(image_uint8_rgb, coord_df, patch_px)``
  triple that the encoder can consume.
- ``compute_gigapath_features``: build the ``vit_giant_patch14_dinov2``
  Gigapath tile encoder, batch-encode per-spot crops, return ``(N, 1536)``
  float32 features.
- ``save_gigapath_h5``: write the sidecar in the schema the STPath adapter
  reads (``embeddings``, ``barcodes``, ``coords``).
"""

import json
from pathlib import Path
from typing import Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from PIL import Image

# PIL refuses very large H&E images by default; lift the cap.
Image.MAX_IMAGE_PIXELS = None

# ImageNet normalisation — Prov-Gigapath was trained with it (see
# references/STPath/stpath/hest_utils/encoder.py:172-178).
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_GIGAPATH_INPUT_PX = 224

_HIRES_QUALITY_WARNING = (
    "[stpath/gigapath] Falling back to the hires PNG. Prov-Gigapath was "
    "trained on ~0.5 mpp pathology tiles, and the Visium hires PNG is much "
    "lower resolution — embedding quality may degrade. Pass --fullres-image "
    "for best results."
)


def _normalize_image_to_uint8_rgb(img: np.ndarray) -> np.ndarray:
    """Coerce an H&E image to ``(H, W, 3) uint8`` regardless of source dtype.

    Mirrors the recipe in ``src/adapters/loki.py:_resolve_spatial`` so the
    STPath sidecar is built from the same pixel values Loki sees.
    """
    if img.dtype in (np.float32, np.float64):
        img = (img * 255.0).clip(0, 255).astype(np.uint8)
    elif img.dtype == np.uint16:
        img = (img / 256).clip(0, 255).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(
            f"Expected an (H, W, 3) RGB image; got shape {img.shape}."
        )
    return img


def _load_visium_folder(
    spatial_dir: Path,
    adata_obs_names: pd.Index,
) -> Tuple[np.ndarray, pd.DataFrame, dict, str]:
    """Load a Visium ``spatial/`` folder. Prefers fullres if present, hires otherwise.

    Returns ``(image, positions_df, scalefactors, image_kind)`` where
    ``positions_df`` is indexed by barcode with columns ``pxl_col_in_fullres``
    and ``pxl_row_in_fullres`` (raw fullres coords; caller scales for hires).
    """
    scalef = json.loads((spatial_dir / "scalefactors_json.json").read_text())

    fullres_candidates = [
        spatial_dir / name
        for name in (
            "tissue_fullres_image.tif",
            "tissue_fullres_image.tiff",
            "tissue_fullres_image.png",
            "tissue_fullres_image.btf",
        )
    ]
    fullres_path = next((p for p in fullres_candidates if p.exists()), None)

    if fullres_path is not None:
        img = np.asarray(Image.open(fullres_path).convert("RGB"))
        image_kind = "fullres"
    else:
        hires_path = spatial_dir / "tissue_hires_image.png"
        if not hires_path.exists():
            raise FileNotFoundError(
                f"No fullres or hires image found in {spatial_dir}."
            )
        img = np.asarray(Image.open(hires_path))
        image_kind = "hires"

    pos_path = spatial_dir / "tissue_positions.csv"
    if pos_path.exists():
        pos = pd.read_csv(pos_path).set_index("barcode")
    else:
        pos_list_path = spatial_dir / "tissue_positions_list.csv"
        if not pos_list_path.exists():
            raise FileNotFoundError(
                f"Neither tissue_positions.csv nor tissue_positions_list.csv "
                f"found in {spatial_dir}."
            )
        pos = pd.read_csv(
            pos_list_path,
            header=None,
            names=[
                "barcode", "in_tissue", "array_row", "array_col",
                "pxl_row_in_fullres", "pxl_col_in_fullres",
            ],
        ).set_index("barcode")

    common = adata_obs_names.intersection(pos.index)
    if len(common) == 0:
        raise ValueError(
            "No shared barcodes between adata and tissue_positions; check "
            "that the spatial folder corresponds to this h5ad."
        )
    pos = pos.loc[common]
    return img, pos, scalef, image_kind


def resolve_he_inputs(
    adata: AnnData,
    spatial_dir: Optional[str],
    fullres_image: Optional[str],
    library_id: Optional[str],
    patch_px: Optional[int],
) -> Tuple[np.ndarray, pd.DataFrame, int, str]:
    """Resolve ``(image, coord_df, patch_px, source_tag)`` for Gigapath encoding.

    Source priority:

    1. ``fullres_image`` → load from disk; coords from ``adata.obsm['spatial']``.
       Requires either ``adata.uns['spatial'][lib]['scalefactors']`` or an
       explicit ``patch_px`` to know the crop size.
    2. ``spatial_dir`` → Visium folder. Uses ``tissue_fullres_image.*`` if
       present, else falls back to ``tissue_hires_image.png`` with a warning.
    3. In-h5ad ``adata.uns['spatial'][lib]['images']`` → uses ``fullres`` if
       cached, else ``hires`` with a warning.

    The returned ``coord_df`` is indexed by barcode; columns ``pixel_x`` (col)
    and ``pixel_y`` (row) are in the same pixel space as the returned image.
    """
    if "spatial" not in adata.obsm:
        raise ValueError(
            "STPath requires spatial coordinates in adata.obsm['spatial']."
        )
    spatial = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    if spatial.ndim != 2 or spatial.shape[1] != 2:
        raise ValueError(
            f"adata.obsm['spatial'] must have shape (n_obs, 2); got {spatial.shape}."
        )

    # Helper to derive patch_px from scalefactors when not given.
    def _resolve_patch_px(scalefactors: Optional[dict], image_kind: str) -> int:
        if patch_px is not None:
            return int(patch_px)
        if scalefactors is None or "spot_diameter_fullres" not in scalefactors:
            raise ValueError(
                "Cannot infer crop size: scalefactors are absent. Pass "
                "--patch-px to set the crop side length explicitly."
            )
        diameter = float(scalefactors["spot_diameter_fullres"])
        if image_kind == "hires":
            diameter *= float(scalefactors.get("tissue_hires_scalef", 1.0))
        return max(1, int(round(diameter)))

    # ---- 1. Explicit fullres image ----
    if fullres_image is not None:
        img = np.asarray(Image.open(fullres_image).convert("RGB"))
        scalefactors = None
        # Try to pull scalefactors from in-h5ad uns for patch_px when patch_px
        # is not supplied. If patch_px is given, scalefactors are unnecessary.
        if patch_px is None and "spatial" in adata.uns:
            lib = library_id or next(iter(adata.uns["spatial"]))
            scalefactors = adata.uns["spatial"].get(lib, {}).get("scalefactors")
        resolved_patch_px = _resolve_patch_px(scalefactors, "fullres")
        coord_df = pd.DataFrame(
            {"pixel_x": spatial[:, 0], "pixel_y": spatial[:, 1]},
            index=adata.obs_names,
        )
        return _normalize_image_to_uint8_rgb(img), coord_df, resolved_patch_px, "fullres-image"

    # ---- 2. Visium spatial/ folder ----
    if spatial_dir is not None:
        img, pos, scalefactors, image_kind = _load_visium_folder(
            Path(spatial_dir), adata.obs_names
        )
        if image_kind == "hires":
            print(_HIRES_QUALITY_WARNING)
        scalef_hires = float(scalefactors.get("tissue_hires_scalef", 1.0))
        coord_df = pd.DataFrame(
            {
                "pixel_x": pos["pxl_col_in_fullres"].astype(float)
                * (scalef_hires if image_kind == "hires" else 1.0),
                "pixel_y": pos["pxl_row_in_fullres"].astype(float)
                * (scalef_hires if image_kind == "hires" else 1.0),
            },
            index=pos.index,
        )
        resolved_patch_px = _resolve_patch_px(scalefactors, image_kind)
        return (
            _normalize_image_to_uint8_rgb(img),
            coord_df,
            resolved_patch_px,
            f"spatial-dir/{image_kind}",
        )

    # ---- 3. In-h5ad spatial metadata ----
    if "spatial" in adata.uns and adata.uns["spatial"]:
        lib = library_id or next(iter(adata.uns["spatial"]))
        entry = adata.uns["spatial"][lib]
        images = entry.get("images", {})
        if "fullres" in images:
            img = np.asarray(images["fullres"])
            image_kind = "fullres"
        elif "hires" in images:
            img = np.asarray(images["hires"])
            image_kind = "hires"
            print(_HIRES_QUALITY_WARNING)
        else:
            raise ValueError(
                f"adata.uns['spatial'][{lib!r}]['images'] has neither 'fullres' "
                f"nor 'hires'."
            )
        scalefactors = entry.get("scalefactors")
        scalef_hires = float((scalefactors or {}).get("tissue_hires_scalef", 1.0))
        scale = scalef_hires if image_kind == "hires" else 1.0
        coord_df = pd.DataFrame(
            {"pixel_x": spatial[:, 0] * scale, "pixel_y": spatial[:, 1] * scale},
            index=adata.obs_names,
        )
        resolved_patch_px = _resolve_patch_px(scalefactors, image_kind)
        return (
            _normalize_image_to_uint8_rgb(img),
            coord_df,
            resolved_patch_px,
            f"adata.uns/{image_kind}",
        )

    raise ValueError(
        "Cannot locate an H&E image. Pass --fullres-image, or --spatial-dir, "
        "or write an image into adata.uns['spatial'][lib]['images']."
    )


class _SpotPatchDataset(torch.utils.data.Dataset):
    """Iterates spots and yields preprocessed 224x224 tensors for the encoder."""

    def __init__(
        self,
        image: np.ndarray,
        coord_df: pd.DataFrame,
        patch_px: int,
        transform,
    ):
        self._image = image
        self._coords = coord_df[["pixel_x", "pixel_y"]].to_numpy(dtype=np.float64)
        self._patch_px = int(patch_px)
        self._transform = transform
        self._H, self._W = image.shape[:2]

    def __len__(self) -> int:
        return len(self._coords)

    def __getitem__(self, idx: int) -> torch.Tensor:
        x, y = self._coords[idx]
        half = self._patch_px // 2
        x0 = int(round(x)) - half
        y0 = int(round(y)) - half
        x1 = x0 + self._patch_px
        y1 = y0 + self._patch_px
        x0c, y0c = max(x0, 0), max(y0, 0)
        x1c, y1c = min(x1, self._W), min(y1, self._H)
        patch = np.zeros((self._patch_px, self._patch_px, 3), dtype=np.uint8)
        if x1c > x0c and y1c > y0c:
            patch[y0c - y0 : y1c - y0, x0c - x0 : x1c - x0] = self._image[y0c:y1c, x0c:x1c]
        return self._transform(Image.fromarray(patch))


def _build_gigapath_encoder(device: torch.device) -> Tuple[torch.nn.Module, "torchvision.transforms.Compose"]:
    """Construct the ``vit_giant_patch14_dinov2`` tile encoder + eval transform.

    Mirrors ``references/STPath/stpath/hest_utils/encoder.py:162-179`` exactly,
    pulling weights from ``hf_hub:prov-gigapath/prov-gigapath``.
    """
    try:
        import timm
        from torchvision import transforms
    except ImportError as e:
        raise ImportError(
            "Computing Gigapath features requires `timm` and `torchvision`. "
            "Install via: `pip install timm>=1.0 torchvision`."
        ) from e

    model = timm.create_model(
        "vit_giant_patch14_dinov2",
        pretrained=False,
        img_size=_GIGAPATH_INPUT_PX,
        in_chans=3,
        patch_size=16,
        embed_dim=1536,
        depth=40,
        num_heads=24,
        init_values=1e-5,
        mlp_ratio=5.33334,
        num_classes=0,
    )
    try:
        from huggingface_hub import hf_hub_download

        ckpt_path = hf_hub_download("prov-gigapath/prov-gigapath", "pytorch_model.bin")
    except Exception as e:
        raise RuntimeError(
            "Failed to download Prov-Gigapath weights from Hugging Face. "
            "The repo is gated: accept the EULA at "
            "https://huggingface.co/prov-gigapath/prov-gigapath and run "
            "`huggingface-cli login` (or set HF_TOKEN). "
            f"Underlying error: {e}"
        ) from e
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)

    transform = transforms.Compose(
        [
            transforms.Resize(_GIGAPATH_INPUT_PX),
            transforms.CenterCrop(_GIGAPATH_INPUT_PX),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]
    )
    return model, transform


def compute_gigapath_features(
    image: np.ndarray,
    coord_df: pd.DataFrame,
    patch_px: int,
    device: torch.device,
    batch_size: int = 32,
    num_workers: int = 2,
    precision: str = "fp32",
) -> Tuple[np.ndarray, list]:
    """Run the Gigapath tile encoder over per-spot crops.

    ``coord_df`` is indexed by barcode. The returned embeddings preserve that
    order; barcodes are returned alongside as a parallel list.
    """
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(
            f"image must be (H, W, 3) uint8 RGB; got shape {image.shape} "
            f"dtype {image.dtype}."
        )
    if precision not in ("fp32", "fp16"):
        raise ValueError(f"precision must be 'fp32' or 'fp16'; got {precision!r}.")

    model, transform = _build_gigapath_encoder(device)

    H, W = image.shape[:2]
    n_spots = len(coord_df)
    print(
        f"[stpath/gigapath] Encoding {n_spots} spots from a "
        f"{H}x{W} image; patch_px={patch_px}, batch_size={batch_size}, "
        f"precision={precision}, device={device}."
    )

    dataset = _SpotPatchDataset(image, coord_df, patch_px, transform)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
    )

    use_amp = precision == "fp16" and device.type == "cuda"
    out_chunks = []
    with torch.inference_mode():
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    feats = model(batch)
            else:
                feats = model(batch)
            out_chunks.append(feats.float().cpu())

    embeddings = torch.cat(out_chunks, dim=0).numpy().astype(np.float32)
    if embeddings.shape != (n_spots, 1536):
        raise RuntimeError(
            f"Unexpected Gigapath output shape {embeddings.shape}; expected "
            f"({n_spots}, 1536)."
        )
    barcodes = list(coord_df.index.astype(str))
    return embeddings, barcodes


def save_gigapath_h5(
    path: str,
    embeddings: np.ndarray,
    barcodes: list,
    coords: np.ndarray,
) -> None:
    """Write a sidecar in the schema the STPath adapter consumes.

    Datasets:
    - ``embeddings``: (N, 1536) float32, gzip-4
    - ``barcodes``:   (N,) UTF-8 strings
    - ``coords``:     (N, 2) float32 (debug-only; adapter ignores it)
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset(
            "embeddings",
            data=np.ascontiguousarray(embeddings, dtype=np.float32),
            compression="gzip",
            compression_opts=4,
        )
        f.create_dataset(
            "barcodes",
            data=np.asarray(list(barcodes), dtype=object),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
        f.create_dataset(
            "coords",
            data=np.ascontiguousarray(coords, dtype=np.float32),
        )
