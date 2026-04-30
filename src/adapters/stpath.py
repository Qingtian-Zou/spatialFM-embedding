"""STPath adapter for the unified embedding framework.

Runs STPath's full-context imputation forward pass (every spot supplies its own
log1p expression as context) and surfaces the 512-d backbone hidden state as a
unified multi-modal embedding fusing H&E morphology, transcriptome, spatial
coordinates, and organ/tech priors.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse
import torch
from anndata import AnnData

from src.models.stpath.gigapath import (
    compute_gigapath_features,
    resolve_he_inputs,
    save_gigapath_h5,
)
from src.models.stpath.inference import STPathInference
from src.models.stpath.hest_utils import read_assets_from_h5


_STARTUP_NOTICE = (
    "[stpath] X_stpath is a unified multi-modal embedding (H&E morphology + "
    "transcriptome + spatial coordinates + organ/tech). It is NOT a cell-state "
    "embedding and is NOT directly comparable with X_scgpt_spatial / "
    "X_nicheformer / X_loki_*. The all-context forward pass is OOD relative to "
    "STPath's training-time mask ratios in [0.1, 0.95] — validate on a "
    "downstream task before relying on it."
)

_MIN_GENES_MAPPED = 100


def _resolve_device(device: str) -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
        print("[stpath] CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device)


def _to_dense_float32(x) -> np.ndarray:
    if scipy.sparse.issparse(x):
        x = x.toarray()
    return np.asarray(x, dtype=np.float32)


def _load_or_compute_gigapath(
    *,
    adata: AnnData,
    output_dir: str,
    gigapath_h5: Optional[str],
    gigapath_cache: Optional[str],
    spatial_dir: Optional[str],
    fullres_image: Optional[str],
    library_id: Optional[str],
    patch_px: Optional[int],
    gigapath_batch_size: int,
    gigapath_precision: str,
    device: torch.device,
) -> dict:
    """Return a Gigapath ``assets`` dict, loading from disk or computing inline.

    Resolution order:
    1. If ``gigapath_h5`` is given, load it (fail fast if missing).
    2. Else if a cache exists at ``gigapath_cache`` (default
       ``<output_dir>/gigapath_features.h5``), load it.
    3. Else compute features in-process and write the cache for next time.
    """
    if gigapath_h5:
        gigapath_path = Path(gigapath_h5)
        if not gigapath_path.exists():
            raise FileNotFoundError(f"Gigapath sidecar not found: {gigapath_path}")
        print(f"[stpath] Using Gigapath sidecar: {gigapath_path}")
        assets, _ = read_assets_from_h5(str(gigapath_path))
        return assets

    cache_path = Path(gigapath_cache) if gigapath_cache else Path(output_dir) / "gigapath_features.h5"
    if cache_path.exists():
        print(f"[stpath] Reusing cached Gigapath features from {cache_path}")
        assets, _ = read_assets_from_h5(str(cache_path))
        return assets

    print(
        "[stpath] No Gigapath sidecar provided; computing features inline "
        "(this is a one-time cost, then cached)."
    )
    image, coord_df, resolved_patch_px, source_tag = resolve_he_inputs(
        adata=adata,
        spatial_dir=spatial_dir,
        fullres_image=fullres_image,
        library_id=library_id,
        patch_px=patch_px,
    )
    print(f"[stpath] Gigapath input source: {source_tag}; patch_px={resolved_patch_px}")
    embeddings, barcodes = compute_gigapath_features(
        image=image,
        coord_df=coord_df,
        patch_px=resolved_patch_px,
        device=device,
        batch_size=gigapath_batch_size,
        precision=gigapath_precision,
    )
    coords = coord_df.loc[barcodes, ["pixel_x", "pixel_y"]].to_numpy(dtype=np.float32)
    save_gigapath_h5(str(cache_path), embeddings, barcodes, coords)
    print(f"[stpath] Cached Gigapath features to {cache_path}")
    return {
        "embeddings": embeddings,
        "barcodes": np.asarray(barcodes, dtype=object),
        "coords": coords,
    }


def run(
    input_path: str,
    output_dir: str,
    model_dir: str,
    gigapath_h5: Optional[str] = None,
    spatial_dir: Optional[str] = None,
    fullres_image: Optional[str] = None,
    library_id: Optional[str] = None,
    patch_px: Optional[int] = None,
    gigapath_cache: Optional[str] = None,
    gigapath_batch_size: int = 32,
    gigapath_precision: str = "fp32",
    organ_type: str = "Others",
    tech_type: Optional[str] = None,
    save_imputed_expression: bool = False,
    device: str = "cuda",
) -> AnnData:
    """Run STPath full-context embedding extraction and export results.

    Args:
        input_path: Path to input .h5ad (must have ``obsm['spatial']`` and HGNC
            gene symbols in ``var_names``).
        output_dir: Directory for output files.
        model_dir: Directory with ``stpath.pkl`` and ``symbol2ensembl.json``.
        gigapath_h5: Optional path to a precomputed Gigapath sidecar .h5
            (datasets: ``embeddings`` [n, 1536], ``barcodes`` [n], optional
            ``coords`` [n, 2]). When omitted, the adapter computes Gigapath
            features inline and caches them at ``gigapath_cache`` (default:
            ``<output_dir>/gigapath_features.h5``); subsequent runs reuse the
            cache.
        spatial_dir: Optional Visium ``spatial/`` folder used to resolve the
            H&E image when computing Gigapath features inline.
        fullres_image: Optional path to a full-resolution H&E image. Strongly
            preferred over the Visium hires PNG (Gigapath was trained on
            ~0.5 mpp tiles).
        library_id: Key under ``adata.uns['spatial']`` (default: first found).
        patch_px: Override for the per-spot crop side length in image-pixel
            units. Defaults to ``spot_diameter_fullres`` from the Visium
            scalefactors (scaled when falling back to the hires image).
        gigapath_cache: Optional sidecar path to read/write when ``gigapath_h5``
            is not supplied. Defaults to ``<output_dir>/gigapath_features.h5``.
        gigapath_batch_size: Mini-batch size for Gigapath inference (default 32).
        gigapath_precision: ``"fp32"`` or ``"fp16"`` for inline encoding.

    Expected files in ``model_dir``:
        - ``stfm.pth`` — STFM state dict (~190 MB, name as published on HF).
        - ``symbol2ensembl.json`` — gene symbol to Ensembl ID mapping (38,984 entries).
        organ_type: One of the 25 STPath organ tokens (default ``"Others"``).
        tech_type: One of ``"Spatial Transcriptomics"``, ``"Visium"``,
            ``"Xenium"``, ``"Visium HD"``, or ``None`` (uses pad token).
        save_imputed_expression: If set, also write ``imputed_expression.h5ad``
            with the model's refined log1p expression on STPath's 38,984-gene
            vocabulary (sibling file; not stored as a layer because the column
            count does not match the input AnnData).
        device: ``"cuda"`` or ``"cpu"``.

    Returns:
        Input AnnData with embeddings in ``adata.obsm["X_stpath"]``.
    """
    print(_STARTUP_NOTICE)

    resolved_device = _resolve_device(device)

    adata = sc.read_h5ad(input_path)
    adata.var_names_make_unique()

    # 1) Validate spatial coordinates (PR5 — fail fast).
    if "spatial" not in adata.obsm:
        raise ValueError(
            "STPath requires spatial coordinates in adata.obsm['spatial']. "
            "Got AnnData with obsm keys: " + repr(list(adata.obsm.keys()))
        )
    spatial = np.asarray(adata.obsm["spatial"])
    if spatial.ndim != 2 or spatial.shape[1] != 2:
        raise ValueError(
            f"adata.obsm['spatial'] must have shape (n_obs, 2); got {spatial.shape}."
        )

    # 2) Load Gigapath features and align by barcode.
    assets = _load_or_compute_gigapath(
        adata=adata,
        output_dir=output_dir,
        gigapath_h5=gigapath_h5,
        gigapath_cache=gigapath_cache,
        spatial_dir=spatial_dir,
        fullres_image=fullres_image,
        library_id=library_id,
        patch_px=patch_px,
        gigapath_batch_size=gigapath_batch_size,
        gigapath_precision=gigapath_precision,
        device=resolved_device,
    )
    if "embeddings" not in assets or "barcodes" not in assets:
        raise ValueError(
            f"Gigapath sidecar must contain 'embeddings' and 'barcodes' datasets; "
            f"got keys: {sorted(assets.keys())}"
        )
    gp_embeddings = np.asarray(assets["embeddings"])
    gp_barcodes_raw = np.asarray(assets["barcodes"]).flatten()
    gp_barcodes = np.array([
        b.decode("utf-8") if isinstance(b, (bytes, bytearray)) else str(b)
        for b in gp_barcodes_raw
    ])

    if gp_embeddings.ndim != 2:
        raise ValueError(
            f"Gigapath 'embeddings' must be 2D; got shape {gp_embeddings.shape}."
        )
    if gp_embeddings.shape[0] != len(gp_barcodes):
        raise ValueError(
            "Gigapath 'embeddings' and 'barcodes' have mismatched lengths: "
            f"{gp_embeddings.shape[0]} vs {len(gp_barcodes)}."
        )

    # Subset adata to barcodes present in the gigapath file, preserving gigapath order.
    adata_barcodes = set(adata.obs_names.tolist())
    keep_mask = np.array([b in adata_barcodes for b in gp_barcodes])
    n_aligned = int(keep_mask.sum())
    if n_aligned == 0:
        raise ValueError(
            "No barcodes shared between AnnData and Gigapath sidecar. "
            f"Sample AnnData barcodes: {adata.obs_names[:3].tolist()}; "
            f"sample Gigapath barcodes: {gp_barcodes[:3].tolist()}"
        )
    print(
        f"[stpath] Barcode alignment: {n_aligned}/{len(adata.obs_names)} "
        f"AnnData spots present in Gigapath sidecar (sidecar has {len(gp_barcodes)} entries)."
    )
    aligned_barcodes = gp_barcodes[keep_mask]
    aligned_embeddings = gp_embeddings[keep_mask]
    adata = adata[aligned_barcodes].copy()

    # 3) Instantiate STPath.
    # Upstream Hugging Face artifact is named "stfm.pth"; accept either that or
    # the legacy "stpath.pkl" filename.
    model_dir_path = Path(model_dir)
    weights_path = model_dir_path / "stfm.pth"
    if not weights_path.exists():
        legacy = model_dir_path / "stpath.pkl"
        if legacy.exists():
            weights_path = legacy
        else:
            raise FileNotFoundError(
                f"STPath weights not found. Expected {model_dir_path / 'stfm.pth'} "
                f"(or legacy {legacy})."
            )
    vocab_path = model_dir_path / "symbol2ensembl.json"
    if not vocab_path.exists():
        raise FileNotFoundError(f"STPath gene vocabulary not found: {vocab_path}")
    agent = STPathInference(
        gene_voc_path=str(vocab_path),
        model_weight_path=str(weights_path),
        device=resolved_device,
    )

    # 4) Build full-context inputs.
    coords_t = torch.from_numpy(np.ascontiguousarray(adata.obsm["spatial"])).float().to(resolved_device)
    coords_t = agent._normalize_coords(coords_t)

    img_features_t = torch.from_numpy(np.ascontiguousarray(aligned_embeddings)).float().to(resolved_device)

    counts = _to_dense_float32(adata.X)
    context_gene_exps_t = torch.from_numpy(counts).to(resolved_device)
    context_gene_exps_t = agent._log1p(context_gene_exps_t)

    context_gene_names = adata.var_names.tolist()
    context_gene_ids, valid_ids = agent.tokenizer.ge_tokenizer.symbol2id(
        context_gene_names, return_valid_positions=True
    )

    n_input_genes = len(context_gene_names)
    n_mapped = len(valid_ids)
    n_dropped = n_input_genes - n_mapped
    dropped_sample = []
    if n_dropped > 0:
        valid_set = set(valid_ids)
        dropped_sample = [
            context_gene_names[i] for i in range(n_input_genes) if i not in valid_set
        ][:10]
    print(
        f"[stpath] Gene mapping: {n_mapped}/{n_input_genes} input symbols mapped to "
        f"STPath's 38,984-gene Ensembl vocabulary ({n_dropped} dropped)."
    )
    if dropped_sample:
        print(f"[stpath] Sample of dropped symbols: {dropped_sample}")
    if n_mapped < _MIN_GENES_MAPPED:
        raise ValueError(
            f"Too few genes mapped to STPath's vocabulary "
            f"({n_mapped} < {_MIN_GENES_MAPPED}). Inputs must use HGNC gene "
            f"symbols (the symbol2ensembl.json dictionary is symbol-keyed). "
            f"Sample input symbols: {context_gene_names[:5]}."
        )

    context_gene_exps_t = context_gene_exps_t[:, valid_ids]
    context_gene_ids_t = torch.tensor(context_gene_ids, dtype=torch.long, device=resolved_device)
    n_tokens = agent.tokenizer.ge_tokenizer.n_tokens
    ge_tokens = agent.tokenizer.ge_tokenizer.convert_gene_exp_to_one_hot_tensor(
        n_tokens, context_gene_exps_t, context_gene_ids_t
    )

    # Organ + tech tokens.
    organ_id = agent.tokenizer.organ_tokenizer.encode(organ_type, align_first=True)
    organ_ids_t = torch.full((adata.n_obs,), organ_id, dtype=torch.long, device=resolved_device)
    print(f"[stpath] Organ token: {organ_type!r} -> id {organ_id}")
    if tech_type is None:
        tech_ids_t = torch.full(
            (adata.n_obs,),
            agent.tokenizer.tech_tokenizer.pad_token_id,
            dtype=torch.long,
            device=resolved_device,
        )
        print("[stpath] Tech token: <pad> (no --tech-type provided)")
    else:
        tech_id = agent.tokenizer.tech_tokenizer.encode(tech_type, align_first=True)
        tech_ids_t = torch.full((adata.n_obs,), tech_id, dtype=torch.long, device=resolved_device)
        print(f"[stpath] Tech token: {tech_type!r} -> id {tech_id}")

    batch_idx_t = torch.zeros(adata.n_obs, dtype=torch.long, device=resolved_device)

    # 5) Forward pass — capture both predicted expression and hidden state.
    with torch.no_grad():
        predicted_expression, hidden_state = agent.model.prediction_head(
            img_tokens=img_features_t,
            coords=coords_t,
            ge_tokens=ge_tokens,
            batch_idx=batch_idx_t,
            tech_tokens=tech_ids_t,
            organ_tokens=organ_ids_t,
            return_all=True,
        )

    embeddings = hidden_state.cpu().numpy().astype(np.float32)
    adata.obsm["X_stpath"] = embeddings

    # 6) Export embeddings (.h5ad/.npy/.csv/.tsv).
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    adata.write_h5ad(out / "embeddings.h5ad")
    np.save(out / "embeddings.npy", embeddings)

    df = pd.DataFrame(embeddings, index=adata.obs_names)
    df.to_csv(out / "embeddings.csv")
    df.to_csv(out / "embeddings.tsv", sep="\t")

    print(f"[stpath] Saved embeddings to {out}/")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Formats: .h5ad, .npy, .csv, .tsv")

    # 7) Optional: imputed expression (sibling file, not a layer).
    if save_imputed_expression:
        # Drop pad/mask columns (cols 0 and 1) per upstream convention.
        imputed = predicted_expression[:, 2:].cpu().numpy().astype(np.float32)
        gene_names = agent.tokenizer.ge_tokenizer.get_available_genes()
        assert imputed.shape[1] == len(gene_names), (
            f"Imputed expression has {imputed.shape[1]} columns but tokenizer "
            f"reports {len(gene_names)} genes."
        )
        from anndata import AnnData as _AnnData
        imputed_adata = _AnnData(X=imputed)
        imputed_adata.obs_names = adata.obs_names
        imputed_adata.var_names = pd.Index(gene_names)
        imputed_adata.obsm["spatial"] = adata.obsm["spatial"]
        imputed_adata.write_h5ad(out / "imputed_expression.h5ad")
        np.save(out / "imputed_expression.npy", imputed)
        print(f"[stpath] Saved imputed expression to {out / 'imputed_expression.h5ad'} (shape {imputed.shape})")

    return adata
