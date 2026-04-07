"""scGPT-spatial adapter for the unified embedding framework."""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData

from src.models.scgpt_spatial.cell_emb import embed_data


def run(
    input_path: str,
    output_dir: str,
    model_dir: str,
    gene_col: str = "feature_name",
    max_length: int = 1200,
    batch_size: int = 64,
    device: str = "cuda",
) -> AnnData:
    """
    Run scGPT-spatial embedding extraction and export results.

    Args:
        input_path: Path to input .h5ad file.
        output_dir: Directory for output files.
        model_dir: Path to model weights directory (vocab.json, args.json,
            best_model.pt, all_dict_mean_std.csv).
        gene_col: Column in adata.var containing gene names, or "index".
        max_length: Maximum sequence length for the transformer.
        batch_size: Batch size for inference.
        device: "cuda" or "cpu".

    Returns:
        AnnData with embeddings in adata.obsm["X_scgpt_spatial"].
    """
    adata = sc.read_h5ad(input_path)
    adata.var_names_make_unique()

    # Auto-detect gene_col: if the requested column doesn't exist, use index
    if gene_col != "index" and gene_col not in adata.var.columns:
        print(f"Column '{gene_col}' not found in adata.var. Using var index instead.")
        gene_col = "index"

    adata = embed_data(
        adata_or_file=adata,
        model_dir=model_dir,
        gene_col=gene_col,
        max_length=max_length,
        batch_size=batch_size,
        device=device,
        use_fast_transformer=False,
        return_new_adata=False,
    )

    # Export
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    embeddings = adata.obsm["X_scgpt_spatial"]

    adata.write_h5ad(out / "embeddings.h5ad")
    np.save(out / "embeddings.npy", embeddings)

    df = pd.DataFrame(embeddings, index=adata.obs_names)
    df.to_csv(out / "embeddings.csv")
    df.to_csv(out / "embeddings.tsv", sep="\t")

    print(f"Saved embeddings to {out}/")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Formats: .h5ad, .npy, .csv, .tsv")

    return adata
