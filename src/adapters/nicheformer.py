"""Nicheformer adapter for the unified embedding framework."""

from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from anndata import AnnData
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.nicheformer.model import NicheformerInference
from src.models.nicheformer.preprocess import NicheformerDataset, align_genes


def run(
    input_path: str,
    output_dir: str,
    model_dir: str,
    technology: str = "dissociated",
    batch_size: int = 64,
    device: str = "cuda",
) -> AnnData:
    """Run Nicheformer embedding extraction and export results.

    Args:
        input_path: Path to input .h5ad file.
        output_dir: Directory for output files.
        model_dir: Path to converted model directory (hparams.json,
            model_state_dict.pt, model.h5ad, *_mean_script.npy).
        technology: Platform for technology normalization. One of
            "cosmx", "dissociated", "iss", "merfish", "xenium".
        batch_size: Batch size for inference.
        device: "cuda" or "cpu".

    Returns:
        AnnData with embeddings in adata.obsm["X_nicheformer"].
    """
    model_dir = Path(model_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- Load input ---
    adata = sc.read_h5ad(input_path)
    adata.var_names_make_unique()

    # --- Load gene vocabulary and technology mean ---
    vocab_adata = anndata.read_h5ad(model_dir / "model.h5ad")
    gene_vocab = list(vocab_adata.var_names)

    mean_path = model_dir / f"{technology}_mean_script.npy"
    if not mean_path.exists():
        available = [p.stem for p in model_dir.glob("*_mean_script.npy")]
        raise FileNotFoundError(
            f"Technology mean not found: {mean_path.name}. "
            f"Available: {available}"
        )
    technology_mean = np.load(mean_path)

    # --- Align genes ---
    adata, technology_mean = align_genes(adata, gene_vocab, technology_mean)

    # --- Load model ---
    model = NicheformerInference.load_from_model_dir(str(model_dir), device)

    # --- Build dataset and dataloader ---
    dataset = NicheformerDataset(adata, technology_mean)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # --- Inference ---
    all_embeddings = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Embedding"):
            batch = {k: v.to(device) for k, v in batch.items()}
            emb = model.get_embeddings(batch)
            all_embeddings.append(emb.cpu().numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)
    adata.obsm["X_nicheformer"] = embeddings

    # --- Export ---
    adata.write_h5ad(out / "embeddings.h5ad")
    np.save(out / "embeddings.npy", embeddings)

    df = pd.DataFrame(embeddings, index=adata.obs_names)
    df.to_csv(out / "embeddings.csv")
    df.to_csv(out / "embeddings.tsv", sep="\t")

    print(f"Saved Nicheformer embeddings to {out}/")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Formats: .h5ad, .npy, .csv, .tsv")

    return adata
