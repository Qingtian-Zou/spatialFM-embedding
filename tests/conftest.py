"""Shared fixtures for spatialFMs test suite."""

import json
import os
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "scgpt_spatial"
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_H5AD = DATA_DIR / "GSE244084" / "GSM7806336" / "GSM7806336.h5ad"


def _has_model_weights() -> bool:
    return (MODEL_DIR / "best_model.pt").exists()


def _has_sample_data() -> bool:
    return SAMPLE_H5AD.exists()


requires_model_weights = pytest.mark.skipif(
    not _has_model_weights(),
    reason="Model weights not found at models/scgpt_spatial/",
)
requires_sample_data = pytest.mark.skipif(
    not _has_sample_data(),
    reason="Sample data not found at data/GSE244084/GSM7806336/GSM7806336.h5ad",
)


# ---------------------------------------------------------------------------
# Lightweight fixtures (no disk I/O, no model loading)
# ---------------------------------------------------------------------------

@pytest.fixture
def small_gene_list():
    """A short list of gene names for unit tests."""
    return ["TP53", "BRCA1", "EGFR", "MYC", "KRAS"]


@pytest.fixture
def vocab_with_specials(small_gene_list):
    """A GeneVocab built from a gene list with special tokens."""
    from src.models.scgpt_spatial.gene_tokenizer import GeneVocab

    return GeneVocab(
        small_gene_list,
        specials=["<pad>", "<cls>", "<eoc>"],
        special_first=True,
    )


@pytest.fixture
def small_adata():
    """A tiny AnnData (20 cells × 10 genes) with synthetic expression data."""
    np.random.seed(42)
    n_cells, n_genes = 20, 10
    X = sp.random(n_cells, n_genes, density=0.5, format="csr", dtype=np.float32)
    X.data = np.abs(X.data) * 100  # make values positive counts

    gene_names = [f"GENE{i}" for i in range(n_genes)]
    adata = anndata.AnnData(
        X=X,
        var=pd.DataFrame(index=gene_names),
    )
    return adata


@pytest.fixture
def vocab_json_file(tmp_path):
    """Write a small vocab JSON to a temp file and return its path."""
    token2idx = {
        "<pad>": 0,
        "<cls>": 1,
        "<eoc>": 2,
        "TP53": 3,
        "BRCA1": 4,
        "EGFR": 5,
        "MYC": 6,
        "KRAS": 7,
    }
    p = tmp_path / "vocab.json"
    p.write_text(json.dumps(token2idx))
    return p


@pytest.fixture
def model_dir_path():
    """Path to the real model directory (may not exist in CI)."""
    return MODEL_DIR


@pytest.fixture
def sample_h5ad_path():
    """Path to the real sample h5ad file (may not exist in CI)."""
    return SAMPLE_H5AD
