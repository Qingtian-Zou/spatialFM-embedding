"""Tests for src.embed — CLI entry point."""

import json
import subprocess
import sys

import anndata
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch

from tests.conftest import (
    requires_model_weights,
    requires_sample_data,
    MODEL_DIR,
    SAMPLE_H5AD,
    PROJECT_ROOT,
)


@pytest.fixture
def cli_synthetic_setup(tmp_path):
    """Set up synthetic model dir + h5ad for CLI tests."""
    from src.models.scgpt_spatial.gene_tokenizer import GeneVocab
    from src.models.scgpt_spatial.model import TransformerModel

    np.random.seed(0)
    gene_names = [f"GENE{i}" for i in range(20)]
    special_tokens = ["<pad>", "<cls>", "<eoc>"]
    token2idx = {t: i for i, t in enumerate(special_tokens + gene_names)}

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "vocab.json").write_text(json.dumps(token2idx))
    (model_dir / "args.json").write_text(json.dumps({
        "embsize": 32, "nheads": 4, "d_hid": 32, "nlayers": 1,
        "n_layers_cls": 1, "dropout": 0.0,
        "pad_token": "<pad>", "pad_value": -2,
    }))

    vocab = GeneVocab.from_dict(token2idx)
    model = TransformerModel(
        ntoken=len(vocab), d_model=32, nhead=4, d_hid=32, nlayers=1,
        nlayers_cls=1, n_cls=1, vocab=vocab,
        pad_token="<pad>", pad_value=-2,
        do_mvc=True, use_fast_transformer=False,
        use_MVC_impute=True, use_moe_dec=True,
    )
    torch.save(model.state_dict(), model_dir / "best_model.pt")

    stats = pd.DataFrame(
        {"mean": np.random.rand(len(token2idx)) + 0.1},
        index=range(len(token2idx)),
    )
    stats.to_csv(model_dir / "all_dict_mean_std.csv")

    # h5ad
    n_cells = 6
    X = sp.random(n_cells, len(gene_names), density=0.4, format="csr", dtype=np.float32)
    X.data = np.abs(X.data) * 100
    adata = anndata.AnnData(X=X, var=pd.DataFrame(index=gene_names))
    h5ad_path = tmp_path / "input.h5ad"
    adata.write_h5ad(h5ad_path)

    return model_dir, h5ad_path


class TestCLI:
    def test_scgpt_model_runs(self, cli_synthetic_setup, tmp_path):
        model_dir, h5ad_path = cli_synthetic_setup
        out_dir = tmp_path / "output"

        result = subprocess.run(
            [
                sys.executable, str(PROJECT_ROOT / "src" / "embed.py"),
                "--model", "scgpt_spatial",
                "--input", str(h5ad_path),
                "--output", str(out_dir),
                "--model-dir", str(model_dir),
                "--device", "cpu",
                "--gene-col", "index",
                "--batch-size", "4",
                "--max-length", "50",
            ],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        assert result.returncode == 0, f"CLI failed:\n{result.stderr}"
        assert (out_dir / "embeddings.npy").exists()
        assert (out_dir / "embeddings.h5ad").exists()
        assert (out_dir / "embeddings.csv").exists()
        assert (out_dir / "embeddings.tsv").exists()

    def test_unimplemented_model_exits_nonzero(self, tmp_path):
        result = subprocess.run(
            [
                sys.executable, str(PROJECT_ROOT / "src" / "embed.py"),
                "--model", "nicheformer",
                "--input", "dummy.h5ad",
                "--output", str(tmp_path),
                "--model-dir", str(tmp_path),
            ],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        assert result.returncode != 0
        assert "not yet implemented" in result.stderr

    def test_invalid_model_exits_nonzero(self, tmp_path):
        result = subprocess.run(
            [
                sys.executable, str(PROJECT_ROOT / "src" / "embed.py"),
                "--model", "invalid_model",
                "--input", "dummy.h5ad",
                "--output", str(tmp_path),
                "--model-dir", str(tmp_path),
            ],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        assert result.returncode != 0

    def test_missing_required_args_exits_nonzero(self):
        result = subprocess.run(
            [
                sys.executable, str(PROJECT_ROOT / "src" / "embed.py"),
            ],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        assert result.returncode != 0

    @requires_model_weights
    @requires_sample_data
    def test_real_data_cli(self, tmp_path):
        out_dir = tmp_path / "output"
        result = subprocess.run(
            [
                sys.executable, str(PROJECT_ROOT / "src" / "embed.py"),
                "--model", "scgpt_spatial",
                "--input", str(SAMPLE_H5AD),
                "--output", str(out_dir),
                "--model-dir", str(MODEL_DIR),
                "--device", "cpu",
                "--gene-col", "index",
                "--batch-size", "16",
            ],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=900,
        )
        assert result.returncode == 0, f"CLI failed:\n{result.stderr}"
        emb = np.load(out_dir / "embeddings.npy")
        assert emb.shape[1] == 512
