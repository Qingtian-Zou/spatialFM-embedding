"""Tests for src.adapters.scgpt_spatial — adapter wrapper and output formats."""

import json

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
)


@pytest.fixture
def synthetic_model_dir(tmp_path):
    """Create a minimal synthetic model directory for adapter tests."""
    from src.models.scgpt_spatial.gene_tokenizer import GeneVocab
    from src.models.scgpt_spatial.model import TransformerModel

    np.random.seed(0)
    gene_names = [f"GENE{i}" for i in range(30)]
    special_tokens = ["<pad>", "<cls>", "<eoc>"]
    token2idx = {t: i for i, t in enumerate(special_tokens + gene_names)}

    (tmp_path / "vocab.json").write_text(json.dumps(token2idx))
    (tmp_path / "args.json").write_text(json.dumps({
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
    torch.save(model.state_dict(), tmp_path / "best_model.pt")

    stats = pd.DataFrame(
        {"mean": np.random.rand(len(token2idx)) + 0.1},
        index=range(len(token2idx)),
    )
    stats.to_csv(tmp_path / "all_dict_mean_std.csv")

    return tmp_path, gene_names


@pytest.fixture
def synthetic_h5ad(tmp_path, synthetic_model_dir):
    """Create a synthetic .h5ad file matching the synthetic model's gene names."""
    _, gene_names = synthetic_model_dir
    n_cells = 8
    X = sp.random(n_cells, len(gene_names), density=0.4, format="csr", dtype=np.float32)
    X.data = np.abs(X.data) * 100
    adata = anndata.AnnData(X=X, var=pd.DataFrame(index=gene_names))
    path = tmp_path / "test_input.h5ad"
    adata.write_h5ad(path)
    return path


class TestAdapterSynthetic:
    def test_run_produces_all_outputs(self, synthetic_model_dir, synthetic_h5ad, tmp_path):
        from src.adapters.scgpt_spatial import run

        model_dir, _ = synthetic_model_dir
        out_dir = tmp_path / "output"

        run(
            input_path=str(synthetic_h5ad),
            output_dir=str(out_dir),
            model_dir=str(model_dir),
            gene_col="index",
            max_length=50,
            batch_size=4,
            device="cpu",
        )

        assert (out_dir / "embeddings.h5ad").exists()
        assert (out_dir / "embeddings.npy").exists()
        assert (out_dir / "embeddings.csv").exists()
        assert (out_dir / "embeddings.tsv").exists()

    def test_npy_output_shape(self, synthetic_model_dir, synthetic_h5ad, tmp_path):
        from src.adapters.scgpt_spatial import run

        model_dir, _ = synthetic_model_dir
        out_dir = tmp_path / "output"

        run(
            input_path=str(synthetic_h5ad),
            output_dir=str(out_dir),
            model_dir=str(model_dir),
            gene_col="index",
            max_length=50,
            batch_size=4,
            device="cpu",
        )

        emb = np.load(out_dir / "embeddings.npy")
        assert emb.shape == (8, 32)  # 8 cells, d_model=32

    def test_csv_output_parseable(self, synthetic_model_dir, synthetic_h5ad, tmp_path):
        from src.adapters.scgpt_spatial import run

        model_dir, _ = synthetic_model_dir
        out_dir = tmp_path / "output"

        run(
            input_path=str(synthetic_h5ad),
            output_dir=str(out_dir),
            model_dir=str(model_dir),
            gene_col="index",
            max_length=50,
            batch_size=4,
            device="cpu",
        )

        df = pd.read_csv(out_dir / "embeddings.csv", index_col=0)
        assert df.shape == (8, 32)

    def test_tsv_output_parseable(self, synthetic_model_dir, synthetic_h5ad, tmp_path):
        from src.adapters.scgpt_spatial import run

        model_dir, _ = synthetic_model_dir
        out_dir = tmp_path / "output"

        run(
            input_path=str(synthetic_h5ad),
            output_dir=str(out_dir),
            model_dir=str(model_dir),
            gene_col="index",
            max_length=50,
            batch_size=4,
            device="cpu",
        )

        df = pd.read_csv(out_dir / "embeddings.tsv", sep="\t", index_col=0)
        assert df.shape == (8, 32)

    def test_h5ad_output_has_obsm(self, synthetic_model_dir, synthetic_h5ad, tmp_path):
        from src.adapters.scgpt_spatial import run
        import scanpy as sc

        model_dir, _ = synthetic_model_dir
        out_dir = tmp_path / "output"

        run(
            input_path=str(synthetic_h5ad),
            output_dir=str(out_dir),
            model_dir=str(model_dir),
            gene_col="index",
            max_length=50,
            batch_size=4,
            device="cpu",
        )

        result = sc.read_h5ad(out_dir / "embeddings.h5ad")
        assert "X_scgpt_spatial" in result.obsm

    def test_gene_col_auto_detection(self, synthetic_model_dir, synthetic_h5ad, tmp_path):
        """When gene_col doesn't exist in var, adapter should fall back to index."""
        from src.adapters.scgpt_spatial import run

        model_dir, _ = synthetic_model_dir
        out_dir = tmp_path / "output"

        # Pass a non-existent gene_col — should auto-detect and use index
        run(
            input_path=str(synthetic_h5ad),
            output_dir=str(out_dir),
            model_dir=str(model_dir),
            gene_col="nonexistent_column",
            max_length=50,
            batch_size=4,
            device="cpu",
        )

        emb = np.load(out_dir / "embeddings.npy")
        assert emb.shape[0] == 8


@requires_model_weights
@requires_sample_data
class TestAdapterReal:
    def test_real_data_all_outputs(self, tmp_path):
        from src.adapters.scgpt_spatial import run

        out_dir = tmp_path / "output"
        result = run(
            input_path=str(SAMPLE_H5AD),
            output_dir=str(out_dir),
            model_dir=str(MODEL_DIR),
            gene_col="index",
            max_length=1200,
            batch_size=32,
            device="cpu",
        )

        assert (out_dir / "embeddings.h5ad").exists()
        assert (out_dir / "embeddings.npy").exists()
        assert (out_dir / "embeddings.csv").exists()
        assert (out_dir / "embeddings.tsv").exists()

        emb = np.load(out_dir / "embeddings.npy")
        assert emb.shape[1] == 512
        norms = np.linalg.norm(emb, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)
