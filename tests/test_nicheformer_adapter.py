"""Tests for the Nicheformer adapter — preprocessing, tokenization, model, and
end-to-end embedding extraction.

Unit tests use synthetic data and random weights.  End-to-end tests that
require the converted checkpoint are gated behind ``requires_nicheformer_weights``.
"""

import anndata
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch

from tests.conftest import (
    NICHEFORMER_MODEL_DIR,
    SAMPLE_H5AD,
    requires_nicheformer_weights,
    requires_sample_data,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adata(n_cells=20, n_genes=100):
    """Synthetic AnnData with positive counts."""
    rng = np.random.default_rng(42)
    X = sp.csr_matrix(rng.random((n_cells, n_genes), dtype=np.float32) * 100)
    gene_names = [f"GENE{i}" for i in range(n_genes)]
    return anndata.AnnData(X=X, var=pd.DataFrame(index=gene_names))


def _make_dense_adata(n_cells=10, n_genes=50):
    rng = np.random.default_rng(7)
    X = rng.random((n_cells, n_genes), dtype=np.float32) * 100
    gene_names = [f"GENE{i}" for i in range(n_genes)]
    return anndata.AnnData(X=X, var=pd.DataFrame(index=gene_names))


# ---------------------------------------------------------------------------
# sf_normalize
# ---------------------------------------------------------------------------

class TestSfNormalize:
    def test_output_sums_to_10000_sparse(self):
        from src.models.nicheformer.preprocess import sf_normalize

        ad = _make_adata(n_cells=5, n_genes=50)
        normed = sf_normalize(ad.X)
        row_sums = np.array(normed.sum(axis=1)).ravel()
        np.testing.assert_allclose(row_sums, 10_000.0, rtol=1e-5)

    def test_output_sums_to_10000_dense(self):
        from src.models.nicheformer.preprocess import sf_normalize

        ad = _make_dense_adata()
        normed = sf_normalize(ad.X)
        row_sums = normed.sum(axis=1)
        np.testing.assert_allclose(row_sums, 10_000.0, rtol=1e-5)

    def test_zero_row_no_error(self):
        from src.models.nicheformer.preprocess import sf_normalize

        X = np.zeros((3, 5), dtype=np.float32)
        X[1, :] = [1, 2, 3, 4, 5]
        normed = sf_normalize(X)
        assert normed[0].sum() == 0.0
        np.testing.assert_allclose(normed[1].sum(), 10_000.0, rtol=1e-5)


# ---------------------------------------------------------------------------
# tokenize_data
# ---------------------------------------------------------------------------

class TestTokenizeData:
    def test_output_shape(self):
        from src.models.nicheformer.preprocess import tokenize_data

        rng = np.random.default_rng(0)
        X = rng.random((8, 50), dtype=np.float32) * 10
        med = rng.random(50, dtype=np.float64) + 0.1
        tokens = tokenize_data(X, med, max_seq_len=4096)
        assert tokens.shape == (8, 4096)
        assert tokens.dtype == np.int32

    def test_token_offset(self):
        from src.models.nicheformer.preprocess import tokenize_data

        rng = np.random.default_rng(1)
        X = rng.random((4, 20), dtype=np.float32) * 10
        med = np.ones(20, dtype=np.float64)
        tokens = tokenize_data(X, med, max_seq_len=100)
        # All non-padding tokens must be >= aux_tokens (30)
        nonzero = tokens[tokens != 0]
        assert (nonzero >= 30).all()

    def test_padding_is_zero(self):
        from src.models.nicheformer.preprocess import tokenize_data

        # 3 non-zero genes → at most 3 tokens, rest should be 0
        X = np.zeros((2, 100), dtype=np.float32)
        X[0, 0] = 5.0
        X[0, 1] = 3.0
        X[0, 2] = 1.0
        med = np.ones(100, dtype=np.float64)
        tokens = tokenize_data(X, med, max_seq_len=50)
        assert (tokens[0, 3:] == 0).all()
        assert (tokens[1, :] == 0).all()  # all-zero cell


# ---------------------------------------------------------------------------
# complete_masking
# ---------------------------------------------------------------------------

class TestCompleteMasking:
    def test_no_masking_at_p_zero(self):
        from src.models.nicheformer.preprocess import complete_masking

        X = torch.tensor([[30, 35, 40, 0, 0]], dtype=torch.int64)
        batch = {"X": X.clone()}
        out = complete_masking(batch, 0.0, 20345)

        # Padding 0s become 1s in X
        expected_x = torch.tensor([[30, 35, 40, 1, 1]], dtype=torch.int64)
        torch.testing.assert_close(out["X"], expected_x)
        # masked_indices should equal remapped X (no masking at p=0)
        torch.testing.assert_close(out["masked_indices"], expected_x)

    def test_attention_mask_marks_padding(self):
        from src.models.nicheformer.preprocess import complete_masking

        X = torch.tensor([[30, 35, 0, 0]], dtype=torch.int64)
        batch = {"X": X.clone()}
        out = complete_masking(batch, 0.0, 20345)
        expected_mask = torch.tensor([[False, False, True, True]])
        torch.testing.assert_close(out["attention_mask"], expected_mask)


# ---------------------------------------------------------------------------
# NicheformerDataset
# ---------------------------------------------------------------------------

class TestNicheformerDataset:
    def test_length_matches_adata(self):
        from src.models.nicheformer.preprocess import NicheformerDataset

        ad = _make_adata(n_cells=12, n_genes=30)
        tech_mean = np.ones(30, dtype=np.float64)
        ds = NicheformerDataset(ad, tech_mean, max_seq_len=100)
        assert len(ds) == 12

    def test_item_has_X_key(self):
        from src.models.nicheformer.preprocess import NicheformerDataset

        ad = _make_adata(n_cells=5, n_genes=20)
        tech_mean = np.ones(20, dtype=np.float64)
        ds = NicheformerDataset(ad, tech_mean, max_seq_len=50)
        item = ds[0]
        assert "X" in item
        assert item["X"].shape == (50,)
        assert item["X"].dtype in (torch.int32, torch.int64)


# ---------------------------------------------------------------------------
# Model construction and forward pass (random weights)
# ---------------------------------------------------------------------------

class TestNicheformerModel:
    def test_model_construction(self):
        from src.models.nicheformer.model import NicheformerInference

        model = NicheformerInference(
            dim_model=64, nheads=4, dim_feedforward=128, nlayers=2,
            n_tokens=100, context_length=50,
        )
        assert model.dim_model == 64
        assert model.nlayers == 2

    def test_get_embeddings_shape(self):
        from src.models.nicheformer.model import NicheformerInference

        model = NicheformerInference(
            dim_model=64, nheads=4, dim_feedforward=128, nlayers=2,
            n_tokens=100, context_length=50,
            specie=False, assay=False, modality=False,
        )
        model.eval()

        # Synthetic batch: 4 cells, seq_len=50, tokens in range [30, 100+5)
        rng = np.random.default_rng(42)
        tokens = rng.integers(30, 105, size=(4, 50), dtype=np.int32)
        tokens[:, 10:] = 0  # pad most positions
        batch = {"X": torch.tensor(tokens, dtype=torch.long)}

        emb = model.get_embeddings(batch)
        assert emb.shape == (4, 64)

    def test_get_embeddings_with_context_tokens(self):
        from src.models.nicheformer.model import NicheformerInference

        model = NicheformerInference(
            dim_model=32, nheads=4, dim_feedforward=64, nlayers=1,
            n_tokens=100, context_length=20,
            specie=True, assay=True, modality=True,
        )
        model.eval()

        batch = {
            "X": torch.zeros(2, 20, dtype=torch.long),
            "specie": torch.tensor([5, 5], dtype=torch.int64),
            "assay": torch.tensor([6, 6], dtype=torch.int64),
            "modality": torch.tensor([7, 7], dtype=torch.int64),
        }
        batch["X"][:, :5] = torch.tensor([30, 31, 32, 33, 34])

        emb = model.get_embeddings(batch)
        assert emb.shape == (2, 32)


# ---------------------------------------------------------------------------
# Gene alignment
# ---------------------------------------------------------------------------

class TestAlignGenes:
    def test_shared_genes_subset(self):
        from src.models.nicheformer.preprocess import align_genes

        ad = anndata.AnnData(
            X=sp.csr_matrix(np.ones((3, 5), dtype=np.float32)),
            var=pd.DataFrame(index=["A", "B", "C", "D", "E"]),
        )
        vocab = ["B", "D", "F", "A"]
        tech_mean = np.array([1.0, 2.0, 3.0, 4.0])

        aligned_ad, aligned_mean = align_genes(ad, vocab, tech_mean)
        assert list(aligned_ad.var_names) == ["B", "D", "A"]
        np.testing.assert_array_equal(aligned_mean, [1.0, 2.0, 4.0])

    def test_no_overlap_raises(self):
        from src.models.nicheformer.preprocess import align_genes

        ad = anndata.AnnData(
            X=sp.csr_matrix(np.ones((2, 3), dtype=np.float32)),
            var=pd.DataFrame(index=["X", "Y", "Z"]),
        )
        with pytest.raises(ValueError, match="No input genes match"):
            align_genes(ad, ["A", "B"], np.array([1.0, 2.0]))


# ---------------------------------------------------------------------------
# End-to-end (requires converted Nicheformer weights)
# ---------------------------------------------------------------------------

@requires_nicheformer_weights
@requires_sample_data
class TestNicheformerAdapterReal:
    def test_run_produces_embeddings(self, tmp_path):
        from src.adapters.nicheformer import run

        out_dir = tmp_path / "out"
        adata = run(
            input_path=str(SAMPLE_H5AD),
            output_dir=str(out_dir),
            model_dir=str(NICHEFORMER_MODEL_DIR),
            technology="merfish",
            batch_size=32,
            device="cpu",
        )
        assert "X_nicheformer" in adata.obsm
        assert adata.obsm["X_nicheformer"].shape[1] == 512
        assert (out_dir / "embeddings.npy").exists()
        assert (out_dir / "embeddings.h5ad").exists()
        assert (out_dir / "embeddings.csv").exists()
        assert (out_dir / "embeddings.tsv").exists()
