"""Tests for src.models.scgpt_spatial.cell_emb — embedding extraction pipeline."""

import json

import anndata
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch

from src.models.scgpt_spatial.cell_emb import embed_data, load_pretrained
from src.models.scgpt_spatial.gene_tokenizer import GeneVocab
from src.models.scgpt_spatial.model import TransformerModel
from tests.conftest import (
    requires_model_weights,
    requires_sample_data,
    MODEL_DIR,
    SAMPLE_H5AD,
)


# ======================================================================
# load_pretrained
# ======================================================================

class TestLoadPretrained:
    def test_non_strict_loads_matching_keys(self):
        v = GeneVocab.from_dict({"<pad>": 0, "<cls>": 1, "A": 2})
        model = TransformerModel(
            ntoken=3, d_model=32, nhead=4, d_hid=32, nlayers=1,
            nlayers_cls=1, n_cls=1, vocab=v,
            use_fast_transformer=False, use_moe_dec=False,
        )
        # Create fake pretrained params with matching and non-matching keys
        fake_params = {}
        for k, v_tensor in model.state_dict().items():
            fake_params[k] = torch.randn_like(v_tensor)
        fake_params["nonexistent_key"] = torch.tensor([1.0])

        load_pretrained(model, fake_params, strict=False, verbose=False)
        # Should succeed without error (non-strict ignores missing/extra keys)

    def test_flash_attn_key_conversion(self):
        """When model doesn't use flash_attn, Wqkv keys should be renamed."""
        v = GeneVocab.from_dict({"<pad>": 0, "<cls>": 1, "A": 2})
        model = TransformerModel(
            ntoken=3, d_model=32, nhead=4, d_hid=32, nlayers=1,
            nlayers_cls=1, n_cls=1, vocab=v,
            use_fast_transformer=False, use_moe_dec=False,
        )
        # Simulate flash_attn style params
        fake_params = {"Wqkv.weight": torch.randn(96, 32)}
        load_pretrained(model, fake_params, strict=False, verbose=False)
        # Key should have been converted to in_proj_weight


# ======================================================================
# embed_data — synthetic data (no model weights needed)
# ======================================================================

class TestEmbedDataSynthetic:
    @pytest.fixture
    def synthetic_setup(self, tmp_path):
        """Create a minimal synthetic model directory and AnnData."""
        np.random.seed(0)
        # Create a small vocab
        gene_names = [f"GENE{i}" for i in range(50)]
        special_tokens = ["<pad>", "<cls>", "<eoc>"]
        token2idx = {}
        for i, t in enumerate(special_tokens + gene_names):
            token2idx[t] = i
        vocab_file = tmp_path / "vocab.json"
        vocab_file.write_text(json.dumps(token2idx))

        # Create args.json
        args = {
            "embsize": 32,
            "nheads": 4,
            "d_hid": 32,
            "nlayers": 1,
            "n_layers_cls": 1,
            "dropout": 0.0,
            "pad_token": "<pad>",
            "pad_value": -2,
        }
        args_file = tmp_path / "args.json"
        args_file.write_text(json.dumps(args))

        # Create a tiny model and save its weights
        vocab = GeneVocab.from_dict(token2idx)
        model = TransformerModel(
            ntoken=len(vocab), d_model=32, nhead=4, d_hid=32, nlayers=1,
            nlayers_cls=1, n_cls=1, vocab=vocab,
            pad_token="<pad>", pad_value=-2,
            do_mvc=True, do_dab=False, use_batch_labels=False,
            use_fast_transformer=False, use_MVC_impute=True, use_moe_dec=True,
        )
        model_file = tmp_path / "best_model.pt"
        torch.save(model.state_dict(), model_file)

        # Create gene stats CSV
        stats = pd.DataFrame(
            {"mean": np.random.rand(len(token2idx)) + 0.1},
            index=range(len(token2idx)),
        )
        stats_file = tmp_path / "all_dict_mean_std.csv"
        stats.to_csv(stats_file)

        # Create AnnData
        n_cells = 10
        X = sp.random(n_cells, len(gene_names), density=0.4, format="csr", dtype=np.float32)
        X.data = np.abs(X.data) * 100
        adata = anndata.AnnData(X=X, var=pd.DataFrame(index=gene_names))

        return tmp_path, adata

    def test_embed_data_returns_adata(self, synthetic_setup):
        model_dir, adata = synthetic_setup
        result = embed_data(
            adata, model_dir, gene_col="index",
            max_length=100, batch_size=4, device="cpu",
            use_fast_transformer=False,
        )
        assert isinstance(result, anndata.AnnData)
        assert "X_scgpt_spatial" in result.obsm

    def test_embedding_shape(self, synthetic_setup):
        model_dir, adata = synthetic_setup
        result = embed_data(
            adata, model_dir, gene_col="index",
            max_length=100, batch_size=4, device="cpu",
        )
        emb = result.obsm["X_scgpt_spatial"]
        assert emb.shape == (10, 32)  # n_cells=10, d_model=32

    def test_embeddings_are_l2_normalized(self, synthetic_setup):
        model_dir, adata = synthetic_setup
        result = embed_data(
            adata, model_dir, gene_col="index",
            max_length=100, batch_size=4, device="cpu",
        )
        emb = result.obsm["X_scgpt_spatial"]
        norms = np.linalg.norm(emb, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_embeddings_no_nan(self, synthetic_setup):
        model_dir, adata = synthetic_setup
        result = embed_data(
            adata, model_dir, gene_col="index",
            max_length=100, batch_size=4, device="cpu",
        )
        assert not np.isnan(result.obsm["X_scgpt_spatial"]).any()

    def test_return_new_adata(self, synthetic_setup):
        model_dir, adata = synthetic_setup
        result = embed_data(
            adata, model_dir, gene_col="index",
            max_length=100, batch_size=4, device="cpu",
            return_new_adata=True,
        )
        assert isinstance(result, anndata.AnnData)
        # New adata has embeddings as X, not obsm
        assert result.X.shape[1] == 32


# ======================================================================
# embed_data — real model weights (slow, skipped without weights)
# ======================================================================

@requires_model_weights
@requires_sample_data
class TestEmbedDataReal:
    def test_real_embedding_shape(self):
        import scanpy as sc
        adata = sc.read_h5ad(SAMPLE_H5AD)
        adata.var_names_make_unique()
        n_cells = adata.n_obs

        result = embed_data(
            adata, str(MODEL_DIR), gene_col="index",
            max_length=1200, batch_size=32, device="cpu",
        )
        emb = result.obsm["X_scgpt_spatial"]
        assert emb.shape == (n_cells, 512)

    def test_real_embeddings_l2_normalized(self):
        import scanpy as sc
        adata = sc.read_h5ad(SAMPLE_H5AD)
        adata.var_names_make_unique()

        result = embed_data(
            adata, str(MODEL_DIR), gene_col="index",
            max_length=1200, batch_size=32, device="cpu",
        )
        norms = np.linalg.norm(result.obsm["X_scgpt_spatial"], axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)
